from __future__ import annotations

import threading
import time
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
    Sized,
)
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType, TracebackType
from typing import Any, Generic, Literal, TypeVar, cast

import numpy as np
from numpy.typing import DTypeLike, NDArray

from zcu_tools.progress_bar import make_pbar
from zcu_tools.utils.func_tools import MinIntervalFunc, min_interval

from .base import current_stop_flag
from .repeat import run_with_retries
from .state import T_Cfg, TaskState
from .task import Task, default_raw2signal_fn

T_Raw = TypeVar("T_Raw")
T_Value = TypeVar("T_Value")
T_NestedValue = TypeVar("T_NestedValue")
T_BatchResult = TypeVar("T_BatchResult")
T_Index = TypeVar("T_Index", bound=Hashable)


@dataclass(frozen=True)
class MeasureSnapshot(Generic[T_Cfg]):
    """Read-only session update surface for live plotting callbacks."""

    root_data: Any
    cfg: T_Cfg
    env: Mapping[str, Any]


class MeasureSession(Generic[T_Cfg]):
    """Python-like orchestration frontend over the existing runner leaf task."""

    def __init__(
        self,
        init_cfg: T_Cfg,
        *,
        on_update: Callable[[MeasureSnapshot[T_Cfg]], None] | None = None,
        update_interval: float | None = 0.1,
        stop_flag: threading.Event | None = None,
    ) -> None:
        self.cfg = deepcopy(init_cfg)
        self.env: dict[str, Any] = {}
        self._root_data: Any | None = None
        self._has_unnamed_root_buffer = False
        self._update_interval = update_interval
        self._on_update = min_interval(on_update, update_interval)
        active_stop_flag = current_stop_flag()
        if stop_flag is not None:
            self._stop_flag = stop_flag
        elif active_stop_flag is not None:
            self._stop_flag = active_stop_flag
        else:
            self._stop_flag = threading.Event()

    def __enter__(self) -> MeasureSession[T_Cfg]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        return False

    @property
    def root_data(self) -> Any:
        if self._root_data is None:
            raise ValueError("MeasureSession has no root data; create a buffer first")
        return self._root_data

    def is_stop(self) -> bool:
        return self._stop_flag.is_set()

    def set_stop(self) -> None:
        self._stop_flag.set()

    def buffer(
        self,
        shape: int | Sequence[int],
        *,
        dtype: DTypeLike = np.complex128,
        name: Hashable | None = None,
        on_update: Callable[[NDArray[Any]], None] | None = None,
    ) -> MeasureBuffer[T_Cfg]:
        array = _make_nan_array(shape, dtype)
        self._register_root_buffer(array, name=name)
        return MeasureBuffer(
            session=self,
            owner=self,
            array=array,
            on_update=min_interval(on_update, self._update_interval),
        )

    def scan(
        self, name: str, values: Iterable[T_Value]
    ) -> Iterator[MeasureStep[T_Cfg, int, T_Value]]:
        yield from self._scan(parent=self, name=name, values=values)

    def repeat(
        self, name: str, times: int, interval: float = 0.0
    ) -> Iterator[MeasureStep[T_Cfg, int, int]]:
        yield from self._repeat(parent=self, name=name, times=times, interval=interval)

    def batch(
        self,
        tasks: Mapping[
            Hashable, Callable[[MeasureStep[T_Cfg, Hashable, Hashable]], T_BatchResult]
        ],
        *,
        retry: int = 0,
    ) -> dict[Hashable, T_BatchResult]:
        if retry < 0:
            raise ValueError("retry must be non-negative")

        root = self._ensure_mapping_root()
        results: dict[Hashable, T_BatchResult] = {}
        task_items = list(tasks.items())
        pbar = make_pbar(total=len(task_items), smoothing=0, leave=True)

        try:
            for key, child_fn in task_items:
                if self.is_stop():
                    break

                pbar.set_description(f"Task [{str(key)}]")
                completed = False
                for attempt in range(retry + 1):
                    root.pop(key, None)
                    step: MeasureStep[T_Cfg, Hashable, Hashable] = MeasureStep(
                        session=self,
                        name=str(key),
                        index=key,
                        value=key,
                        cfg=deepcopy(self.cfg),
                        path=(key,),
                        dynamic_pbar=True,
                    )

                    try:
                        result = child_fn(step)
                    except KeyboardInterrupt:
                        self.set_stop()
                        break
                    except Exception:
                        if attempt == retry:
                            raise
                        continue

                    results[key] = result
                    if key not in root:
                        root[key] = result
                    self._trigger_update(step.cfg)
                    completed = True
                    break

                if completed:
                    pbar.update()

                if self.is_stop():
                    break
        finally:
            pbar.close()

        return results

    def _scan(
        self,
        *,
        parent: MeasureSession[T_Cfg] | MeasureStep[T_Cfg, Any, Any],
        name: str,
        values: Iterable[T_Value],
    ) -> Iterator[MeasureStep[T_Cfg, int, T_Value]]:
        if isinstance(values, Sized):
            total = len(values)
            sweep_values = values
        else:
            materialized_values = list(values)
            total = len(materialized_values)
            sweep_values = materialized_values

        pbar = make_pbar(
            total=total,
            smoothing=0,
            desc=name,
            leave=_leave_pbar(parent),
        )

        try:
            for index, value in enumerate(sweep_values):
                if self.is_stop():
                    break

                yield MeasureStep(
                    session=self,
                    name=name,
                    index=index,
                    value=value,
                    cfg=deepcopy(parent.cfg),
                    path=_extend_path(parent, name, index),
                    dynamic_pbar=True,
                )

                pbar.update()
        finally:
            pbar.close()

    def _repeat(
        self,
        *,
        parent: MeasureSession[T_Cfg] | MeasureStep[T_Cfg, Any, Any],
        name: str,
        times: int,
        interval: float,
    ) -> Iterator[MeasureStep[T_Cfg, int, int]]:
        if times < 0:
            raise ValueError("times must be non-negative")
        if interval < 0.0:
            raise ValueError("interval must be non-negative")

        leave = _leave_pbar(parent)
        iter_pbar = make_pbar(total=times, smoothing=0, desc=name, leave=leave)
        time_pbar = make_pbar(
            total=interval,
            smoothing=0,
            desc="Passing Time",
            leave=leave,
            miniters=0.2,
            bar_format="{desc}: {bar} {n:.1f}/{total:.1f} s",
            disable=interval == 0.0,
        )
        start_t = time.time() - 2 * interval

        try:
            for index in range(times):
                if self.is_stop():
                    break

                while time.time() - start_t < interval:
                    if self.is_stop():
                        break
                    pass_time = round(time.time() - start_t, 1)
                    time_pbar.update(pass_time - time_pbar.n)
                    time.sleep(0.1)
                time_pbar.reset()

                if self.is_stop():
                    break

                start_t = time.time()
                self.env["repeat_idx"] = index

                yield MeasureStep(
                    session=self,
                    name=name,
                    index=index,
                    value=index,
                    cfg=deepcopy(parent.cfg),
                    path=_extend_path(parent, name, index),
                    dynamic_pbar=True,
                )

                iter_pbar.update()

                if time.time() - start_t < interval:
                    with MinIntervalFunc.force_execute():
                        self._trigger_update(parent.cfg)
        finally:
            iter_pbar.close()
            time_pbar.close()

    def _register_root_buffer(
        self, array: NDArray[Any], *, name: Hashable | None
    ) -> None:
        if name is None:
            if self._root_data is not None or self._has_unnamed_root_buffer:
                raise ValueError(
                    "MeasureSession supports only one unnamed root buffer; "
                    "use named buffers for multiple root results"
                )
            self._root_data = array
            self._has_unnamed_root_buffer = True
            return

        root = self._ensure_mapping_root()
        if name in root:
            raise ValueError(f"Duplicate buffer name: {name!r}")
        root[name] = array

    def _register_child_buffer(
        self,
        path: tuple[Hashable, ...],
        array: NDArray[Any],
        *,
        name: Hashable | None,
    ) -> None:
        if not path:
            self._register_root_buffer(array, name=name)
            return

        root = self._ensure_mapping_root()
        if len(path) != 1:
            raise ValueError(
                "Step-local buffers are only supported for batch child steps"
            )

        child_key = path[0]
        if name is None:
            if child_key in root:
                raise ValueError(
                    "Cannot add an unnamed child buffer after named child buffers exist"
                )
            root[child_key] = array
            return

        child_data = root.setdefault(child_key, {})
        if not isinstance(child_data, MutableMapping):
            raise ValueError(
                "Cannot add a named buffer after an unnamed child buffer exists"
            )
        if name in child_data:
            raise ValueError(f"Duplicate child buffer name: {name!r}")
        child_data[name] = array

    def _ensure_mapping_root(self) -> MutableMapping[Hashable, Any]:
        if self._root_data is None:
            self._root_data = {}
            return cast(MutableMapping[Hashable, Any], self._root_data)
        if not isinstance(self._root_data, MutableMapping):
            raise ValueError(
                "MeasureSession root data is an unnamed buffer; "
                "cannot add named buffers or batch results"
            )
        return cast(MutableMapping[Hashable, Any], self._root_data)

    def _trigger_update(self, cfg: T_Cfg) -> None:
        if self._on_update is None or self._root_data is None:
            return
        self._on_update(
            MeasureSnapshot(
                root_data=self._root_data,
                cfg=cfg,
                env=MappingProxyType(self.env),
            )
        )


@dataclass
class MeasureStep(Generic[T_Cfg, T_Index, T_Value]):
    session: MeasureSession[T_Cfg]
    name: str
    index: T_Index
    value: T_Value
    cfg: T_Cfg
    path: tuple[Hashable, ...]
    dynamic_pbar: bool = True
    _has_unnamed_buffer: bool = False

    @property
    def env(self) -> dict[str, Any]:
        return self.session.env

    def is_stop(self) -> bool:
        return self.session.is_stop()

    def set_stop(self) -> None:
        self.session.set_stop()

    def buffer(
        self,
        shape: int | Sequence[int],
        *,
        dtype: DTypeLike = np.complex128,
        name: Hashable | None = None,
        on_update: Callable[[NDArray[Any]], None] | None = None,
    ) -> MeasureBuffer[T_Cfg]:
        if name is None:
            if self._has_unnamed_buffer:
                raise ValueError(
                    "MeasureStep supports only one unnamed buffer; "
                    "use named buffers for multiple child results"
                )
            self._has_unnamed_buffer = True

        array = _make_nan_array(shape, dtype)
        self.session._register_child_buffer(self.path, array, name=name)
        return MeasureBuffer(
            session=self.session,
            owner=self,
            array=array,
            on_update=min_interval(on_update, self.session._update_interval),
        )

    def scan(
        self, name: str, values: Iterable[T_NestedValue]
    ) -> Iterator[MeasureStep[T_Cfg, int, T_NestedValue]]:
        yield from self.session._scan(parent=self, name=name, values=values)

    def repeat(
        self, name: str, times: int, interval: float = 0.0
    ) -> Iterator[MeasureStep[T_Cfg, int, int]]:
        yield from self.session._repeat(
            parent=self, name=name, times=times, interval=interval
        )


@dataclass
class MeasureBuffer(Generic[T_Cfg]):
    session: MeasureSession[T_Cfg]
    owner: MeasureSession[T_Cfg] | MeasureStep[T_Cfg, Any, Any]
    array: NDArray[Any]
    on_update: Callable[[NDArray[Any]], None] | None = None

    def measure(
        self,
        measure_fn: Callable[
            [TaskState[NDArray[Any], NDArray[Any], T_Cfg], Callable[[int, T_Raw], Any]],
            T_Raw,
        ],
        *,
        raw2signal_fn: Callable[[T_Raw], NDArray[Any]] | None = None,
        pbar_n: int | None = None,
        retry: int = 0,
    ) -> NDArray[Any]:
        return self.at().measure(
            measure_fn,
            raw2signal_fn=raw2signal_fn,
            pbar_n=pbar_n,
            retry=retry,
        )

    def at(self, *index_or_step: Any) -> MeasureSlot[T_Cfg]:
        index_parts: tuple[Any, ...]
        if len(index_or_step) == 1 and isinstance(index_or_step[0], tuple):
            index_parts = index_or_step[0]
        else:
            index_parts = index_or_step

        owner = self.owner
        resolved_index: list[Any] = []
        for part in index_parts:
            if isinstance(part, MeasureStep):
                if part.session is not self.session:
                    raise ValueError("MeasureStep belongs to a different session")
                owner = part
                resolved_index.append(part.index)
            else:
                resolved_index.append(part)

        view = _writable_view(self.array, tuple(resolved_index))
        return MeasureSlot(session=self.session, job=owner, buffer=self, view=view)

    def __getitem__(self, index_or_step: Any) -> MeasureSlot[T_Cfg]:
        return self.at(index_or_step)

    def _trigger_update(self) -> None:
        if self.on_update is not None:
            self.on_update(self.array)


@dataclass
class MeasureSlot(Generic[T_Cfg]):
    session: MeasureSession[T_Cfg]
    job: MeasureSession[T_Cfg] | MeasureStep[T_Cfg, Any, Any]
    buffer: MeasureBuffer[T_Cfg]
    view: NDArray[Any]

    def measure(
        self,
        measure_fn: Callable[
            [TaskState[NDArray[Any], NDArray[Any], T_Cfg], Callable[[int, T_Raw], Any]],
            T_Raw,
        ],
        *,
        raw2signal_fn: Callable[[T_Raw], NDArray[Any]] | None = None,
        pbar_n: int | None = None,
        retry: int = 0,
    ) -> NDArray[Any]:
        if retry < 0:
            raise ValueError("retry must be non-negative")

        cfg = self.job.cfg
        signal_fn = (
            raw2signal_fn
            if raw2signal_fn is not None
            else cast(Callable[[T_Raw], NDArray[Any]], default_raw2signal_fn)
        )
        task = Task(
            measure_fn=measure_fn,
            raw2signal_fn=signal_fn,
            result_shape=self.view.shape,
            dtype=self.view.dtype.type,
            pbar_n=pbar_n,
        )
        state: TaskState[NDArray[Any], NDArray[Any], T_Cfg] = TaskState(
            root_data=self.view,
            cfg=cfg,
            env=self.session.env,
            on_update=lambda _snap: self._trigger_update(cfg),
            _stop_flag=self.session._stop_flag,
        )

        task.init(dynamic_pbar=_dynamic_pbar(self.job))
        try:
            if retry == 0:
                task.run(state)
            else:
                run_with_retries(
                    task,
                    state,
                    retry_time=retry,
                    dynamic_pbar=_dynamic_pbar(self.job),
                )
        except KeyboardInterrupt:
            self.session.set_stop()
        finally:
            task.cleanup()

        return self.view

    def _trigger_update(self, cfg: T_Cfg) -> None:
        self.buffer._trigger_update()
        self.session._trigger_update(cfg)


def _make_nan_array(shape: int | Sequence[int], dtype: DTypeLike) -> NDArray[Any]:
    normalized_shape = (shape,) if isinstance(shape, int) else tuple(shape)
    return np.full(normalized_shape, np.nan, dtype=dtype)


def _writable_view(array: NDArray[Any], index: tuple[Any, ...]) -> NDArray[Any]:
    if not index:
        return array

    direct = array[index]
    if isinstance(direct, np.ndarray):
        if not np.shares_memory(direct, array):
            raise ValueError("MeasureSlot indexing must select a writable view")
        return direct

    scalar_index: list[slice] = []
    for axis, part in enumerate(index):
        if not isinstance(part, int):
            raise ValueError("Scalar MeasureSlot indexing only supports integer axes")
        axis_size = array.shape[axis]
        normalized = part + axis_size if part < 0 else part
        if normalized < 0 or normalized >= axis_size:
            raise IndexError(f"index {part} is out of bounds for axis {axis}")
        scalar_index.append(slice(normalized, normalized + 1))

    return array[tuple(scalar_index)].reshape(())


def _dynamic_pbar(job: MeasureSession[T_Cfg] | MeasureStep[T_Cfg, Any, Any]) -> bool:
    return isinstance(job, MeasureStep) and job.dynamic_pbar


def _leave_pbar(parent: MeasureSession[T_Cfg] | MeasureStep[T_Cfg, Any, Any]) -> bool:
    return isinstance(parent, MeasureSession)


def _extend_path(
    parent: MeasureSession[T_Cfg] | MeasureStep[T_Cfg, Any, Any],
    name: str,
    index: int,
) -> tuple[Hashable, ...]:
    if isinstance(parent, MeasureSession):
        return (name, index)
    return parent.path + (name, index)
