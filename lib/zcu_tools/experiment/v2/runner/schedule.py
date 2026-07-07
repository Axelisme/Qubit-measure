from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    Sized,
)
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, Self, TypeAlias, cast, overload

import numpy as np
from numpy.typing import DTypeLike, NDArray
from typing_extensions import TypeVar

from zcu_tools.program.base import CancelFlagProtocol
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Module,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
)
from zcu_tools.progress_bar import BaseProgressBar, make_pbar
from zcu_tools.utils.func_tools import min_interval

from ._path import get_path, set_target, writable_view

T_Cfg = TypeVar("T_Cfg")
T_ChildCfg = TypeVar("T_ChildCfg")
T_Env = TypeVar("T_Env", default=dict[str, Any])

T_Raw = TypeVar("T_Raw")
T_Value = TypeVar("T_Value")
T_NestedValue = TypeVar("T_NestedValue")
T_BatchResult = TypeVar("T_BatchResult")
T_AcquireRaw_co = TypeVar("T_AcquireRaw_co", covariant=True)
T_DecimatedRaw_co = TypeVar("T_DecimatedRaw_co", covariant=True)
SignalArray: TypeAlias = NDArray[Any]
RunStatus: TypeAlias = Literal["completed", "stopped", "interrupted", "failed"]
StopCondition: TypeAlias = Callable[[], bool]
ErrorStatus: TypeAlias = Literal["interrupted", "failed"]


class ScheduleOutcomeError(RuntimeError):
    """Exception wrapper for a Schedule failure that returned partial data."""

    def __init__(
        self,
        status: ErrorStatus,
        reason: str,
        exception: BaseException | None,
    ) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.exception = exception


class ProgramProtocol(Protocol[T_AcquireRaw_co, T_DecimatedRaw_co]):
    @property
    def cfg_model(self) -> ProgramV2Cfg: ...

    def acquire(
        self,
        soc: object,
        *,
        progress: bool,
        round_hook: Callable[[int, T_AcquireRaw_co, CancelFlagProtocol], object],
        cancel_flag: CancelFlagProtocol,
        **kwargs: object,
    ) -> T_AcquireRaw_co: ...

    def acquire_decimated(
        self,
        soc: object,
        *,
        progress: bool,
        round_hook: Callable[[int, T_DecimatedRaw_co, CancelFlagProtocol], object],
        cancel_flag: CancelFlagProtocol,
        **kwargs: object,
    ) -> T_DecimatedRaw_co: ...


T_Program = TypeVar("T_Program", bound=ProgramProtocol[Any, Any])
ProgramFactory: TypeAlias = Callable[..., T_Program]


class BufferProtocol(Protocol):
    """Minimal data/update surface consumed by Schedule."""

    @property
    def data(self) -> Any: ...

    def trigger_update(
        self, step: ScheduleStep[Any, Any, Any] | None = None, *, flush: bool = False
    ) -> None: ...


def default_raw2signal_fn(raw: Sequence[NDArray[np.float64]]) -> NDArray[np.complex128]:
    return raw[0][0].dot([1, 1j])


def default_decimated_raw2signal_fn(
    raw: Sequence[NDArray[np.float64]],
) -> NDArray[np.complex128]:
    return raw[0].dot([1, 1j])


class StopSignal:
    """Schedule-owned stop signal shared by host loops and program acquire."""

    def __init__(self, event: threading.Event | None = None) -> None:
        self._event = event if event is not None else threading.Event()
        self._error: ScheduleOutcomeError | None = None
        self._lock = threading.Lock()

    def is_stop(self) -> bool:
        return self._event.is_set()

    def set_stop(self) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def set(self) -> None:
        self._event.set()

    def clear_stop(self) -> None:
        with self._lock:
            self._error = None
        self._event.clear()

    @property
    def event(self) -> threading.Event:
        return self._event

    @property
    def error(self) -> ScheduleOutcomeError | None:
        with self._lock:
            return self._error

    def set_error(
        self,
        status: ErrorStatus,
        reason: str,
        exception: BaseException | None,
    ) -> None:
        error = ScheduleOutcomeError(status, reason, exception)
        with self._lock:
            if self._error is None:
                self._error = error
        self._event.set()

    def raise_if_error(self) -> None:
        error = self.error
        if error is None:
            return
        if error.exception is not None:
            raise error from error.exception
        raise error


class _AcquireCancelFlag:
    """Program-local stop flag that observes Schedule stop without mutating it."""

    def __init__(self, external: StopSignal) -> None:
        self._external = external
        self._local = threading.Event()

    def is_set(self) -> bool:
        return self._external.is_set() or self._local.is_set()

    def set(self) -> None:
        self._local.set()


@dataclass(frozen=True)
class ScheduleOutcome:
    """Completion state for a Schedule run."""

    status: RunStatus = "completed"
    reason: str | None = None
    exception: BaseException | None = None

    @property
    def is_partial(self) -> bool:
        return self.status != "completed"


_current_stop_signal: ContextVar[StopSignal | None] = ContextVar(
    "zcu_tools_schedule_stop_signal", default=None
)


@contextmanager
def schedule_stop_scope(stop: StopSignal) -> Iterator[StopSignal]:
    token = _current_stop_signal.set(stop)
    try:
        yield stop
    finally:
        _current_stop_signal.reset(token)


def current_stop_signal() -> StopSignal | None:
    return _current_stop_signal.get()


class SignalBuffer:
    """Array-backed result buffer with runner-aware update hooks."""

    def __init__(
        self,
        shape: int | Sequence[int],
        *,
        dtype: DTypeLike = np.complex128,
        on_update: Callable[[NDArray[Any]], None] | None = None,
        update_interval: float | None = 0.1,
    ) -> None:
        self.array = _make_nan_array(shape, dtype)
        self._throttled_update = min_interval(on_update, update_interval)

    @property
    def data(self) -> NDArray[Any]:
        return self.array

    def at(self, *index_or_step: Any) -> SignalSlot:
        return SignalSlot(
            buffer=self, view=writable_view(self.array, _resolve_index(index_or_step))
        )

    def __getitem__(self, index_or_step: Any) -> SignalSlot:
        if isinstance(index_or_step, tuple):
            return self.at(*index_or_step)
        return self.at(index_or_step)

    def set(self, value: NDArray[Any]) -> None:
        np.copyto(dst=self.array, src=value)
        self.trigger_update()

    def trigger_update(
        self, step: ScheduleStep[Any, Any, Any] | None = None, *, flush: bool = False
    ) -> None:
        if self._throttled_update is not None:
            self._throttled_update(self.array)


@dataclass(frozen=True)
class SignalSlot:
    buffer: SignalBuffer
    view: NDArray[Any]

    @property
    def value(self) -> NDArray[Any]:
        return self.view

    def set(self, value: NDArray[Any]) -> None:
        np.copyto(dst=self.view, src=value)
        self.buffer.trigger_update()


class Schedule(Generic[T_Cfg, T_Env]):
    """Runtime scope for Python-like measurement orchestration."""

    def __init__(
        self,
        init_cfg: T_Cfg,
        *buffers: BufferProtocol,
        env: T_Env | None = None,
        stop: StopSignal | None = None,
    ) -> None:
        self._ensure_single_root_buffer_count(len(buffers))
        self.cfg = deepcopy(init_cfg)
        self._env = cast(T_Env, {} if env is None else env)
        self._buffers = list(buffers)
        self._local_buffers: dict[tuple[Hashable, ...], SignalBuffer] = {}
        self._outcome = ScheduleOutcome()
        self._is_active = False
        self._is_closed = False
        resolved_stop = stop if stop is not None else _current_stop_signal.get()
        self._stop = resolved_stop if resolved_stop is not None else StopSignal()

    def __enter__(self) -> Schedule[T_Cfg, T_Env]:
        if self._is_closed:
            raise RuntimeError("Schedule cannot be reused after exiting its context")
        if self._is_active:
            raise RuntimeError("Schedule context is already active")
        self._is_active = True
        return self

    def __exit__(self, *_) -> Literal[False]:
        self._is_active = False
        self._is_closed = True
        return False

    @property
    def data(self) -> Any:
        return self._get_data(self)

    @property
    def env(self) -> T_Env:
        return self._env

    def register_buffer(self, *buffers: BufferProtocol) -> None:
        self._ensure_active()
        if not buffers:
            raise ValueError("register_buffer requires at least one SignalBuffer")
        self._ensure_single_root_buffer_count(len(self._buffers) + len(buffers))
        self._buffers.extend(buffers)

    def trigger_update(self, *, flush: bool = False) -> None:
        self._trigger_update(self, flush=flush)

    def set_data(self, value: Any, *, flush: bool = False) -> None:
        self._set_data(self, value, flush=flush)

    @property
    def path(self) -> tuple[Hashable, ...]:
        return ()

    @property
    def stop(self) -> StopSignal:
        return self._stop

    @property
    def outcome(self) -> ScheduleOutcome:
        return self._outcome

    def is_stop(self) -> bool:
        return self._stop.is_stop()

    def set_stop(self) -> None:
        self._mark_stopped("stop requested")

    def clear_stop(self) -> None:
        self._stop.clear_stop()

    def scan(
        self, name: str, values: Iterable[T_Value]
    ) -> Iterator[tuple[T_Value, ScheduleStep[T_Cfg, T_Value, T_Env]]]:
        self._ensure_active()
        yield from self._scan(parent=self, name=name, values=values)

    def repeat(
        self, name: str, times: int, interval: float = 0.0
    ) -> Iterator[tuple[int, ScheduleStep[T_Cfg, int, T_Env]]]:
        self._ensure_active()
        yield from self._repeat(parent=self, name=name, times=times, interval=interval)

    def batch(
        self,
        tasks: Mapping[
            Hashable, Callable[[ScheduleStep[T_Cfg, Hashable, T_Env]], T_BatchResult]
        ],
    ) -> dict[Hashable, T_BatchResult]:
        self._ensure_active()
        return self._batch(parent=self, tasks=tasks)

    @overload
    def prog_builder(
        self,
        soc: object,
        soccfg: object,
        *,
        cfg: object | None = None,
        program_cls: None = None,
        **program_kwargs: object,
    ) -> ProgramBuilder[ModularProgramV2]: ...

    @overload
    def prog_builder(
        self,
        soc: object,
        soccfg: object,
        *,
        cfg: object | None = None,
        program_cls: ProgramFactory[T_Program],
        **program_kwargs: object,
    ) -> ProgramBuilder[T_Program]: ...

    def prog_builder(
        self,
        soc: object,
        soccfg: object,
        *,
        cfg: object | None = None,
        program_cls: ProgramFactory[T_Program] | None = None,
        **program_kwargs: object,
    ) -> ProgramBuilder[T_Program] | ProgramBuilder[ModularProgramV2]:
        self._ensure_active()
        if program_cls is None:
            return ProgramBuilder(
                owner=self,
                soc=soc,
                soccfg=soccfg,
                cfg=self.cfg if cfg is None else cfg,
                program_cls=ModularProgramV2,
                program_kwargs=program_kwargs,
            )
        return ProgramBuilder(
            owner=self,
            soc=soc,
            soccfg=soccfg,
            cfg=self.cfg if cfg is None else cfg,
            program_cls=program_cls,
            program_kwargs=program_kwargs,
        )

    def _ensure_active(self) -> None:
        if not self._is_active:
            raise RuntimeError(
                "Schedule operations must run inside 'with Schedule(...)'"
            )

    def _mark_stopped(self, reason: str) -> None:
        self._stop.set_stop()
        self._set_outcome("stopped", reason=reason)

    def _mark_interrupted(self, exc: BaseException) -> None:
        reason = _exception_reason(exc)
        if self._set_outcome("interrupted", reason=reason, exception=exc):
            self._stop.set_error("interrupted", reason, exc)
        else:
            self._stop.set_stop()

    def _mark_failed(self, exc: BaseException) -> None:
        reason = _exception_reason(exc)
        if self._set_outcome("failed", reason=reason, exception=exc):
            self._stop.set_error("failed", reason, exc)
        else:
            self._stop.set_stop()

    def _set_outcome(
        self,
        status: RunStatus,
        *,
        reason: str | None = None,
        exception: BaseException | None = None,
    ) -> bool:
        if self._outcome.status == "completed":
            self._outcome = ScheduleOutcome(
                status=status, reason=reason, exception=exception
            )
            return True
        return False

    def _reset_outcome_for_retry(self) -> None:
        self._outcome = ScheduleOutcome()
        self._stop.clear_stop()

    def _check_stop_requested(self) -> bool:
        if self.is_stop():
            self._mark_stopped("stop requested")
            return True
        return False

    def _should_retry_after_failed_attempt(self, attempt: int, retry: int) -> bool:
        if self._outcome.status != "failed" or attempt >= retry:
            return False
        self._reset_outcome_for_retry()
        return True

    @staticmethod
    def _ensure_single_root_buffer_count(buffer_count: int) -> None:
        if buffer_count > 1:
            raise ValueError("Schedule supports at most one root result buffer")

    def _scan(
        self,
        *,
        parent: Schedule[T_Cfg, T_Env] | ScheduleStep[T_Cfg, Any, T_Env],
        name: str,
        values: Iterable[T_Value],
    ) -> Iterator[tuple[T_Value, ScheduleStep[T_Cfg, T_Value, T_Env]]]:
        if isinstance(values, Sized):
            sweep_values = values
            total = len(values)
        else:
            sweep_values = list(values)
            total = len(sweep_values)
        pbar = make_pbar(
            total=total,
            smoothing=0,
            desc=name,
            leave=isinstance(parent, Schedule),
        )
        try:
            for index, value in enumerate(sweep_values):
                if self._check_stop_requested():
                    break
                step = ScheduleStep(
                    schedule=self,
                    name=name,
                    index=index,
                    value=value,
                    cfg=deepcopy(parent.cfg),
                    path=parent.path + (index,),
                )
                try:
                    yield value, step
                finally:
                    self._clear_local_buffers(step.path)
                pbar.update()
        finally:
            pbar.close()

    def _repeat(
        self,
        *,
        parent: Schedule[T_Cfg, T_Env] | ScheduleStep[T_Cfg, Any, T_Env],
        name: str,
        times: int,
        interval: float,
    ) -> Iterator[tuple[int, ScheduleStep[T_Cfg, int, T_Env]]]:
        if times < 0:
            raise ValueError("times must be non-negative")
        if interval < 0.0:
            raise ValueError("interval must be non-negative")

        leave = isinstance(parent, Schedule)
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
                if self._check_stop_requested():
                    break

                while time.time() - start_t < interval:
                    if self._check_stop_requested():
                        break
                    passed_time = round(time.time() - start_t, 1)
                    time_pbar.set_progress(passed_time)
                    time.sleep(0.1)
                time_pbar.reset()

                if self._check_stop_requested():
                    break

                step = ScheduleStep(
                    schedule=self,
                    name=name,
                    index=index,
                    value=index,
                    cfg=deepcopy(parent.cfg),
                    path=parent.path + (index,),
                )
                try:
                    yield index, step
                finally:
                    self._clear_local_buffers(step.path)
                iter_pbar.update()
                start_t = time.time()
        finally:
            iter_pbar.close()
            time_pbar.close()

    def _batch(
        self,
        *,
        parent: Schedule[T_Cfg, T_Env] | ScheduleStep[T_Cfg, Any, T_Env],
        tasks: Mapping[
            Hashable, Callable[[ScheduleStep[T_Cfg, Hashable, T_Env]], T_BatchResult]
        ],
    ) -> dict[Hashable, T_BatchResult]:
        results: dict[Hashable, T_BatchResult] = {}
        task_items = list(tasks.items())
        pbar = make_pbar(
            total=len(task_items),
            smoothing=0,
            leave=isinstance(parent, Schedule),
        )
        try:
            for key, child_fn in task_items:
                if self._check_stop_requested():
                    break

                pbar.set_description(f"Batch [{str(key)}]")
                step: ScheduleStep[T_Cfg, Hashable, T_Env] = ScheduleStep(
                    schedule=self,
                    name=str(key),
                    index=key,
                    value=key,
                    cfg=deepcopy(parent.cfg),
                    path=parent.path + (key,),
                )
                self._clear_local_buffers(step.path)
                try:
                    result = child_fn(step)
                except KeyboardInterrupt as exc:
                    self._clear_local_buffers(step.path)
                    self._mark_interrupted(exc)
                    break
                except Exception as exc:
                    self._clear_local_buffers(step.path)
                    self._mark_failed(exc)
                    break

                self._clear_local_buffers(step.path)
                results[key] = result
                pbar.update()
                if self._check_stop_requested():
                    break
        finally:
            pbar.close()

        return results

    def _get_data(self, owner: Schedule[Any, Any] | ScheduleStep[Any, Any, Any]) -> Any:
        self._ensure_active()
        return get_path(self._data_root(), owner.path)

    def _data_root(self) -> Any:
        if len(self._buffers) == 1:
            return self._buffers[0].data
        if not self._buffers:
            return ()
        raise ValueError("Schedule supports at most one root result buffer")

    def _set_data(
        self,
        owner: Schedule[Any, Any] | ScheduleStep[Any, Any, Any],
        value: Any,
        *,
        flush: bool = False,
    ) -> None:
        target = self._get_data(owner)
        set_target(target, value)
        self._trigger_update(owner, flush=flush)

    def _register_local_buffer(
        self,
        owner: Schedule[Any, Any] | ScheduleStep[Any, Any, Any],
        buffer: SignalBuffer,
    ) -> None:
        self._ensure_active()
        target = self._get_data(owner)
        if not isinstance(target, np.ndarray):
            raise ValueError(
                "ScheduleStep.buffer target must point to an NDArray in result data, "
                f"got {type(target)} at path {owner.path}"
            )
        if target.shape != buffer.array.shape:
            raise ValueError(
                "ScheduleStep.buffer shape must match result data target shape; "
                f"target shape={target.shape}, buffer shape={buffer.array.shape}"
            )
        self._local_buffers[owner.path] = buffer

    def _trigger_update(
        self,
        owner: Schedule[Any, Any] | ScheduleStep[Any, Any, Any],
        *,
        flush: bool = False,
    ) -> None:
        self._ensure_active()
        if not self._buffers:
            return
        if isinstance(owner, ScheduleStep):
            step = owner
        else:
            step = None
        for buffer in self._buffers:
            if flush:
                buffer.trigger_update(step, flush=True)
            else:
                buffer.trigger_update(step)

    def _clear_local_buffers(self, prefix: tuple[Hashable, ...]) -> None:
        for path in list(self._local_buffers):
            if path[: len(prefix)] == prefix:
                self._local_buffers.pop(path, None)

    def _default_slot(
        self, owner: Schedule[T_Cfg, T_Env] | ScheduleStep[T_Cfg, Any, T_Env]
    ) -> SignalSlot:
        self._ensure_active()
        if owner.path in self._local_buffers:
            return self._local_buffers[owner.path].at()
        if len(self._buffers) != 1 or not isinstance(self._buffers[0], SignalBuffer):
            raise ValueError(
                "Schedule default acquire target requires exactly one SignalBuffer"
            )
        if not all(isinstance(part, int) for part in owner.path):
            raise ValueError(
                "Schedule default acquire target requires an integer-indexed path; "
                "write to an explicit SignalBuffer slot for non-integer batch keys"
            )
        return self._buffers[0].at(*owner.path)


@dataclass(frozen=True)
class ScheduleStep(Generic[T_Cfg, T_Value, T_Env]):
    schedule: Schedule[T_Cfg, T_Env]
    name: str
    index: Hashable
    value: T_Value
    cfg: T_Cfg
    path: tuple[Hashable, ...]

    @property
    def env(self) -> T_Env:
        return self.schedule.env

    @property
    def data(self) -> Any:
        return self.schedule._get_data(self)

    @property
    def array_data(self) -> NDArray[Any]:
        data = self.data
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"ScheduleStep.array_data requires NDArray data, got {type(data)}"
            )
        return data

    @property
    def stop(self) -> StopSignal:
        return self.schedule.stop

    def is_stop(self) -> bool:
        return self.schedule.is_stop()

    def set_stop(self) -> None:
        self.schedule.set_stop()

    def set_data(self, value: Any, *, flush: bool = False) -> None:
        self.schedule._set_data(self, value, flush=flush)

    def trigger_update(self, *, flush: bool = False) -> None:
        self.schedule._trigger_update(self, flush=flush)

    @overload
    def child(
        self, addr: Hashable, *, cfg: None = None
    ) -> ScheduleStep[T_Cfg, Hashable, T_Env]: ...

    @overload
    def child(
        self, addr: Hashable, *, cfg: T_ChildCfg
    ) -> ScheduleStep[T_ChildCfg, Hashable, T_Env]: ...

    def child(
        self, addr: Hashable, *, cfg: object | None = None
    ) -> ScheduleStep[Any, Hashable, T_Env]:
        self.schedule._ensure_active()
        return ScheduleStep(
            schedule=self.schedule,
            name=str(addr),
            index=addr,
            value=addr,
            cfg=deepcopy(self.cfg if cfg is None else cfg),
            path=self.path + (addr,),
        )

    def buffer(
        self,
        shape: int | Sequence[int],
        *,
        dtype: DTypeLike = np.complex128,
    ) -> SignalBuffer:
        self.schedule._ensure_active()
        buffer = SignalBuffer(
            shape,
            dtype=dtype,
            on_update=lambda data: self.set_data(data),
            update_interval=None,
        )
        self.schedule._register_local_buffer(self, buffer)
        return buffer

    def scan(
        self, name: str, values: Iterable[T_NestedValue]
    ) -> Iterator[tuple[T_NestedValue, ScheduleStep[T_Cfg, T_NestedValue, T_Env]]]:
        self.schedule._ensure_active()
        yield from self.schedule._scan(parent=self, name=name, values=values)

    def repeat(
        self, name: str, times: int, interval: float = 0.0
    ) -> Iterator[tuple[int, ScheduleStep[T_Cfg, int, T_Env]]]:
        self.schedule._ensure_active()
        yield from self.schedule._repeat(
            parent=self, name=name, times=times, interval=interval
        )

    def batch(
        self,
        tasks: Mapping[
            Hashable, Callable[[ScheduleStep[T_Cfg, Hashable, T_Env]], T_BatchResult]
        ],
    ) -> dict[Hashable, T_BatchResult]:
        self.schedule._ensure_active()
        return self.schedule._batch(parent=self, tasks=tasks)

    @overload
    def prog_builder(
        self,
        soc: object,
        soccfg: object,
        *,
        cfg: object | None = None,
        program_cls: None = None,
        **program_kwargs: object,
    ) -> ProgramBuilder[ModularProgramV2]: ...

    @overload
    def prog_builder(
        self,
        soc: object,
        soccfg: object,
        *,
        cfg: object | None = None,
        program_cls: ProgramFactory[T_Program],
        **program_kwargs: object,
    ) -> ProgramBuilder[T_Program]: ...

    def prog_builder(
        self,
        soc: object,
        soccfg: object,
        *,
        cfg: object | None = None,
        program_cls: ProgramFactory[T_Program] | None = None,
        **program_kwargs: object,
    ) -> ProgramBuilder[T_Program] | ProgramBuilder[ModularProgramV2]:
        self.schedule._ensure_active()
        if program_cls is None:
            return ProgramBuilder(
                owner=self,
                soc=soc,
                soccfg=soccfg,
                cfg=self.cfg if cfg is None else cfg,
                program_cls=ModularProgramV2,
                program_kwargs=program_kwargs,
            )
        return ProgramBuilder(
            owner=self,
            soc=soc,
            soccfg=soccfg,
            cfg=self.cfg if cfg is None else cfg,
            program_cls=program_cls,
            program_kwargs=program_kwargs,
        )


class ModuleFacade(ABC):
    @abstractmethod
    def add(self, *modules: Module | Sequence[Module]) -> Self: ...

    def add_reset(self, name: str, cfg: ResetCfg | None) -> Self:
        return self.add(Reset(name, cfg))

    def add_pulse(self, name: str, cfg: PulseCfg | None, **kwargs: Any) -> Self:
        return self.add(Pulse(name, cfg, **kwargs))

    def add_readout(self, name: str, cfg: ReadoutCfg) -> Self:
        return self.add(Readout(name, cfg))


class ProgramBuilder(ModuleFacade, Generic[T_Program]):
    """Build and acquire a single QICK program within a Schedule scope."""

    def __init__(
        self,
        *,
        owner: Schedule[Any, Any] | ScheduleStep[Any, Any, Any],
        soc: object,
        soccfg: object,
        cfg: object,
        program_cls: ProgramFactory[T_Program],
        program_kwargs: dict[str, object],
    ) -> None:
        self._owner = owner
        self._schedule = owner if isinstance(owner, Schedule) else owner.schedule
        self._soc = soc
        self._soccfg = soccfg
        self._cfg = cfg
        self._program_cls = program_cls
        self._program_kwargs = dict(program_kwargs)
        self._modules: list[Module] = []
        self._sweeps: list[tuple[str, SweepCfg | int]] = []
        self._raw2signal_fn: Callable[[Any], SignalArray] | None = None

    def add(self, *modules: Module | Sequence[Module]) -> Self:
        for module_or_group in modules:
            if isinstance(module_or_group, Module):
                self._modules.append(module_or_group)
            else:
                self._modules.extend(module_or_group)
        return self

    def declare_sweep(self, name: str, spec: SweepCfg | int) -> Self:
        self._sweeps.append((name, spec))
        return self

    def set_raw2signal_fn(self, fn: Callable[[T_Raw], SignalArray]) -> Self:
        self._raw2signal_fn = fn
        return self

    def build(self) -> T_Program:
        self._schedule._ensure_active()
        self._ensure_modules()
        return self._build_program(self._isolated_cfg())

    def build_and_acquire(
        self,
        *,
        raw2signal_fn: Callable[[Any], SignalArray] | None = None,
        retry: int = 0,
        progress: bool = False,
        progress_label: str = "rounds",
        progress_leave: bool | None = None,
        stop_condition: StopCondition | None = None,
        **acquire_kwargs: object,
    ) -> SignalArray:
        self._schedule._ensure_active()
        self._ensure_modules()
        return self._build_and_run(
            runner=self.run_program,
            raw2signal_fn=raw2signal_fn,
            retry=retry,
            progress=progress,
            progress_label=progress_label,
            progress_leave=progress_leave,
            stop_condition=stop_condition,
            acquire_kwargs=acquire_kwargs,
        )

    def build_and_acquire_decimated(
        self,
        *,
        raw2signal_fn: Callable[[Any], SignalArray] | None = None,
        retry: int = 0,
        progress: bool = False,
        progress_label: str = "rounds",
        progress_leave: bool | None = None,
        stop_condition: StopCondition | None = None,
        **acquire_kwargs: object,
    ) -> SignalArray:
        self._schedule._ensure_active()
        self._ensure_modules()
        return self._build_and_run(
            runner=self.run_program_decimated,
            raw2signal_fn=raw2signal_fn,
            retry=retry,
            progress=progress,
            progress_label=progress_label,
            progress_leave=progress_leave,
            stop_condition=stop_condition,
            acquire_kwargs=acquire_kwargs,
        )

    def run_program(
        self,
        program: ProgramProtocol[T_Raw, object],
        *,
        raw2signal_fn: Callable[[T_Raw], SignalArray] | None = None,
        retry: int = 0,
        progress: bool = False,
        progress_label: str = "rounds",
        progress_leave: bool | None = None,
        stop_condition: StopCondition | None = None,
        **acquire_kwargs: object,
    ) -> SignalArray:
        return self._run_program(
            acquire=program.acquire,
            rounds=self._rounds_from(program),
            raw2signal_fn=raw2signal_fn,
            default_raw2signal_fn=default_raw2signal_fn,
            retry=retry,
            progress=progress,
            progress_label=progress_label,
            progress_leave=progress_leave,
            stop_condition=stop_condition,
            acquire_kwargs=acquire_kwargs,
        )

    def run_program_decimated(
        self,
        program: ProgramProtocol[object, T_Raw],
        *,
        raw2signal_fn: Callable[[T_Raw], SignalArray] | None = None,
        retry: int = 0,
        progress: bool = False,
        progress_label: str = "rounds",
        progress_leave: bool | None = None,
        stop_condition: StopCondition | None = None,
        **acquire_kwargs: object,
    ) -> SignalArray:
        return self._run_program(
            acquire=program.acquire_decimated,
            rounds=self._rounds_from(program),
            raw2signal_fn=raw2signal_fn,
            default_raw2signal_fn=default_decimated_raw2signal_fn,
            retry=retry,
            progress=progress,
            progress_label=progress_label,
            progress_leave=progress_leave,
            stop_condition=stop_condition,
            acquire_kwargs=acquire_kwargs,
        )

    def _build_and_run(
        self,
        *,
        runner: Callable[..., SignalArray],
        raw2signal_fn: Callable[[Any], SignalArray] | None,
        retry: int,
        progress: bool,
        progress_label: str,
        progress_leave: bool | None,
        stop_condition: StopCondition | None,
        acquire_kwargs: dict[str, object],
    ) -> SignalArray:
        if retry < 0:
            raise ValueError("retry must be non-negative")

        slot = self._default_slot()
        for attempt in range(retry + 1):
            if self._schedule._check_stop_requested():
                return slot.view
            try:
                program = self._build_program(self._isolated_cfg())
            except KeyboardInterrupt as exc:
                self._schedule._mark_interrupted(exc)
                return slot.view
            except Exception as exc:
                if self._schedule._check_stop_requested():
                    return slot.view
                if attempt == retry:
                    self._schedule._mark_failed(exc)
                    return slot.view
                continue

            result = runner(
                program,
                raw2signal_fn=raw2signal_fn,
                retry=0,
                progress=progress,
                progress_label=progress_label,
                progress_leave=progress_leave,
                stop_condition=stop_condition,
                **acquire_kwargs,
            )
            if self._schedule._should_retry_after_failed_attempt(attempt, retry):
                continue
            return result

        return self._default_slot().view

    def _run_program(
        self,
        *,
        acquire: Callable[..., T_Raw],
        rounds: int,
        raw2signal_fn: Callable[[T_Raw], SignalArray] | None,
        default_raw2signal_fn: Callable[[Any], SignalArray],
        retry: int,
        progress: bool,
        progress_label: str,
        progress_leave: bool | None,
        stop_condition: StopCondition | None,
        acquire_kwargs: dict[str, object],
    ) -> SignalArray:
        if retry < 0:
            raise ValueError("retry must be non-negative")
        self._schedule._ensure_active()

        slot = self._default_slot()
        signal_fn = self._resolve_raw2signal_fn(raw2signal_fn, default_raw2signal_fn)

        pbar = make_pbar(
            total=rounds,
            smoothing=0,
            desc=progress_label,
            leave=self._resolve_progress_leave(progress_leave),
            disable=rounds == 1,
        )
        try:
            for attempt in range(retry + 1):
                if self._schedule._check_stop_requested():
                    break
                pbar.reset()
                try:
                    raw = self._acquire_once(
                        acquire=acquire,
                        progress=progress,
                        acquire_kwargs=acquire_kwargs,
                        stop_condition=stop_condition,
                        signal_fn=signal_fn,
                        slot=slot,
                        pbar=pbar,
                    )
                except KeyboardInterrupt as exc:
                    self._schedule._mark_interrupted(exc)
                    break
                except Exception as exc:
                    if self._schedule._check_stop_requested():
                        break
                    if attempt == retry:
                        self._schedule._mark_failed(exc)
                        break
                    continue

                if self._schedule._check_stop_requested():
                    break

                try:
                    pbar.set_progress(rounds)
                    slot.set(signal_fn(raw))
                except KeyboardInterrupt as exc:
                    self._schedule._mark_interrupted(exc)
                except Exception as exc:
                    self._schedule._mark_failed(exc)
                break
        except KeyboardInterrupt as exc:
            self._schedule._mark_interrupted(exc)
        finally:
            pbar.close()

        return slot.view

    def _resolve_raw2signal_fn(
        self,
        raw2signal_fn: Callable[[T_Raw], SignalArray] | None,
        default_fn: Callable[[Any], SignalArray],
    ) -> Callable[[Any], SignalArray]:
        signal_fn = raw2signal_fn if raw2signal_fn is not None else self._raw2signal_fn
        if signal_fn is None:
            return default_fn
        return signal_fn

    def _ensure_modules(self) -> None:
        if not self._modules:
            raise ValueError("ProgramBuilder requires explicit modules")

    def _isolated_cfg(self) -> ProgramV2Cfg:
        return deepcopy(_program_cfg_from(self._cfg))

    def _build_program(self, cfg: ProgramV2Cfg) -> T_Program:
        program_kwargs = dict(self._program_kwargs)
        program_kwargs["modules"] = deepcopy(self._modules)
        return self._program_cls(
            self._soccfg,
            cfg,
            sweep=deepcopy(self._sweeps) or None,
            **program_kwargs,
        )

    def _acquire_once(
        self,
        *,
        acquire: Callable[..., T_Raw],
        progress: bool,
        acquire_kwargs: dict[str, object],
        stop_condition: StopCondition | None,
        signal_fn: Callable[[T_Raw], SignalArray],
        slot: SignalSlot,
        pbar: BaseProgressBar,
    ) -> T_Raw:
        acquire_cancel_flag = _AcquireCancelFlag(self._schedule.stop)

        def update_hook(ir: int, raw: T_Raw, cancel_flag: CancelFlagProtocol) -> None:
            pbar.set_progress(ir)
            slot.set(signal_fn(raw))
            if stop_condition is not None and stop_condition():
                cancel_flag.set()

        return acquire(
            self._soc,
            progress=progress,
            round_hook=update_hook,
            cancel_flag=acquire_cancel_flag,
            **dict(acquire_kwargs),
        )

    def _rounds_from(self, program: ProgramProtocol[Any, Any]) -> int:
        rounds = program.cfg_model.rounds
        if rounds < 1:
            raise ValueError("Program cfg.rounds must be positive")
        return rounds

    def _resolve_progress_leave(self, progress_leave: bool | None) -> bool:
        if progress_leave is not None:
            return progress_leave
        return not isinstance(self._owner, ScheduleStep)

    def _default_slot(self) -> SignalSlot:
        return self._schedule._default_slot(self._owner)


def _program_cfg_from(cfg: object) -> ProgramV2Cfg:
    if isinstance(cfg, ProgramV2Cfg):
        return cfg
    program_fields = ProgramV2Cfg.model_fields
    if isinstance(cfg, Mapping):
        data = {key: cfg[key] for key in program_fields if key in cfg}
    else:
        data = {key: getattr(cfg, key) for key in program_fields if hasattr(cfg, key)}
    if not data:
        raise TypeError(
            "ProgramBuilder cfg must be ProgramV2Cfg or provide at least one "
            "ProgramV2Cfg field; pass ProgramV2Cfg() for default program runtime "
            "settings"
        )
    return ProgramV2Cfg.model_validate(data)


def _exception_reason(exc: BaseException) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def _make_nan_array(shape: int | Sequence[int], dtype: DTypeLike) -> NDArray[Any]:
    normalized_shape = (shape,) if isinstance(shape, int) else tuple(shape)
    return np.full(normalized_shape, np.nan, dtype=dtype)


def _resolve_index(index_or_step: tuple[Any, ...]) -> tuple[Any, ...]:
    resolved: list[Any] = []
    for part in index_or_step:
        if isinstance(part, ScheduleStep):
            resolved.extend(part.path)
        else:
            resolved.append(part)
    return tuple(resolved)
