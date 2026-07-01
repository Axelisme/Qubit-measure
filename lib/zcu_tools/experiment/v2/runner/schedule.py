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
from typing import Any, Generic, Literal, Protocol, Self, TypeAlias, TypeVar, overload

import numpy as np
from numpy.typing import DTypeLike, NDArray

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

T_Cfg = TypeVar("T_Cfg", bound=ProgramV2Cfg)

T_Raw = TypeVar("T_Raw")
T_Value = TypeVar("T_Value")
T_NestedValue = TypeVar("T_NestedValue")
T_BatchResult = TypeVar("T_BatchResult")
T_AcquireRaw_co = TypeVar("T_AcquireRaw_co", covariant=True)
T_DecimatedRaw_co = TypeVar("T_DecimatedRaw_co", covariant=True)
SignalArray: TypeAlias = NDArray[Any]


class ProgramProtocol(Protocol[T_AcquireRaw_co, T_DecimatedRaw_co]):
    @property
    def cfg_model(self) -> ProgramV2Cfg: ...

    def acquire(
        self,
        soc: object,
        *,
        progress: bool,
        round_hook: Callable[[int, T_AcquireRaw_co], object],
        stop_checkers: Sequence[Callable[[], bool]],
        **kwargs: object,
    ) -> T_AcquireRaw_co: ...

    def acquire_decimated(
        self,
        soc: object,
        *,
        progress: bool,
        round_hook: Callable[[int, T_DecimatedRaw_co], object],
        stop_checkers: Sequence[Callable[[], bool]],
        **kwargs: object,
    ) -> T_DecimatedRaw_co: ...


T_Program = TypeVar("T_Program", bound=ProgramProtocol[Any, Any])
ProgramFactory: TypeAlias = Callable[..., T_Program]


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

    def is_stop(self) -> bool:
        return self._event.is_set()

    def set_stop(self) -> None:
        self._event.set()

    def clear_stop(self) -> None:
        self._event.clear()

    @property
    def event(self) -> threading.Event:
        return self._event


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

    def at(self, *index_or_step: Any) -> SignalSlot:
        return SignalSlot(
            buffer=self, view=_writable_view(self.array, _resolve_index(index_or_step))
        )

    def __getitem__(self, index_or_step: Any) -> SignalSlot:
        if isinstance(index_or_step, tuple):
            return self.at(*index_or_step)
        return self.at(index_or_step)

    def set(self, value: NDArray[Any]) -> None:
        np.copyto(dst=self.array, src=value)
        self.trigger_update()

    def trigger_update(self) -> None:
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


class Schedule(Generic[T_Cfg]):
    """Runtime scope for Python-like measurement orchestration."""

    def __init__(
        self,
        init_cfg: T_Cfg,
        *buffers: SignalBuffer,
        env_dict: dict[str, Any] | None = None,
        stop: StopSignal | None = None,
    ) -> None:
        self.cfg = deepcopy(init_cfg)
        self.env = env_dict if env_dict is not None else {}
        self._buffers = list(buffers)
        self._is_active = False
        self._is_closed = False
        resolved_stop = stop if stop is not None else _current_stop_signal.get()
        self._stop = resolved_stop if resolved_stop is not None else StopSignal()

    def __enter__(self) -> Schedule[T_Cfg]:
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
    def root_data(self) -> Any:
        if len(self._buffers) == 1:
            return self._buffers[0].array
        return tuple(buffer.array for buffer in self._buffers)

    def register_buffer(self, *buffers: SignalBuffer) -> None:
        self._ensure_active()
        if not buffers:
            raise ValueError("register_buffer requires at least one SignalBuffer")
        self._buffers.extend(buffers)

    @property
    def path(self) -> tuple[Hashable, ...]:
        return ()

    @property
    def stop(self) -> StopSignal:
        return self._stop

    def is_stop(self) -> bool:
        return self._stop.is_stop()

    def set_stop(self) -> None:
        self._stop.set_stop()

    def clear_stop(self) -> None:
        self._stop.clear_stop()

    def scan(
        self, name: str, values: Iterable[T_Value]
    ) -> Iterator[tuple[T_Value, ScheduleStep[T_Cfg, T_Value]]]:
        self._ensure_active()
        yield from self._scan(parent=self, name=name, values=values)

    def repeat(
        self, name: str, times: int, interval: float = 0.0
    ) -> Iterator[tuple[int, ScheduleStep[T_Cfg, int]]]:
        self._ensure_active()
        yield from self._repeat(parent=self, name=name, times=times, interval=interval)

    def batch(
        self,
        tasks: Mapping[
            Hashable, Callable[[ScheduleStep[T_Cfg, Hashable]], T_BatchResult]
        ],
        *,
        retry: int = 0,
    ) -> dict[Hashable, T_BatchResult]:
        self._ensure_active()
        return self._batch(parent=self, tasks=tasks, retry=retry)

    @overload
    def prog_builder(
        self,
        soc: object,
        soccfg: object,
        *,
        program_cls: None = None,
        **program_kwargs: object,
    ) -> ProgramBuilder[T_Cfg, ModularProgramV2]: ...

    @overload
    def prog_builder(
        self,
        soc: object,
        soccfg: object,
        *,
        program_cls: ProgramFactory[T_Program],
        **program_kwargs: object,
    ) -> ProgramBuilder[T_Cfg, T_Program]: ...

    def prog_builder(
        self,
        soc: object,
        soccfg: object,
        *,
        program_cls: ProgramFactory[T_Program] | None = None,
        **program_kwargs: object,
    ) -> ProgramBuilder[T_Cfg, T_Program] | ProgramBuilder[T_Cfg, ModularProgramV2]:
        self._ensure_active()
        if program_cls is None:
            return ProgramBuilder(
                owner=self,
                soc=soc,
                soccfg=soccfg,
                program_cls=ModularProgramV2,
                program_kwargs=program_kwargs,
            )
        return ProgramBuilder(
            owner=self,
            soc=soc,
            soccfg=soccfg,
            program_cls=program_cls,
            program_kwargs=program_kwargs,
        )

    def _ensure_active(self) -> None:
        if not self._is_active:
            raise RuntimeError(
                "Schedule operations must run inside 'with Schedule(...)'"
            )

    def _scan(
        self,
        *,
        parent: Schedule[T_Cfg] | ScheduleStep[T_Cfg, Any],
        name: str,
        values: Iterable[T_Value],
    ) -> Iterator[tuple[T_Value, ScheduleStep[T_Cfg, T_Value]]]:
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
                if self.is_stop():
                    break
                self.env[name] = value
                self.env[f"{name}_idx"] = index
                step = ScheduleStep(
                    schedule=self,
                    name=name,
                    index=index,
                    value=value,
                    cfg=deepcopy(parent.cfg),
                    path=parent.path + (index,),
                )
                yield value, step
                pbar.update()
        finally:
            pbar.close()

    def _repeat(
        self,
        *,
        parent: Schedule[T_Cfg] | ScheduleStep[T_Cfg, Any],
        name: str,
        times: int,
        interval: float,
    ) -> Iterator[tuple[int, ScheduleStep[T_Cfg, int]]]:
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
                if self.is_stop():
                    break

                while time.time() - start_t < interval:
                    if self.is_stop():
                        break
                    passed_time = round(time.time() - start_t, 1)
                    time_pbar.update(passed_time - time_pbar.n)
                    time.sleep(0.1)
                time_pbar.reset()

                if self.is_stop():
                    break

                self.env["repeat_idx"] = index
                step = ScheduleStep(
                    schedule=self,
                    name=name,
                    index=index,
                    value=index,
                    cfg=deepcopy(parent.cfg),
                    path=parent.path + (index,),
                )
                yield index, step
                iter_pbar.update()
                start_t = time.time()
        finally:
            iter_pbar.close()
            time_pbar.close()

    def _batch(
        self,
        *,
        parent: Schedule[T_Cfg] | ScheduleStep[T_Cfg, Any],
        tasks: Mapping[
            Hashable, Callable[[ScheduleStep[T_Cfg, Hashable]], T_BatchResult]
        ],
        retry: int,
    ) -> dict[Hashable, T_BatchResult]:
        if retry < 0:
            raise ValueError("retry must be non-negative")

        results: dict[Hashable, T_BatchResult] = {}
        task_items = list(tasks.items())
        pbar = make_pbar(
            total=len(task_items),
            smoothing=0,
            leave=isinstance(parent, Schedule),
        )
        try:
            for key, child_fn in task_items:
                if self.is_stop():
                    break

                pbar.set_description(f"Task [{str(key)}]")
                completed = False
                for attempt in range(retry + 1):
                    step: ScheduleStep[T_Cfg, Hashable] = ScheduleStep(
                        schedule=self,
                        name=str(key),
                        index=key,
                        value=key,
                        cfg=deepcopy(parent.cfg),
                        path=parent.path + (key,),
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
                    completed = True
                    break

                if completed:
                    pbar.update()
                if self.is_stop():
                    break
        finally:
            pbar.close()

        return results

    def _default_slot(
        self, owner: Schedule[T_Cfg] | ScheduleStep[T_Cfg, Any]
    ) -> SignalSlot:
        if len(self._buffers) != 1:
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
class ScheduleStep(Generic[T_Cfg, T_Value]):
    schedule: Schedule[T_Cfg]
    name: str
    index: Hashable
    value: T_Value
    cfg: T_Cfg
    path: tuple[Hashable, ...]

    @property
    def env(self) -> dict[str, Any]:
        return self.schedule.env

    def is_stop(self) -> bool:
        return self.schedule.is_stop()

    def set_stop(self) -> None:
        self.schedule.set_stop()

    def scan(
        self, name: str, values: Iterable[T_NestedValue]
    ) -> Iterator[tuple[T_NestedValue, ScheduleStep[T_Cfg, T_NestedValue]]]:
        self.schedule._ensure_active()
        yield from self.schedule._scan(parent=self, name=name, values=values)

    def repeat(
        self, name: str, times: int, interval: float = 0.0
    ) -> Iterator[tuple[int, ScheduleStep[T_Cfg, int]]]:
        self.schedule._ensure_active()
        yield from self.schedule._repeat(
            parent=self, name=name, times=times, interval=interval
        )

    def batch(
        self,
        tasks: Mapping[
            Hashable, Callable[[ScheduleStep[T_Cfg, Hashable]], T_BatchResult]
        ],
        *,
        retry: int = 0,
    ) -> dict[Hashable, T_BatchResult]:
        self.schedule._ensure_active()
        return self.schedule._batch(parent=self, tasks=tasks, retry=retry)

    @overload
    def prog_builder(
        self,
        soc: object,
        soccfg: object,
        *,
        program_cls: None = None,
        **program_kwargs: object,
    ) -> ProgramBuilder[T_Cfg, ModularProgramV2]: ...

    @overload
    def prog_builder(
        self,
        soc: object,
        soccfg: object,
        *,
        program_cls: ProgramFactory[T_Program],
        **program_kwargs: object,
    ) -> ProgramBuilder[T_Cfg, T_Program]: ...

    def prog_builder(
        self,
        soc: object,
        soccfg: object,
        *,
        program_cls: ProgramFactory[T_Program] | None = None,
        **program_kwargs: object,
    ) -> ProgramBuilder[T_Cfg, T_Program] | ProgramBuilder[T_Cfg, ModularProgramV2]:
        self.schedule._ensure_active()
        if program_cls is None:
            return ProgramBuilder(
                owner=self,
                soc=soc,
                soccfg=soccfg,
                program_cls=ModularProgramV2,
                program_kwargs=program_kwargs,
            )
        return ProgramBuilder(
            owner=self,
            soc=soc,
            soccfg=soccfg,
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


class ProgramBuilder(ModuleFacade, Generic[T_Cfg, T_Program]):
    """Build and acquire a single QICK program within a Schedule scope."""

    def __init__(
        self,
        *,
        owner: Schedule[T_Cfg] | ScheduleStep[T_Cfg, Any],
        soc: object,
        soccfg: object,
        program_cls: ProgramFactory[T_Program],
        program_kwargs: dict[str, object],
    ) -> None:
        self._owner = owner
        self._schedule = owner if isinstance(owner, Schedule) else owner.schedule
        self._soc = soc
        self._soccfg = soccfg
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
        stop_checkers: Sequence[Callable[[], bool]] | None = None,
        **acquire_kwargs: object,
    ) -> SignalArray:
        self._schedule._ensure_active()
        self._ensure_modules()
        return self._build_and_run(
            runner=self.run_program,
            raw2signal_fn=raw2signal_fn,
            retry=retry,
            progress=progress,
            stop_checkers=stop_checkers,
            acquire_kwargs=acquire_kwargs,
        )

    def build_and_acquire_decimated(
        self,
        *,
        raw2signal_fn: Callable[[Any], SignalArray] | None = None,
        retry: int = 0,
        progress: bool = False,
        stop_checkers: Sequence[Callable[[], bool]] | None = None,
        **acquire_kwargs: object,
    ) -> SignalArray:
        self._schedule._ensure_active()
        self._ensure_modules()
        return self._build_and_run(
            runner=self.run_program_decimated,
            raw2signal_fn=raw2signal_fn,
            retry=retry,
            progress=progress,
            stop_checkers=stop_checkers,
            acquire_kwargs=acquire_kwargs,
        )

    def run_program(
        self,
        program: ProgramProtocol[T_Raw, object],
        *,
        raw2signal_fn: Callable[[T_Raw], SignalArray] | None = None,
        retry: int = 0,
        progress: bool = False,
        stop_checkers: Sequence[Callable[[], bool]] | None = None,
        **acquire_kwargs: object,
    ) -> SignalArray:
        return self._run_program(
            acquire=program.acquire,
            rounds=self._rounds_from(program),
            raw2signal_fn=raw2signal_fn,
            default_raw2signal_fn=default_raw2signal_fn,
            retry=retry,
            progress=progress,
            stop_checkers=stop_checkers,
            acquire_kwargs=acquire_kwargs,
        )

    def run_program_decimated(
        self,
        program: ProgramProtocol[object, T_Raw],
        *,
        raw2signal_fn: Callable[[T_Raw], SignalArray] | None = None,
        retry: int = 0,
        progress: bool = False,
        stop_checkers: Sequence[Callable[[], bool]] | None = None,
        **acquire_kwargs: object,
    ) -> SignalArray:
        return self._run_program(
            acquire=program.acquire_decimated,
            rounds=self._rounds_from(program),
            raw2signal_fn=raw2signal_fn,
            default_raw2signal_fn=default_decimated_raw2signal_fn,
            retry=retry,
            progress=progress,
            stop_checkers=stop_checkers,
            acquire_kwargs=acquire_kwargs,
        )

    def _build_and_run(
        self,
        *,
        runner: Callable[..., SignalArray],
        raw2signal_fn: Callable[[Any], SignalArray] | None,
        retry: int,
        progress: bool,
        stop_checkers: Sequence[Callable[[], bool]] | None,
        acquire_kwargs: dict[str, object],
    ) -> SignalArray:
        if retry < 0:
            raise ValueError("retry must be non-negative")

        for attempt in range(retry + 1):
            program = self._build_program(self._isolated_cfg())
            try:
                return runner(
                    program,
                    raw2signal_fn=raw2signal_fn,
                    retry=0,
                    progress=progress,
                    stop_checkers=stop_checkers,
                    **acquire_kwargs,
                )
            except Exception:
                if attempt == retry:
                    raise
                continue

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
        stop_checkers: Sequence[Callable[[], bool]] | None,
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
            desc="rounds",
            leave=not isinstance(self._owner, ScheduleStep),
            disable=rounds == 1,
        )
        try:
            for attempt in range(retry + 1):
                pbar.reset()
                try:
                    raw = self._acquire_once(
                        acquire=acquire,
                        progress=progress,
                        acquire_kwargs=acquire_kwargs,
                        stop_checkers=stop_checkers,
                        signal_fn=signal_fn,
                        slot=slot,
                        pbar=pbar,
                    )
                except KeyboardInterrupt:
                    self._schedule.set_stop()
                    break
                except Exception:
                    if attempt == retry:
                        raise
                    continue

                pbar.update(rounds - pbar.n)
                slot.set(signal_fn(raw))
                break
        except KeyboardInterrupt:
            self._schedule.set_stop()
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

    def _isolated_cfg(self) -> T_Cfg:
        return deepcopy(self._owner.cfg)

    def _build_program(self, cfg: T_Cfg) -> T_Program:
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
        stop_checkers: Sequence[Callable[[], bool]] | None,
        signal_fn: Callable[[T_Raw], SignalArray],
        slot: SignalSlot,
        pbar: BaseProgressBar,
    ) -> T_Raw:
        def update_hook(ir: int, raw: T_Raw) -> None:
            pbar.update(ir - pbar.n)
            slot.set(signal_fn(raw))

        return acquire(
            self._soc,
            progress=progress,
            round_hook=update_hook,
            stop_checkers=[self._schedule.is_stop, *(stop_checkers or [])],
            **dict(acquire_kwargs),
        )

    def _rounds_from(self, program: ProgramProtocol[Any, Any]) -> int:
        rounds = program.cfg_model.rounds
        if rounds < 1:
            raise ValueError("Program cfg.rounds must be positive")
        return rounds

    def _default_slot(self) -> SignalSlot:
        return self._schedule._default_slot(self._owner)


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


def _writable_view(array: NDArray[Any], index: tuple[Any, ...]) -> NDArray[Any]:
    if not index:
        return array

    direct = array[index]
    if isinstance(direct, np.ndarray):
        if not np.shares_memory(direct, array):
            raise ValueError("SignalBuffer indexing must select a writable view")
        return direct

    scalar_index: list[slice] = []
    for axis, part in enumerate(index):
        if not isinstance(part, int):
            raise ValueError("Scalar SignalBuffer indexing only supports integer axes")
        axis_size = array.shape[axis]
        normalized = part + axis_size if part < 0 else part
        if normalized < 0 or normalized >= axis_size:
            raise IndexError(f"index {part} is out of bounds for axis {axis}")
        scalar_index.append(slice(normalized, normalized + 1))

    return array[tuple(scalar_index)].reshape(())
