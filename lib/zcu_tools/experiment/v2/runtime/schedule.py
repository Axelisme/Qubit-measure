from __future__ import annotations

import time
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    Sized,
)
from copy import deepcopy
from dataclasses import dataclass, replace
from types import TracebackType
from typing import Any, Generic, Protocol, cast, overload

import numpy as np
from numpy.typing import DTypeLike, NDArray
from typing_extensions import TypeVar

from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.v2.runtime._path import get_path, set_target, writable_view
from zcu_tools.experiment.v2.runtime.outcome import (
    RunStatus as RunStatus,
)
from zcu_tools.experiment.v2.runtime.outcome import (
    ScheduleOutcome as ScheduleOutcome,
)
from zcu_tools.experiment.v2.runtime.outcome import (
    exception_reason,
)
from zcu_tools.experiment.v2.runtime.program_builder import (
    ProgramBuilder,
    ProgramFactory,
    ProgramT,
)
from zcu_tools.program.v2 import ModularProgramV2
from zcu_tools.progress_bar import make_pbar
from zcu_tools.utils.func_tools import MinIntervalFunc, min_interval

T_Cfg = TypeVar("T_Cfg")
T_ChildCfg = TypeVar("T_ChildCfg")
T_Env = TypeVar("T_Env", default=dict[str, Any])

T_Value = TypeVar("T_Value")
T_NestedValue = TypeVar("T_NestedValue")
T_BatchResult = TypeVar("T_BatchResult")


class BufferProtocol(Protocol):
    """Minimal data/update surface consumed by Schedule."""

    @property
    def data(self) -> Any: ...

    def trigger_update(
        self, step: ScheduleStep[Any, Any, Any] | None = None, *, flush: bool = False
    ) -> None: ...


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
        self,
        # tolerated: BufferProtocol requires step; this buffer always notifies with the full array.
        step: ScheduleStep[Any, Any, Any] | None = None,  # noqa: ARG002
        *,
        flush: bool = False,
    ) -> None:
        if self._throttled_update is not None:
            if flush and isinstance(self._throttled_update, MinIntervalFunc):
                self._throttled_update.flush(self.array)
            else:
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
    """Runtime scope for Python-like measurement orchestration.

    ScheduleStep and ProgramBuilder use the owner-aware cooperation methods to
    access buffers and record outcomes. Schedule alone owns scope validation,
    local-buffer registration, stop transitions and failed-attempt retry reset.
    """

    def __init__(
        self,
        init_cfg: T_Cfg,
        *buffers: BufferProtocol,
        env: T_Env | None = None,
        stop: StopSignal,
    ) -> None:
        self._ensure_single_root_buffer_count(len(buffers))
        self.cfg = deepcopy(init_cfg)
        # tolerated(cast): T_Env cannot express the required {} fallback for env=None.
        self._env = cast(T_Env, {} if env is None else env)
        self._buffers = list(buffers)
        self._local_buffers: dict[tuple[Hashable, ...], SignalBuffer] = {}
        self._outcome = ScheduleOutcome()
        self._is_active = False
        self._is_closed = False
        self._stop = stop

    def __enter__(self) -> Schedule[T_Cfg, T_Env]:
        if self._is_closed:
            raise RuntimeError("Schedule cannot be reused after exiting its context")
        if self._is_active:
            raise RuntimeError("Schedule context is already active")
        self._is_active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Close the scope using Python's exception type, value and traceback.

        Suppress KeyboardInterrupt and record interrupted on the shared stop
        signal, preserving buffered data for the caller's partial-result return.
        Do not suppress other exceptions or change their existing outcome.
        """
        self._is_active = False
        self._is_closed = True
        if isinstance(exc, KeyboardInterrupt):
            # Let the caller return buffered partial data after a Notebook stop.
            self.mark_interrupted(exc)
            return True
        return False

    @property
    def data(self) -> Any:
        return self.get_data_for(self)

    @property
    def env(self) -> T_Env:
        return self._env

    def register_buffer(self, *buffers: BufferProtocol) -> None:
        self.ensure_active()
        if not buffers:
            raise ValueError("register_buffer requires at least one SignalBuffer")
        self._ensure_single_root_buffer_count(len(self._buffers) + len(buffers))
        self._buffers.extend(buffers)

    def trigger_update(self, *, flush: bool = False) -> None:
        self.trigger_update_for(self, flush=flush)

    def set_data(self, value: Any, *, flush: bool = False) -> None:
        self.set_data_for(self, value, flush=flush)

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
        return self._stop.is_set()

    def set_stop(self) -> None:
        self._mark_stopped("stop requested")

    def clear_stop(self) -> None:
        self._stop.clear_stop()

    def scan(
        self, name: str, values: Iterable[T_Value]
    ) -> Iterator[tuple[T_Value, ScheduleStep[T_Cfg, T_Value, T_Env]]]:
        self.ensure_active()
        yield from self.scan_from(parent=self, name=name, values=values)

    def repeat(
        self, name: str, times: int, interval: float = 0.0
    ) -> Iterator[tuple[int, ScheduleStep[T_Cfg, int, T_Env]]]:
        self.ensure_active()
        yield from self.repeat_from(
            parent=self, name=name, times=times, interval=interval
        )

    def batch(
        self,
        tasks: Mapping[
            Hashable, Callable[[ScheduleStep[T_Cfg, Hashable, T_Env]], T_BatchResult]
        ],
    ) -> dict[Hashable, T_BatchResult]:
        self.ensure_active()
        return self.batch_from(parent=self, tasks=tasks)

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
        program_cls: ProgramFactory[ProgramT],
        **program_kwargs: object,
    ) -> ProgramBuilder[ProgramT]: ...

    def prog_builder(
        self,
        soc: object,
        soccfg: object,
        *,
        cfg: object | None = None,
        program_cls: ProgramFactory[ProgramT] | None = None,
        **program_kwargs: object,
    ) -> ProgramBuilder[ProgramT] | ProgramBuilder[ModularProgramV2]:
        self.ensure_active()
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

    def ensure_active(self) -> None:
        """Raise RuntimeError unless this schedule's context is currently active.

        Steps and builders must call this before starting scoped operations.
        """
        if not self._is_active:
            raise RuntimeError(
                "Schedule operations must run inside 'with Schedule(...)'"
            )

    def _mark_stopped(self, reason: str, *, exception: Exception | None = None) -> None:
        self._stop.set()
        self._set_outcome("stopped", reason=reason, exception=exception)
        if (
            exception is not None
            and self._outcome.status == "stopped"
            and self._outcome.exception is None
        ):
            # Enrich a prior stop without replacing its status or an earlier cause.
            self._outcome = replace(self._outcome, exception=exception)

    def mark_interrupted(self, exc: BaseException) -> None:
        """Stop the run after an interrupt without discarding buffered results.

        Record exc on the outcome and shared stop signal only if no prior
        non-completed outcome exists; otherwise preserve that outcome and cause.
        """
        reason = exception_reason(exc)
        if self._set_outcome("interrupted", reason=reason, exception=exc):
            self._stop.set_error("interrupted", reason, exc)
        else:
            self._stop.set()

    def mark_failed(self, exc: BaseException) -> None:
        """Stop the run after a failure without discarding buffered results.

        Record exc on the outcome and shared stop signal only if no prior
        non-completed outcome exists; otherwise preserve that outcome and cause.
        """
        reason = exception_reason(exc)
        if self._set_outcome("failed", reason=reason, exception=exc):
            self._stop.set_error("failed", reason, exc)
        else:
            self._stop.set()

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

    def check_stop_requested(self, *, exception: Exception | None = None) -> bool:
        """Return whether the shared stop signal is set and record a stopped run.

        Preserve earlier outcomes. If stopped has no cause yet, attach exception
        when supplied; a normal stop does not set the shared signal's error cause.
        """
        if self.is_stop():
            self._mark_stopped("stop requested", exception=exception)
            return True
        return False

    def prepare_retry_after_failed_attempt(self, attempt: int, retry: int) -> bool:
        """Reset a failed outcome and shared stop cause if another attempt remains.

        attempt is the zero-based failed attempt and retry is the allowed retry
        count. Return False without resetting stopped/interrupted outcomes or an
        exhausted budget. The builder checks external stop before marking failed.
        Partial buffer data is retained for the next attempt.
        """
        if self._outcome.status != "failed" or attempt >= retry:
            return False
        self._reset_outcome_for_retry()
        return True

    @staticmethod
    def _ensure_single_root_buffer_count(buffer_count: int) -> None:
        if buffer_count > 1:
            raise ValueError("Schedule supports at most one root result buffer")

    def scan_from(
        self,
        *,
        parent: Schedule[T_Cfg, T_Env] | ScheduleStep[T_Cfg, Any, T_Env],
        name: str,
        values: Iterable[T_Value],
    ) -> Iterator[tuple[T_Value, ScheduleStep[T_Cfg, T_Value, T_Env]]]:
        """Yield named scan children of parent until stop, pairing values and steps.

        parent must belong to this schedule; the caller ensures an active scope
        before iteration. Children deepcopy parent.cfg, share env and extend its
        path with the index. Each child's local buffers are released after yield;
        result data remains. Unsized values are materialized before iteration.
        """
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
                if self.check_stop_requested():
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

    def repeat_from(
        self,
        *,
        parent: Schedule[T_Cfg, T_Env] | ScheduleStep[T_Cfg, Any, T_Env],
        name: str,
        times: int,
        interval: float,
    ) -> Iterator[tuple[int, ScheduleStep[T_Cfg, int, T_Env]]]:
        """Yield named repeat children, checking stop before steps and while waiting.

        parent must belong to this schedule; the caller ensures an active scope
        before iteration. times and interval (seconds between steps) must be
        non-negative or ValueError is raised. Each child deepcopies parent.cfg,
        shares env and appends its index to the path. Release its local buffers
        after yield without clearing result data.
        """
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
                if self.check_stop_requested():
                    break

                while time.time() - start_t < interval:
                    if self.check_stop_requested():
                        break
                    passed_time = round(time.time() - start_t, 1)
                    time_pbar.set_progress(passed_time)
                    time.sleep(0.1)
                time_pbar.reset()

                if self.check_stop_requested():
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

    def batch_from(
        self,
        *,
        parent: Schedule[T_Cfg, T_Env] | ScheduleStep[T_Cfg, Any, T_Env],
        tasks: Mapping[
            Hashable, Callable[[ScheduleStep[T_Cfg, Hashable, T_Env]], T_BatchResult]
        ],
    ) -> dict[Hashable, T_BatchResult]:
        """Run tasks in mapping order with child steps and return completed results.

        parent must belong to this schedule; the caller ensures an active scope.
        Each child deepcopies parent.cfg, shares env and appends its task key to
        the path. Stop skips later tasks. Exception or KeyboardInterrupt records
        failure/interruption and returns prior results without retry. Release
        child-local buffers before and after each task, retaining written data.
        """
        results: dict[Hashable, T_BatchResult] = {}
        task_items = list(tasks.items())
        pbar = make_pbar(
            total=len(task_items),
            smoothing=0,
            leave=isinstance(parent, Schedule),
        )
        try:
            for key, child_fn in task_items:
                if self.check_stop_requested():
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
                    self.mark_interrupted(exc)
                    break
                # tolerated: Child callables may fail arbitrarily; retain prior results and the cause.
                except Exception as exc:  # noqa: BLE001
                    self._clear_local_buffers(step.path)
                    self.mark_failed(exc)
                    break

                self._clear_local_buffers(step.path)
                results[key] = result
                pbar.update()
                if self.check_stop_requested():
                    break
        finally:
            pbar.close()

        return results

    def get_data_for(
        self, owner: Schedule[Any, Any] | ScheduleStep[Any, Any, Any]
    ) -> Any:
        """Return the live result node at owner's path in this active schedule.

        owner must be this schedule or one of its steps. An inactive scope raises
        RuntimeError; an invalid path propagates the result container's error.
        """
        self.ensure_active()
        return get_path(self._data_root(), owner.path)

    def _data_root(self) -> Any:
        if len(self._buffers) == 1:
            return self._buffers[0].data
        if not self._buffers:
            return ()
        raise ValueError("Schedule supports at most one root result buffer")

    def set_data_for(
        self,
        owner: Schedule[Any, Any] | ScheduleStep[Any, Any, Any],
        value: Any,
        *,
        flush: bool = False,
    ) -> None:
        """Write value into owner's existing result node, then notify root buffers.

        owner must belong to this active schedule. Container assignment enforces
        supported value types and array shapes; lookup or write errors propagate
        before update. flush requests immediate delivery through the root hook.
        """
        target = self.get_data_for(owner)
        set_target(target, value)
        self.trigger_update_for(owner, flush=flush)

    def register_local_buffer(
        self,
        owner: Schedule[Any, Any] | ScheduleStep[Any, Any, Any],
        buffer: SignalBuffer,
    ) -> None:
        """Register buffer as the default acquire target at owner's result path.

        owner must belong to this active schedule. Raise ValueError unless its
        result node is an ndarray with buffer's shape. ScheduleStep.buffer wires
        updates back to that node; registration alone does not copy or notify.
        The enclosing scan/repeat/batch controls the registration's lifetime.
        """
        self.ensure_active()
        target = self.get_data_for(owner)
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

    def trigger_update_for(
        self,
        owner: Schedule[Any, Any] | ScheduleStep[Any, Any, Any],
        *,
        flush: bool = False,
    ) -> None:
        """Notify this active schedule's root buffers after an owner-level update.

        owner must belong to this schedule. Step updates pass owner to the hook;
        root updates pass None. flush requests immediate delivery. No registered
        root buffer is a no-op; inactive scopes raise RuntimeError.
        """
        self.ensure_active()
        if not self._buffers:
            return
        step = owner if isinstance(owner, ScheduleStep) else None
        for buffer in self._buffers:
            if flush:
                buffer.trigger_update(step, flush=True)
            else:
                buffer.trigger_update(step)

    def _clear_local_buffers(self, prefix: tuple[Hashable, ...]) -> None:
        for path in list(self._local_buffers):
            if path[: len(prefix)] == prefix:
                self._local_buffers.pop(path, None)

    def default_slot_for(
        self, owner: Schedule[T_Cfg, T_Env] | ScheduleStep[T_Cfg, Any, T_Env]
    ) -> SignalSlot:
        """Resolve the writable acquire slot for an owner in this active schedule.

        Prefer its registered local buffer. Otherwise require one SignalBuffer
        root and an integer-indexed owner path, raising ValueError if unavailable.
        owner must belong to this schedule. The slot retains its buffer's update
        hook, including child-to-root synchronization for a local buffer.
        """
        self.ensure_active()
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
        return self.schedule.get_data_for(self)

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
        self.schedule.set_data_for(self, value, flush=flush)

    def trigger_update(self, *, flush: bool = False) -> None:
        self.schedule.trigger_update_for(self, flush=flush)

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
        self.schedule.ensure_active()
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
        self.schedule.ensure_active()
        buffer = SignalBuffer(
            shape,
            dtype=dtype,
            on_update=lambda data: self.set_data(data),
            update_interval=None,
        )
        self.schedule.register_local_buffer(self, buffer)
        return buffer

    def scan(
        self, name: str, values: Iterable[T_NestedValue]
    ) -> Iterator[tuple[T_NestedValue, ScheduleStep[T_Cfg, T_NestedValue, T_Env]]]:
        self.schedule.ensure_active()
        yield from self.schedule.scan_from(parent=self, name=name, values=values)

    def repeat(
        self, name: str, times: int, interval: float = 0.0
    ) -> Iterator[tuple[int, ScheduleStep[T_Cfg, int, T_Env]]]:
        self.schedule.ensure_active()
        yield from self.schedule.repeat_from(
            parent=self, name=name, times=times, interval=interval
        )

    def batch(
        self,
        tasks: Mapping[
            Hashable, Callable[[ScheduleStep[T_Cfg, Hashable, T_Env]], T_BatchResult]
        ],
    ) -> dict[Hashable, T_BatchResult]:
        self.schedule.ensure_active()
        return self.schedule.batch_from(parent=self, tasks=tasks)

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
        program_cls: ProgramFactory[ProgramT],
        **program_kwargs: object,
    ) -> ProgramBuilder[ProgramT]: ...

    def prog_builder(
        self,
        soc: object,
        soccfg: object,
        *,
        cfg: object | None = None,
        program_cls: ProgramFactory[ProgramT] | None = None,
        **program_kwargs: object,
    ) -> ProgramBuilder[ProgramT] | ProgramBuilder[ModularProgramV2]:
        self.schedule.ensure_active()
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
