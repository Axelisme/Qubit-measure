"""Shared synchronous acquisition policy for builders and Schedule methods."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from logging import getLogger
from math import isfinite
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    NotRequired,
    TypeAlias,
    TypedDict,
)

import numpy as np
from numpy.typing import NDArray
from typing_extensions import ReadOnly, TypeVar

from zcu_tools.experiment.acquisition_facts import StopDecision
from zcu_tools.experiment.v2.runtime.acquire_cancel import AcquireCancelFlag
from zcu_tools.program.acquisition import (
    AcquisitionContractError,
    CancelFlagProtocol,
    StoppedPartialAcquireError,
)
from zcu_tools.progress_bar import BaseProgressBar, make_pbar

if TYPE_CHECKING:
    from zcu_tools.experiment.v2.runtime.schedule import (
        Schedule,
        ScheduleStep,
        SignalSlot,
    )

T_Raw = TypeVar("T_Raw")
T_KeywordRaw_contra = TypeVar("T_KeywordRaw_contra", contravariant=True)
SignalArray: TypeAlias = NDArray[Any]
StopCondition: TypeAlias = Callable[[], bool | StopDecision]
# Preserve the existing logging channel for builder clients.
logger = getLogger("zcu_tools.experiment.v2.runtime.program_builder")


def require_positive_rounds(rounds: object) -> int:
    """Return rounds as a positive integer, excluding bool; otherwise raise ValueError."""
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    return rounds


class _AcquireStopped(Exception):
    """An acquire child was cancelled; retain its completed-round slot."""


class _CallbackError(Exception):
    """Carry callback origin across the producer's synchronous acquire boundary."""

    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.error = error


class SynchronousAcquireKeywords(TypedDict, Generic[T_KeywordRaw_contra]):
    """Keywords required by a synchronous builder-managed acquisition.

    rounds is the software average count and progress controls native progress.
    round_hook receives each completed-round summary and its cancel flag.
    Its keyword entry is read-only: consumers call the supplied hook without
    replacing it. The callback consumes raw data, making these keywords
    contravariant in that data. Passing them as method arguments preserves
    the producer protocol's raw-result covariance.
    cancel_flag also carries external stop requests. Omitted step_rounds keeps
    synchronous operation; stepped preparation is outside this producer seam.
    This protocol does not promise arbitrary keyword acceptance. The builder
    still forwards producer-specific keywords unchanged at runtime.
    """

    rounds: int
    progress: bool
    round_hook: ReadOnly[
        Callable[[int, T_KeywordRaw_contra, CancelFlagProtocol], object]
    ]
    cancel_flag: CancelFlagProtocol
    step_rounds: NotRequired[Literal[False]]


def default_raw2signal_fn(raw: Sequence[NDArray[np.float64]]) -> NDArray[np.complex128]:
    """Convert a single ADC/read's final-axis IQ pairs to complex signals.

    raw follows integrated program acquisition layout: ADC, read, then signal
    axes ending in a length-two I/Q axis. Require exactly one ADC and one read
    per shot, otherwise raise ValueError requesting an explicit converter.
    Other invalid layouts raise NumPy or indexing errors. raw is not modified.
    """
    read_counts = [adc.shape[0] for adc in raw]
    if len(raw) != 1 or read_counts != [1]:
        raise ValueError(
            f"Default integrated conversion: ADC count={len(raw)}, "
            f"read counts={read_counts}; provide an explicit converter "
            "unless there is exactly one ADC and one read per shot"
        )
    return raw[0][0].dot([1, 1j])


def default_decimated_raw2signal_fn(
    raw: Sequence[NDArray[np.float64]],
) -> NDArray[np.complex128]:
    """Convert a single ADC's decimated IQ traces to a complex array.

    raw contains ADC arrays ending in a length-two I/Q axis. Require exactly one
    ADC, otherwise raise ValueError requesting an explicit converter. Preserve
    all repetition, read and sample axes. Other invalid layouts raise NumPy or
    indexing errors. raw is not modified.
    """
    if len(raw) != 1:
        raise ValueError(
            f"Default decimated conversion: ADC count={len(raw)}; "
            "provide an explicit converter unless there is exactly one ADC"
        )
    return raw[0].dot([1, 1j])


@dataclass(frozen=True)
class AcquireOptions:
    """Policy shared by integrated and decimated program acquisition.

    retry is a non-negative number of extra attempts, default zero, including
    shot overrun (AcquisitionContractError). Zero never retries. Exhausted
    producer failures propagate unchanged, even if cancellation was also requested.
    Each retry
    reruns all rounds on the same program with
    fresh producer buffers; overrun discards the failed attempt's slot data.
    External cancellation and interruption are never retried. progress
    defaults to hiding native acquisition progress. progress_label names the
    rounds bar (default "rounds"). progress_leave=None keeps the completed bar
    visible at the root and removes it for a host step; explicit True/False
    overrides this display policy. Every bar closes before acquire returns.
    stop_condition=None disables data-driven early stop; otherwise the callback
    runs after each completed round is converted and written to the buffer.
    Returning True or StopDecision(met=True) requests acquire-local cancellation
    before the next round. StopDecision additionally supplies a scalar observation
    logged if finite and acquisition actually ends early. Each retry logs its
    reason and next attempt number at warning level; early stops log the stop
    value and completed rounds at info level.
    """

    retry: int = 0
    progress: bool = False
    progress_label: str = "rounds"
    progress_leave: bool | None = None
    stop_condition: StopCondition | None = None


@dataclass
class _AcquireProgress:
    pbar: BaseProgressBar
    attempts: int = 0
    completed_rounds: int = 0
    decision: bool | StopDecision = False

    def retry_after(
        self,
        error: Exception,
        retry: int,
        slot: SignalSlot,
        initial_signal: SignalArray,
    ) -> bool:
        overrun = isinstance(error, AcquisitionContractError)
        if overrun:
            slot.set(initial_signal)
        if self.attempts > retry:
            return False
        reason = "shot_overrun" if overrun else type(error).__name__
        logger.warning(
            "Retrying acquisition, attempt %s after %s: %s",
            self.attempts + 1,
            reason,
            error,
        )
        return True

    def log_early_stop(self, requested_rounds: int) -> None:
        if not self.decision or self.completed_rounds >= requested_rounds:
            return
        value = self.decision.value if isinstance(self.decision, StopDecision) else None
        logger.info(
            "Acquisition stopped early: stop value %s, completed rounds %s",
            value if value is not None and isfinite(value) else None,
            self.completed_rounds,
        )


class AcquisitionRunner:
    """Execute shared retry, round hooks and partial-result policy without building.

    owner selects an active Schedule or step's writable default slot. soc is the
    borrowed producer transport. No program assembly or module state is held.
    """

    def __init__(
        self, owner: Schedule[Any, Any] | ScheduleStep[Any, Any, Any], soc: object
    ) -> None:
        from zcu_tools.experiment.v2.runtime.schedule import Schedule

        self._owner = owner
        self._schedule = owner if isinstance(owner, Schedule) else owner.schedule
        self._soc = soc

    def run(
        self,
        *,
        acquire: Callable[..., T_Raw],
        rounds: int,
        signal_fn: Callable[[T_Raw], SignalArray],
        options: AcquireOptions,
        acquire_kwargs: dict[str, object],
    ) -> SignalArray:
        """Acquire synchronously into the owner's live slot and return its view.

        acquire borrows soc and receives rounds, native progress, round_hook,
        cancel_flag and acquire_kwargs. signal_fn converts each cumulative round
        and final raw payload to the slot shape. options selects retry, progress
        and data stop. rounds must be a positive int (not bool); retry must be
        nonnegative. The owner must be active and have a default SignalBuffer slot.
        Interrupt or cancellation records the Schedule outcome and returns partial
        data. Producer failures use the explicit retry budget, including zero
        retries, then propagate unchanged. A concurrent stop prevents retry but
        does not suppress the failure. Overrun restores the pre-acquire slot.
        Conversion/update errors propagate without retry.
        """
        require_positive_rounds(rounds)
        retry = options.retry
        if retry < 0:
            raise ValueError("retry must be non-negative")
        self._schedule.ensure_active()

        slot = self._default_slot()
        initial_signal = slot.view.copy()

        pbar = make_pbar(
            total=rounds,
            smoothing=0,
            desc=options.progress_label,
            leave=self._resolve_progress_leave(progress_leave=options.progress_leave),
            disable=rounds == 1,
        )
        observation = _AcquireProgress(pbar)
        try:
            for _ in range(retry + 1):
                if self._schedule.check_stop_requested():
                    break
                pbar.reset()
                observation.attempts += 1
                observation.completed_rounds = 0
                observation.decision = False
                try:
                    raw = self._acquire_once(
                        acquire=acquire,
                        options=options,
                        acquire_kwargs=dict(acquire_kwargs, rounds=rounds),
                        signal_fn=signal_fn,
                        slot=slot,
                        observation=observation,
                    )
                except _AcquireStopped:
                    self._schedule.check_stop_requested()
                    break
                except _CallbackError as exc:
                    raise exc.error from None
                except KeyboardInterrupt as exc:
                    self._schedule.mark_interrupted(exc)
                    break
                # Retry producer failures; exhausted errors propagate unchanged.
                except Exception as exc:
                    if self._schedule.check_stop_requested(exception=exc):
                        raise
                    if not observation.retry_after(exc, retry, slot, initial_signal):
                        self._schedule.mark_failed(exc)
                        raise
                    continue

                if self._schedule.check_stop_requested():
                    break

                pbar.set_progress(rounds)
                slot.set(signal_fn(raw))
                observation.log_early_stop(rounds)
                break
        except KeyboardInterrupt as exc:
            self._schedule.mark_interrupted(exc)
        finally:
            pbar.close()

        return slot.view

    def _acquire_once(
        self,
        *,
        acquire: Callable[..., T_Raw],
        options: AcquireOptions,
        acquire_kwargs: dict[str, object],
        signal_fn: Callable[[T_Raw], SignalArray],
        slot: SignalSlot,
        observation: _AcquireProgress,
    ) -> T_Raw:
        scope = self._schedule.scope.branch("acquire")
        token = scope.stop
        acquire_cancel_flag = AcquireCancelFlag(token)

        def update_hook(ir: int, raw: T_Raw, cancel_flag: CancelFlagProtocol) -> None:
            observation.completed_rounds = ir
            try:
                observation.pbar.set_progress(ir)
                slot.set(signal_fn(raw))
                if options.stop_condition is not None:
                    observation.decision = options.stop_condition()
                    if observation.decision:
                        cancel_flag.set()
            except Exception as exc:
                raise _CallbackError(exc) from exc

        try:
            try:
                raw = acquire(
                    self._soc,
                    progress=options.progress,
                    round_hook=update_hook,
                    cancel_flag=acquire_cancel_flag,
                    **dict(acquire_kwargs),
                )
            except StoppedPartialAcquireError as exc:
                if token.is_set():
                    raise _AcquireStopped from exc
                raise
            if token.is_set():
                raise _AcquireStopped
            return raw
        finally:
            scope.release()

    def _resolve_progress_leave(self, *, progress_leave: bool | None) -> bool:
        # Import the concrete step at call time to avoid a module initialization cycle.
        from zcu_tools.experiment.v2.runtime.schedule import ScheduleStep

        if progress_leave is not None:
            return progress_leave
        return not isinstance(self._owner, ScheduleStep)

    def _default_slot(self) -> SignalSlot:
        return self._schedule.default_slot_for(self._owner)
