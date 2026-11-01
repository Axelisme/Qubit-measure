"""Typed Schedule acquisition entrypoints without program assembly."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, Unpack

import numpy as np
from numpy.typing import NDArray

from zcu_tools.experiment.v2.runtime.acquisition import (
    AcquireOptions,
    AcquisitionRunner,
    SignalArray,
    StopCondition,
    SynchronousAcquireKeywords,
    default_decimated_raw2signal_fn,
    default_raw2signal_fn,
    require_positive_rounds,
)
from zcu_tools.program.acquisition import ShotSumProcessor

if TYPE_CHECKING:
    from zcu_tools.experiment.v2.runtime.schedule import Schedule, ScheduleStep

T_Raw = TypeVar("T_Raw")
T_Raw_co = TypeVar("T_Raw_co", covariant=True)


@dataclass(frozen=True)
class RunOptions(Generic[T_Raw]):
    """Policy for Schedule run acquisition and slot conversion.

    retry is the nonnegative number of extra attempts (default zero), including
    shot overrun. Zero never retries. Exhausted producer failures propagate as
    the original exception; concurrent cancellation does not suppress them.
    Overrun restores the pre-acquire slot. External stop and interruption never
    retry. progress=False hides native acquisition progress; the rounds bar uses
    label "rounds" and remains visible only at the
    root. stop_condition is a no-argument callback run after each completed round
    has been converted and written. True or StopDecision(met=True) stops this
    acquire, not the Schedule. Finite StopDecision values are logged on early
    completion. converter maps cumulative and final raw payloads to the owner's
    slot shape. None selects the default: integrated requires one ADC and one
    read per shot; decimated requires one ADC and retains all repetition/read/
    sample axes; summed writes sums unchanged. Ambiguous defaults raise ValueError
    requesting an explicit converter. run_acquire_shots instead returns retained
    raw shots without a slot and rejects converter and stop_condition policies.
    Its rounds count must be one. Converter and live update exceptions
    propagate without retry or a failed acquisition outcome.
    on_stop optionally observes external cancellation once per cooperative return,
    after acquire scope release and progress cleanup, before returning the signal
    view or raw-shot stop result. It also runs when already stopped before
    acquisition. Pure data stop,
    producer/converter/update failures and KeyboardInterrupt do not call it.
    Observer exceptions propagate unchanged without retry or outcome changes.
    """

    retry: int = 0
    progress: bool = False
    stop_condition: StopCondition | None = None
    converter: Callable[[T_Raw], SignalArray] | None = None
    on_stop: Callable[[], None] | None = None


def _adapt_run_options(
    options: RunOptions[T_Raw] | None, qick_options: dict[str, object]
) -> tuple[RunOptions[T_Raw], int, AcquireOptions]:
    """Select policy, consume validated rounds and adapt the runner options."""
    selected = RunOptions[T_Raw]() if options is None else options
    rounds = require_positive_rounds(qick_options.pop("rounds", 1))
    return (
        selected,
        rounds,
        AcquireOptions(
            retry=selected.retry,
            progress=selected.progress,
            stop_condition=selected.stop_condition,
            on_stop=selected.on_stop,
        ),
    )


class IntegratedProgram(Protocol[T_Raw_co]):
    """Producer requiring only synchronous acquire, not other acquisition modes.

    Return cumulative raw data and call round_hook after each completed round
    with one-based round index, cumulative raw data and the supplied cancel flag.
    Check cancel_flag before another round. Cancellation retains completed data;
    before any round raise StoppedPartialAcquireError. Native keywords follow
    SynchronousAcquireKeywords. Other producer exceptions use the runner's retry
    budget, then propagate unchanged instead of returning partial data.
    """

    def acquire(
        self, soc: object, **kwargs: Unpack[SynchronousAcquireKeywords[T_Raw_co]]
    ) -> T_Raw_co: ...


class ShotProgram(Protocol):
    """Synchronous acquire producer retaining integer raw shots.

    acquire(soc, **kwargs) accepts SynchronousAcquireKeywords, returns averaged
    IQ data and follows IntegratedProgram's cancellation and failure contract.
    get_raw() returns the retained per-ADC int64 shot buffers, or None when no
    raw buffer is available. The runner returns those buffers without conversion.
    """

    def acquire(
        self,
        soc: object,
        **kwargs: Unpack[SynchronousAcquireKeywords[list[NDArray[np.float64]]]],
    ) -> list[NDArray[np.float64]]: ...

    def get_raw(self) -> list[NDArray[np.int64]] | None:
        """Return retained per-ADC raw shots, or None if unavailable."""
        ...


class DecimatedProgram(Protocol[T_Raw_co]):
    """Producer requiring only synchronous acquire_decimated.

    Raw cumulative traces and round hooks follow IntegratedProgram. Native
    keywords follow SynchronousAcquireKeywords.
    """

    def acquire_decimated(
        self,
        soc: object,
        **kwargs: Unpack[SynchronousAcquireKeywords[T_Raw_co]],
    ) -> T_Raw_co: ...


class SummedProgram(Protocol):
    """Producer requiring only joint-ADC acquire_summed.

    processor maps normalized ADC-keyed (points, reps, reads, 2) IQ to float64
    (points, *axes). Return cumulative unnormalized (*point_dims, *axes) sums,
    or None only in unsupported stepped mode. Round hooks, cancellation and
    failures follow IntegratedProgram's synchronous contract.
    """

    def acquire_summed(
        self,
        soc: object,
        *,
        processor: ShotSumProcessor,
        rounds: int = 1,
        step_rounds: bool = False,
        **qick_options: object,
    ) -> NDArray[np.float64] | None: ...


class RunAcquisitionMixin(ABC):
    """Schedule and step acquisition methods sharing one execution policy.

    Subclasses provide the existing owner via _acquisition_owner; this mixin owns
    no scope, slot or outcome state. Slot entrypoints write converted data and
    trigger live updates, with acquire-local data stops. run_acquire_shots
    instead returns retained raw buffers without conversion or slot writes.
    All entrypoints require an active Schedule and retry producer errors.
    Only slot entrypoints require a writable SignalBuffer slot. Cancellation
    returns partial slot data or None for shots; only a stopped Schedule token
    records a stopped outcome. KeyboardInterrupt propagates unchanged without retry.
    Exhausted producer errors propagate unchanged;
    conversion/update errors propagate without retry.
    """

    @property
    @abstractmethod
    def _acquisition_owner(
        self,
    ) -> Schedule[Any] | ScheduleStep[Any, Any]:
        """Return this Schedule or step; all state remains owned by that object."""
        ...

    def run_acquire(
        self,
        program: IntegratedProgram[list[NDArray[np.float64]]],
        soc: object,
        *,
        options: RunOptions[list[NDArray[np.float64]]] | None = None,
        **qick_options: object,
    ) -> SignalArray:
        """Acquire IQ, write this scope's slot and trigger live updates.

        Run means slot/live updates, automatic retry via options.retry, and
        data-driven stopping via options.stop_condition. Native program acquire
        only returns data; this method owns the Schedule run side effects.
        Requires an active Schedule and a writable default SignalBuffer slot.
        program supplies synchronous acquire; soc is its borrowed transport.
        options controls retry, native progress, stop and raw conversion. With
        no converter, require one ADC and one read per shot, otherwise raise
        ValueError requesting an explicit converter. Final-axis IQ becomes complex.
        Native qick_options are forwarded. rounds defaults to one and must be a positive
        integer excluding bool. Return the live slot, including partial data on
        cancellation. KeyboardInterrupt and exhausted producer errors propagate unchanged.
        Converter and update errors propagate without retry.
        """
        owner = self._acquisition_owner
        selected, rounds, acquire_options = _adapt_run_options(options, qick_options)
        return AcquisitionRunner(owner, soc).run(
            acquire=program.acquire,
            rounds=rounds,
            signal_fn=(
                default_raw2signal_fn
                if selected.converter is None
                else selected.converter
            ),
            options=acquire_options,
            acquire_kwargs=qick_options,
        )

    def run_acquire_shots(
        self,
        program: ShotProgram,
        soc: object,
        *,
        options: RunOptions[list[NDArray[np.float64]]] | None = None,
        **qick_options: object,
    ) -> list[NDArray[np.int64]] | None:
        """Acquire once and return program.get_raw() without conversion or slot writes.

        program supplies synchronous acquire and retained int64 raw shots; soc
        is its borrowed transport. Requires an active Schedule, not a buffer.
        options selects retry, native progress and on_stop. qick_options are
        forwarded to acquire. Return None on cooperative external cancellation;
        only a stopped Schedule token records a stopped Schedule outcome.
        on_stop runs once after acquire scope release and progress cleanup.
        KeyboardInterrupt and exhausted producer errors propagate unchanged.
        Producer failures, including shot overrun, use the nonnegative retry
        budget without slot restoration. A partial-acquire error without an
        external stop propagates unchanged after retry is exhausted.
        ValueError rejects negative retry, converter or stop_condition policies,
        and rounds other than the single positive integer 1. RuntimeError means
        program.get_raw() provided no raw shot buffer after successful acquire.
        Progress callback errors propagate unchanged without retry or marking
        acquisition failed, after acquire scope release and progress cleanup;
        they do not call on_stop. Neither raw-buffer access failures nor on_stop
        failures are retried.
        """
        selected, rounds, acquire_options = _adapt_run_options(options, qick_options)
        if selected.converter is not None or selected.stop_condition is not None:
            raise ValueError("run_acquire_shots does not convert or apply data stop")
        if rounds != 1:
            raise ValueError("run_acquire_shots requires rounds=1")
        completed = AcquisitionRunner(self._acquisition_owner, soc).run_shots(
            acquire=program.acquire,
            options=acquire_options,
            acquire_kwargs=qick_options,
        )
        if not completed:
            return None
        shots = program.get_raw()
        if shots is None:
            raise RuntimeError("program did not provide a raw shot buffer")
        return shots

    def run_acquire_decimated(
        self,
        program: DecimatedProgram[list[NDArray[np.float64]]],
        soc: object,
        *,
        options: RunOptions[list[NDArray[np.float64]]] | None = None,
        **qick_options: object,
    ) -> SignalArray:
        """Acquire traces, write this scope's slot and trigger live updates.

        Run means slot/live updates, automatic retry via options.retry, and
        data-driven stopping via options.stop_condition. Native program acquire
        only returns data; this method owns the Schedule run side effects.
        Requires an active Schedule and a writable default SignalBuffer slot.
        program supplies synchronous acquire_decimated; soc is its transport.
        options controls retry, native progress, stop and raw conversion. With
        no converter, require one ADC, otherwise raise ValueError requesting an
        explicit converter. Final-axis IQ becomes complex, retaining all
        repetition, read and sample axes. Native qick_options are forwarded.
        rounds defaults to one and must be a positive
        integer excluding bool. Return the live slot, including partial data on
        cancellation. KeyboardInterrupt and exhausted producer errors propagate unchanged.
        Converter and update errors propagate without retry.
        """
        owner = self._acquisition_owner
        selected, rounds, acquire_options = _adapt_run_options(options, qick_options)
        return AcquisitionRunner(owner, soc).run(
            acquire=program.acquire_decimated,
            rounds=rounds,
            signal_fn=(
                default_decimated_raw2signal_fn
                if selected.converter is None
                else selected.converter
            ),
            options=acquire_options,
            acquire_kwargs=qick_options,
        )

    def run_acquire_summed(
        self,
        program: SummedProgram,
        soc: object,
        *,
        processor: ShotSumProcessor,
        options: RunOptions[NDArray[np.float64]] | None = None,
        **qick_options: object,
    ) -> SignalArray:
        """Acquire shot sums, write this scope's slot and trigger live updates.

        Run means slot/live updates, automatic retry via options.retry, and
        data-driven stopping via options.stop_condition. Native program acquire
        only returns data; this method owns the Schedule run side effects.
        Requires an active Schedule and a writable default SignalBuffer slot.
        program needs only acquire_summed; soc is its transport. processor maps
        normalized ADC-keyed (points, reps, reads, 2) IQ to float64 (points, *axes).
        options selects retry/progress/stop and conversion; converter=None writes
        cumulative and final sums directly without averaging. Native qick_options
        are forwarded. rounds defaults to one and must be a positive integer
        excluding bool. Return the live slot, retaining completed rounds on
        cancellation. KeyboardInterrupt and exhausted producer errors propagate unchanged.
        None producer results, conversion and update failures propagate without retry.
        """
        owner = self._acquisition_owner
        selected, rounds, acquire_options = _adapt_run_options(options, qick_options)

        def convert(raw: NDArray[np.float64] | None) -> SignalArray:
            if raw is None:
                raise ValueError("run summed acquisition requires synchronous results")
            return raw if selected.converter is None else selected.converter(raw)

        return AcquisitionRunner(owner, soc).run(
            acquire=program.acquire_summed,
            rounds=rounds,
            signal_fn=convert,
            options=acquire_options,
            acquire_kwargs=dict(qick_options, processor=processor),
        )
