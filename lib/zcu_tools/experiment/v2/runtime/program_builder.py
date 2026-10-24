"""Program assembly and synchronous acquisition within a Schedule scope."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, Self, TypeAlias

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeVar

from zcu_tools.experiment.v2.runtime.acquire_cancel import AcquireCancelFlag
from zcu_tools.program.acquisition import CancelFlagProtocol, StoppedPartialAcquireError
from zcu_tools.program.v2 import (
    Module,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
)
from zcu_tools.progress_bar import BaseProgressBar, make_pbar

if TYPE_CHECKING:
    from zcu_tools.experiment.v2.runtime.schedule import (
        Schedule,
        ScheduleStep,
        SignalSlot,
    )

T_Raw = TypeVar("T_Raw")
T_AcquireRaw_co = TypeVar("T_AcquireRaw_co", covariant=True)
T_DecimatedRaw_co = TypeVar("T_DecimatedRaw_co", covariant=True)
SignalArray: TypeAlias = NDArray[Any]
StopCondition: TypeAlias = Callable[[], bool]


def _require_positive_rounds(rounds: object) -> None:
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be a positive integer")


class _AcquireStopped(Exception):
    """An acquire child was cancelled; retain its completed-round slot."""


class _CallbackError(Exception):
    """Carry callback origin across the producer's synchronous acquire boundary."""

    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.error = error


class ProgramProtocol(Protocol[T_AcquireRaw_co, T_DecimatedRaw_co]):
    """Synchronous producer accepted by the builder's run methods.

    rounds supplies the positive integer software average count. Both methods receive
    the caller's soc, native progress policy and extra acquisition keywords.
    They call round_hook with a one-based completed round, cumulative raw data
    and the supplied cancel_flag; they check that flag before starting another
    round. Return the final raw data, retaining completed rounds on cancellation.
    Raise StoppedPartialAcquireError if cancelled before any round completes.
    This synchronous contract uses the default step_rounds=False mode; stepped
    preparation is not a completed raw result. Other producer errors propagate
    to the builder's partial-result policy.
    """

    def acquire(
        self,
        soc: object,
        *,
        rounds: int,
        progress: bool,
        round_hook: Callable[[int, T_AcquireRaw_co, CancelFlagProtocol], object],
        cancel_flag: CancelFlagProtocol,
        step_rounds: Literal[False] = False,
        **kwargs: object,
    ) -> T_AcquireRaw_co: ...

    def acquire_decimated(
        self,
        soc: object,
        *,
        rounds: int,
        progress: bool,
        round_hook: Callable[[int, T_DecimatedRaw_co, CancelFlagProtocol], object],
        cancel_flag: CancelFlagProtocol,
        step_rounds: Literal[False] = False,
        **kwargs: object,
    ) -> T_DecimatedRaw_co: ...


ProgramT = TypeVar("ProgramT", bound=ProgramProtocol[Any, Any])
ProgramFactory: TypeAlias = Callable[..., ProgramT]


def default_raw2signal_fn(raw: Sequence[NDArray[np.float64]]) -> NDArray[np.complex128]:
    """Convert the first receiver/readout's final-axis IQ pairs to complex signals.

    raw follows integrated program acquisition layout: receiver, readout, then
    signal axes ending in a length-two I/Q axis. Invalid layouts raise NumPy or
    indexing errors; this function does not modify raw.
    """
    return raw[0][0].dot([1, 1j])


def default_decimated_raw2signal_fn(
    raw: Sequence[NDArray[np.float64]],
) -> NDArray[np.complex128]:
    """Convert the first receiver's decimated IQ trace to a complex array.

    raw contains receiver arrays ending in a length-two I/Q axis. Invalid layouts
    raise NumPy or indexing errors; this function does not modify raw.
    """
    return raw[0].dot([1, 1j])


class ModuleFacade(ABC):
    """Fluent module assembly shared by the program builder's typed helpers."""

    @abstractmethod
    def add(self, *modules: Module | Sequence[Module]) -> Self:
        """Append modules or ordered module groups and return this builder."""
        ...

    def add_reset(self, name: str, cfg: ResetCfg | None) -> Self:
        """Append a named Reset from cfg (None disables it) and return self."""
        return self.add(Reset(name, cfg))

    def add_pulse(self, name: str, cfg: PulseCfg | None, **kwargs: Any) -> Self:
        """Append a named Pulse; None cfg emits no pulse.

        kwargs accepts Pulse's tag (timing label), pulse_id (explicit registry
        identity), block_mode (wait through pre/post delays, default True) and
        ro_ch (optional ADC channel for frequency matching). Invalid keywords
        or cfg propagate Pulse construction errors.
        """
        return self.add(Pulse(name, cfg, **kwargs))

    def add_readout(self, name: str, cfg: ReadoutCfg) -> Self:
        """Append a named Readout from cfg and return this builder."""
        return self.add(Readout(name, cfg))


@dataclass(frozen=True)
class AcquireOptions:
    """Policy shared by integrated and decimated program acquisition.

    retry is a non-negative number of extra attempts, default zero. progress
    defaults to hiding native acquisition progress. progress_label names the
    rounds bar (default "rounds"). progress_leave=None keeps the completed bar
    visible at the root and removes it for a host step; explicit True/False
    overrides this display policy. Every bar closes before acquire returns.
    stop_condition=None disables data-driven early stop; otherwise the callback
    runs after each completed round is converted and written to the buffer.
    Returning True requests acquire-local cancellation before the next round.
    """

    retry: int = 0
    progress: bool = False
    progress_label: str = "rounds"
    progress_leave: bool | None = None
    stop_condition: StopCondition | None = None


class ProgramBuilder(ModuleFacade, Generic[ProgramT]):
    """Build and synchronously acquire a program within an active Schedule.

    owner is the root Schedule or a ScheduleStep and selects the default writable
    SignalBuffer slot. soc and soccfg are borrowed producer handles.
    program_kwargs contains the positive integer reps control and initial_delay /
    relax_delay in us, as supplied by Schedule.prog_builder. program_cls receives
    soccfg, these keyword controls, modules, sweep and extra compilation keywords. Build deep-copies modules and sweep specs.
    Acquisition rounds default to one and never come from the owner's cfg.

    Build/run operations require an active owner scope and a configured default
    slot for acquisition. Build methods require at least one explicit module.
    Invalid scope, missing modules/slot or negative retry raises immediately.
    All build methods propagate factory errors without retry. Retry applies only
    to acquire on the same built program. Conversion and update callback errors
    propagate without marking an acquire failure.

    All acquire methods write each converted completed round before checking
    stop_condition, then write the final converted result. Data-driven stop is
    acquire-local; external stop, interrupt and exhausted failure retain partial
    data (unwritten slots stay NaN) and record the Schedule outcome. They return
    the slot's live array rather than raising producer errors. Progress bars close
    before return, including on interruption or failure.
    """

    def __init__(
        self,
        *,
        owner: Schedule[Any, Any] | ScheduleStep[Any, Any, Any],
        soc: object,
        soccfg: object,
        program_cls: ProgramFactory[ProgramT],
        program_kwargs: dict[str, object],
    ) -> None:
        # Import the concrete owner at call time to avoid a module initialization cycle.
        from zcu_tools.experiment.v2.runtime.schedule import Schedule

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
        """Append modules or ordered module groups without copying until build."""
        for module_or_group in modules:
            if isinstance(module_or_group, Module):
                self._modules.append(module_or_group)
            else:
                self._modules.extend(module_or_group)
        return self

    def declare_sweep(self, name: str, spec: SweepCfg | int) -> Self:
        """Append a named program-side sweep or loop count, copied at build time."""
        self._sweeps.append((name, spec))
        return self

    def set_raw2signal_fn(self, fn: Callable[[T_Raw], SignalArray]) -> Self:
        """Set the fallback raw-to-slot-array converter for both acquire modes."""
        self._raw2signal_fn = fn
        return self

    def build(self) -> ProgramT:
        """Return a new program with isolated modules/sweeps; factory errors propagate."""
        self._schedule.ensure_active()
        self._ensure_modules()
        return self._build_program()

    def build_and_acquire(
        self,
        *,
        rounds: int = 1,
        raw2signal_fn: Callable[[Any], SignalArray] | None = None,
        options: AcquireOptions | None = None,
        **acquire_kwargs: object,
    ) -> SignalArray:
        """Build and run integrated acquisition with options and producer kwargs.

        raw2signal_fn overrides the builder converter and must return a value
        compatible with the owner's slot. None uses the builder/default converter.
        See the class contract for retry, stop and partial-result behavior.
        """
        self._schedule.ensure_active()
        self._ensure_modules()
        return self._build_and_run(
            rounds=rounds,
            runner=self.run_program,
            raw2signal_fn=raw2signal_fn,
            options=AcquireOptions() if options is None else options,
            acquire_kwargs=acquire_kwargs,
        )

    def build_and_acquire_decimated(
        self,
        *,
        rounds: int = 1,
        raw2signal_fn: Callable[[Any], SignalArray] | None = None,
        options: AcquireOptions | None = None,
        **acquire_kwargs: object,
    ) -> SignalArray:
        """Build and run decimated acquisition under the same policy as integrated.

        raw2signal_fn maps decimated raw data to the owner's slot shape; None
        uses the builder converter or default_decimated_raw2signal_fn. options
        controls retry/progress/stop; acquire_kwargs pass to the producer.
        """
        self._schedule.ensure_active()
        self._ensure_modules()
        return self._build_and_run(
            rounds=rounds,
            runner=self.run_program_decimated,
            raw2signal_fn=raw2signal_fn,
            options=AcquireOptions() if options is None else options,
            acquire_kwargs=acquire_kwargs,
        )

    def run_program(
        self,
        program: ProgramProtocol[T_Raw, object],
        *,
        rounds: int = 1,
        raw2signal_fn: Callable[[T_Raw], SignalArray] | None = None,
        options: AcquireOptions | None = None,
        **acquire_kwargs: object,
    ) -> SignalArray:
        """Run a caller-owned program without rebuilding it, returning its live slot.

        rounds is the positive integer software average count (bool is rejected).
        raw2signal_fn maps raw data to the slot shape; None uses the builder
        converter or default_raw2signal_fn. options controls retry/progress/stop;
        acquire_kwargs pass to acquire. Retry reuses this program. See the class
        contract for failures, partial results and progress cleanup.
        """
        return self._run_program(
            acquire=program.acquire,
            rounds=rounds,
            raw2signal_fn=raw2signal_fn,
            default_raw2signal_fn=default_raw2signal_fn,
            options=AcquireOptions() if options is None else options,
            acquire_kwargs=acquire_kwargs,
        )

    def run_program_decimated(
        self,
        program: ProgramProtocol[object, T_Raw],
        *,
        rounds: int = 1,
        raw2signal_fn: Callable[[T_Raw], SignalArray] | None = None,
        options: AcquireOptions | None = None,
        **acquire_kwargs: object,
    ) -> SignalArray:
        """Run caller-owned decimated acquisition under run_program's policy.

        raw2signal_fn maps decimated data to the slot shape; None uses the builder
        converter or default_decimated_raw2signal_fn. options and acquire_kwargs
        have the same meaning as in run_program; retry reuses this program.
        """
        return self._run_program(
            acquire=program.acquire_decimated,
            rounds=rounds,
            raw2signal_fn=raw2signal_fn,
            default_raw2signal_fn=default_decimated_raw2signal_fn,
            options=AcquireOptions() if options is None else options,
            acquire_kwargs=acquire_kwargs,
        )

    def _build_and_run(
        self,
        *,
        rounds: int,
        runner: Callable[..., SignalArray],
        raw2signal_fn: Callable[[Any], SignalArray] | None,
        options: AcquireOptions,
        acquire_kwargs: dict[str, object],
    ) -> SignalArray:
        _require_positive_rounds(rounds)
        if options.retry < 0:
            raise ValueError("retry must be non-negative")
        if self._schedule.check_stop_requested():
            return self._default_slot().view
        program = self._build_program()
        return runner(
            program,
            rounds=rounds,
            raw2signal_fn=raw2signal_fn,
            options=options,
            **acquire_kwargs,
        )

    def _run_program(
        self,
        *,
        acquire: Callable[..., T_Raw],
        rounds: int,
        raw2signal_fn: Callable[[T_Raw], SignalArray] | None,
        default_raw2signal_fn: Callable[[Any], SignalArray],
        options: AcquireOptions,
        acquire_kwargs: dict[str, object],
    ) -> SignalArray:
        _require_positive_rounds(rounds)
        retry = options.retry
        if retry < 0:
            raise ValueError("retry must be non-negative")
        self._schedule.ensure_active()

        slot = self._default_slot()
        signal_fn = self._resolve_raw2signal_fn(raw2signal_fn, default_raw2signal_fn)

        pbar = make_pbar(
            total=rounds,
            smoothing=0,
            desc=options.progress_label,
            leave=self._resolve_progress_leave(progress_leave=options.progress_leave),
            disable=rounds == 1,
        )
        try:
            for attempt in range(retry + 1):
                if self._schedule.check_stop_requested():
                    break
                pbar.reset()
                try:
                    raw = self._acquire_once(
                        acquire=acquire,
                        options=options,
                        acquire_kwargs=dict(acquire_kwargs, rounds=rounds),
                        signal_fn=signal_fn,
                        slot=slot,
                        pbar=pbar,
                    )
                except _AcquireStopped:
                    self._schedule.check_stop_requested()
                    break
                except _CallbackError as exc:
                    raise exc.error from None
                except KeyboardInterrupt as exc:
                    self._schedule.mark_interrupted(exc)
                    break
                # tolerated: Native acquisition failures retain partial data and may be retried.
                except Exception as exc:  # noqa: BLE001
                    # The producer reports first-round cancellation with an exception.
                    cause = None if isinstance(exc, StoppedPartialAcquireError) else exc
                    if self._schedule.check_stop_requested(exception=cause):
                        break
                    if attempt == retry:
                        self._schedule.mark_failed(exc)
                        break
                    continue

                if self._schedule.check_stop_requested():
                    break

                pbar.set_progress(rounds)
                slot.set(signal_fn(raw))
                break
        except KeyboardInterrupt as exc:
            self._schedule.mark_interrupted(exc)
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

    def _build_program(self) -> ProgramT:
        program_kwargs = dict(self._program_kwargs)
        program_kwargs["modules"] = deepcopy(self._modules)
        return self._program_cls(
            self._soccfg,
            sweep=deepcopy(self._sweeps) or None,
            **program_kwargs,
        )

    def _acquire_once(
        self,
        *,
        acquire: Callable[..., T_Raw],
        options: AcquireOptions,
        acquire_kwargs: dict[str, object],
        signal_fn: Callable[[T_Raw], SignalArray],
        slot: SignalSlot,
        pbar: BaseProgressBar,
    ) -> T_Raw:
        scope = self._schedule.scope.branch("acquire")
        token = scope.stop
        acquire_cancel_flag = AcquireCancelFlag(token)

        def update_hook(ir: int, raw: T_Raw, cancel_flag: CancelFlagProtocol) -> None:
            try:
                pbar.set_progress(ir)
                slot.set(signal_fn(raw))
                if options.stop_condition is not None and options.stop_condition():
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
