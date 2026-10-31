"""Program assembly and synchronous acquisition within a Schedule scope."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    Self,
    TypeAlias,
    Unpack,
)

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeVar

from zcu_tools.experiment.v2.runtime.acquisition import (
    AcquireOptions as AcquireOptions,
)
from zcu_tools.experiment.v2.runtime.acquisition import (
    AcquisitionRunner,
    require_positive_rounds,
)
from zcu_tools.experiment.v2.runtime.acquisition import (
    SignalArray as SignalArray,
)
from zcu_tools.experiment.v2.runtime.acquisition import (
    StopCondition as StopCondition,
)
from zcu_tools.experiment.v2.runtime.acquisition import (
    SynchronousAcquireKeywords as SynchronousAcquireKeywords,
)
from zcu_tools.experiment.v2.runtime.acquisition import (
    default_decimated_raw2signal_fn as default_decimated_raw2signal_fn,
)
from zcu_tools.experiment.v2.runtime.acquisition import (
    default_raw2signal_fn as default_raw2signal_fn,
)
from zcu_tools.program.acquisition import (
    ShotSumProcessor,
)
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
from zcu_tools.program.v2.modules.base import ModuleGroup, as_modules

if TYPE_CHECKING:
    from zcu_tools.experiment.v2.runtime.schedule import (
        Schedule,
        ScheduleStep,
        SignalSlot,
    )

T_Raw = TypeVar("T_Raw")
T_AcquireRaw_co = TypeVar("T_AcquireRaw_co", covariant=True)
T_DecimatedRaw_co = TypeVar("T_DecimatedRaw_co", covariant=True)


class ProgramProtocol(Protocol[T_AcquireRaw_co, T_DecimatedRaw_co]):
    """Synchronous producer accepted by the builder's run methods.

    rounds supplies the positive integer software average count. Both methods
    receive the caller's soc and native progress policy. Only the declared
    synchronous keywords are required by this static contract.
    They call round_hook with a one-based completed round, cumulative raw data
    and the supplied cancel_flag; they check that flag before starting another
    round. Return the final raw data, retaining completed rounds on cancellation.
    Raise StoppedPartialAcquireError if cancelled before any round completes.
    Use acquire_summed with read axes for joint classification and selection.
    This synchronous contract uses the default step_rounds=False mode;
    stepped preparation is not a
    completed raw result.
    Other producer errors propagate to the builder's partial-result policy.
    """

    def acquire(
        self,
        soc: object,
        **kwargs: Unpack[SynchronousAcquireKeywords[T_AcquireRaw_co]],
    ) -> T_AcquireRaw_co: ...

    def acquire_decimated(
        self,
        soc: object,
        **kwargs: Unpack[SynchronousAcquireKeywords[T_DecimatedRaw_co]],
    ) -> T_DecimatedRaw_co: ...


class SummedProgramProtocol(ProgramProtocol[object, object], Protocol):
    """Program with one joint-ADC processor returning per-point shot sums.

    processor receives ADC-keyed normalized float64 (points, reps, reads, 2)
    arrays and returns float64 (points, *axes). Completed rounds add without
    normalization. round_hook and final data have shape (*point_dims, *axes).
    rounds, step_rounds, native keywords and cancellation follow ProgramProtocol;
    step_rounds=True prepares only, returning None. threshold, angle and herald
    raise ValueError. See program.acquire_summed for shape/dtype validation.
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


ProgramT = TypeVar("ProgramT", bound=ProgramProtocol[Any, Any])
ProgramFactory: TypeAlias = Callable[..., ProgramT]


class ModuleFacade(ABC):
    """Fluent module assembly shared by the program builder's typed helpers."""

    @abstractmethod
    def add(self, *modules: ModuleGroup) -> Self:
        """Append modules or ordered groups; skip None (no module), return self."""
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
    acquire-local. External cancellation returns the slot's partial live array
    (unwritten slots stay NaN); cancellation of the Schedule itself also records
    its stopped outcome. KeyboardInterrupt propagates unchanged. on_stop observes
    external cancellation after scope/progress cleanup as defined by AcquireOptions.
    Producer
    failures propagate unchanged after the explicit retry budget is exhausted;
    concurrent cancellation prevents retry but does not suppress a failure.
    Progress bars close before return or propagation.
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

    def add(self, *modules: ModuleGroup) -> Self:
        """Append modules/groups, skipping None (no module); copy only at build."""
        for module_or_group in modules:
            self._modules.extend(as_modules(module_or_group))
        return self

    def declare_sweep(self, name: str, spec: SweepCfg | int) -> Self:
        """Append a named program-side sweep or loop count, copied at build time."""
        self._sweeps.append((name, spec))
        return self

    def set_raw2signal_fn(self, fn: Callable[[T_Raw], SignalArray]) -> Self:
        """Set the fallback IQ converter for integrated and decimated modes.

        Processed acquisition always requires its own explicit converter.
        """
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
        """Build once and run integrated acquisition, returning the live slot.

        rounds is the software average count, default one, independent of cfg.
        It must be a positive integer, excluding bool; invalid values raise
        ValueError before factory invocation or acquisition. The same value is
        passed to the producer and used as the rounds progress total. Retry
        reuses both this value and the built program.

        raw2signal_fn overrides the builder converter and must return a value
        compatible with the owner's slot. None uses the builder converter or
        default_raw2signal_fn. options controls retry, progress and stop;
        acquire_kwargs pass to the producer's acquire method.
        See the class contract for scope, modules, failures and partial results.
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
        """Build once and run decimated acquisition, returning the live slot.

        rounds is the software average count, default one, independent of cfg.
        It must be a positive integer, excluding bool; invalid values raise
        ValueError before factory invocation or acquisition. The same value is
        passed to the producer and used as the rounds progress total. Retry
        reuses both this value and the built program.

        raw2signal_fn maps decimated raw data to the owner's slot shape; None
        uses the builder converter or default_decimated_raw2signal_fn. options
        controls retry, progress and stop; acquire_kwargs pass to the producer's
        acquire_decimated method. See the class contract for scope, modules,
        failures and partial results.
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

    def run_program_summed(
        self,
        program: SummedProgramProtocol,
        *,
        processor: ShotSumProcessor,
        sums2signal_fn: Callable[[NDArray[np.float64]], SignalArray],
        rounds: int = 1,
        options: AcquireOptions | None = None,
        **acquire_kwargs: object,
    ) -> SignalArray:
        """Run a compiled program with one processor and an explicit sum converter.

        Build program first, then resolve any read labels in processor.
        sums2signal_fn converts cumulative and final (*point_dims, *axes) sums
        to the default signal slot's array shape. No IQ or builder-wide converter
        is selected. rounds is a positive software round count. options controls
        retry, progress and data stop exactly as run_program; native
        acquire_kwargs pass through unchanged. Retries reuse program and reset
        acquisition sums. None results raise ValueError during conversion because
        this runner requires synchronous acquisition. Cancellation retains
        completed-round slot data; a stop before any data retains initial NaNs.
        """

        def convert(raw: NDArray[np.float64] | None) -> SignalArray:
            if raw is None:
                raise ValueError("summed builder requires synchronous acquisition")
            return sums2signal_fn(raw)

        return self._run_program(
            acquire=program.acquire_summed,
            rounds=rounds,
            raw2signal_fn=convert,
            default_raw2signal_fn=convert,
            options=AcquireOptions() if options is None else options,
            acquire_kwargs=dict(acquire_kwargs, processor=processor),
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
        require_positive_rounds(rounds)
        if options.retry < 0:
            raise ValueError("retry must be non-negative")
        if self._schedule.check_stop_requested():
            if options.on_stop is not None:
                options.on_stop()
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
        return AcquisitionRunner(self._owner, self._soc).run(
            acquire=acquire,
            rounds=rounds,
            signal_fn=self._resolve_raw2signal_fn(raw2signal_fn, default_raw2signal_fn),
            options=options,
            acquire_kwargs=acquire_kwargs,
        )

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

    def _default_slot(self) -> SignalSlot:
        return self._schedule.default_slot_for(self._owner)
