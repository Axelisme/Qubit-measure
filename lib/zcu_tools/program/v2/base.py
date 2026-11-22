from __future__ import annotations

import logging
from numbers import Real
from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray
from qick import QickConfig
from qick.asm_v2 import AveragerProgramV2, QickParam, QickRawParam

from .ir import IRCompileMixin
from .macro import ImproveAsmV2
from .modules.registry import PulseRegistry
from .simulation import SimulationAcquireMixin
from .utils import is_qick_param

logger = logging.getLogger(__name__)


class TProcMemoryError(RuntimeError):
    """A compiled binary exceeds one tProc memory's capacity.

    ``memory`` names the memory (pmem, wmem or dmem); ``used`` and
    ``capacity`` are the compiled size and board limit in words.
    """

    def __init__(
        self, memory: Literal["pmem", "wmem", "dmem"], used: int, capacity: int
    ) -> None:
        self.memory = memory
        self.used = used
        self.capacity = capacity
        super().__init__(
            f"compiled program uses {used} words of {memory}, "
            f"but the size of that tProc memory is only {capacity}"
        )


def _require_positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


# QICK annotates single ndarrays; our existing mixin returns one array per ADC.
# tolerated: QICK annotates acquisition buffers as ndarray although its implementations return per-channel lists.
class MyProgramV2(  # pyright: ignore[reportIncompatibleMethodOverride]
    SimulationAcquireMixin, ImproveAsmV2, AveragerProgramV2, IRCompileMixin
):
    def __init__(
        self,
        soccfg: QickConfig,
        *,
        reps: int = 1,
        initial_delay: float = 1.0,
        relax_delay: float = 1.0,
        **kwargs,
    ) -> None:
        """Compile with positive integer reps and initial/relaxation delays in us.

        bool is not a repetition count. Invalid reps raises ValueError before
        compilation. Additional keywords are QICK compilation options.
        """
        _require_positive_integer(reps, "reps")
        self.relax_delay = relax_delay

        super().__init__(
            soccfg,
            cfg=None,
            reps=reps,
            initial_delay=initial_delay,
            final_wait=0.0,
            final_delay=relax_delay,
            **kwargs,
        )

        logger.debug("ASM:\n%s", self.asm())

    def native_readout_iq_offset(self, channel: int) -> float:
        """Resolve QICK's DC subtraction for one compiled integrated ADC.

        channel is the physical ADC channel, not its declaration position.
        For a static PFB, call after QICK configures the ADC for the current
        acquire, or raise RuntimeError. Unknown channels raise KeyError.
        This does not acquire or modify data.
        """
        readout = self.ro_chs[channel]
        registers = readout.get("ro_config")
        if "pfb_nout" in self.soccfg["readouts"][channel] and (
            registers is None or "pfb_ch" not in registers
        ):
            raise RuntimeError(f"ADC {channel} requires configured PFB registers")
        return float(self._ro_offset(channel, registers))

    def add_pulse(self, ch: int | list[int], name: str, **kwargs: object) -> None:
        """Declare a QICK pulse after checking its normalized gain.

        ch selects one generator or a list sharing a pulse definition; name is
        the program-local pulse identifier. kwargs are native QICK pulse style
        parameters, including optional gain in DAC full-scale fractions.
        Scalar gains and the minval/maxval of QickParam (including TableParam)
        must lie in [-1, 1]. Otherwise ValueError names the pulse, generator and
        actual/allowed range before QICK conversion. Parameters
        without gain (such as mux masks) and other QICK errors pass through.
        """
        gain = kwargs.get("gain")
        if is_qick_param(gain):
            minimum, maximum = gain.minval(), gain.maxval()
        elif isinstance(gain, Real):
            minimum = maximum = float(gain)
        else:
            super().add_pulse(ch, name, **kwargs)
            return
        if minimum < -1 or maximum > 1:
            raise ValueError(
                f"pulse {name!r} on generator {ch} gain range "
                f"[{minimum}, {maximum}] is outside allowed range [-1, 1]"
            )
        super().add_pulse(ch, name, **kwargs)

    def freq2reg(
        self, f: float | QickParam, gen_ch: int = 0, ro_ch: int | None = None
    ) -> int | QickRawParam:
        """Convert DDS MHz to generator words using QICK's channel matching.

        gen_ch selects the generator; optional ro_ch matches its frequency grid
        to that ADC. Sweeps return QickRawParam, scalars return int. A swept
        QICK range failure raises RuntimeError with sweep and DDS limits in MHz,
        retaining its cause. Other failures propagate unchanged.
        """
        try:
            return self.soccfg.freq2reg(f, gen_ch=gen_ch, ro_ch=ro_ch)
        except RuntimeError as error:
            if not isinstance(f, QickParam) or not f.is_sweep():
                raise
            gencfg = self.soccfg["gens"][gen_ch]
            half_range = gencfg["f_dds"] / 2
            # Keep QICK's comparisons: this translates failure, not admission.
            if gencfg["interpolation"] == 1 or not (f > half_range or f < -half_range):
                raise
            raise RuntimeError(
                f"generator {gen_ch} frequency sweep range "
                f"[{f.minval():g}, {f.maxval():g}] MHz is outside DDS range "
                f"[{-half_range:g}, {half_range:g}] MHz"
            ) from error

    def _init_declarations(self) -> None:
        super()._init_declarations()
        # QICK rebuilds pulse declarations; deduplication belongs to that build.
        self.pulse_registry = PulseRegistry()
        self.clear_shot_mask_sources()

    def _initialize(self, cfg: None) -> None: ...

    def _body(self, cfg: None) -> None: ...

    def _make_binprog(self) -> None:
        """Build QICK binaries, check capacities, and normalize dmem words."""
        # Use QICK's encoders, but own the capacity check rather than parsing its error.
        self.binprog = {}
        self.binprog["pmem"] = self._compile_prog()
        self.binprog["wmem"] = self._compile_waves()
        self.binprog["dmem"] = self.compile_datamem()
        for memory in ("pmem", "wmem", "dmem"):
            binary = self.binprog[memory]
            used = 0 if binary is None else len(binary)
            capacity = self.tproccfg[f"{memory}_size"]
            if used > capacity:
                raise TProcMemoryError(memory, used, capacity)

        dmem = self.binprog["dmem"]
        if dmem is None:
            return

        words = np.asarray(dmem, dtype=np.int32)
        if words.ndim != 1:
            raise ValueError(
                f"compiled dmem must be one-dimensional, got shape {words.shape}"
            )
        self.binprog["dmem"] = [int(word) for word in words]

    def compile(self) -> None:
        super().compile()

        pmem_len = len(self.binprog["pmem"]) if self.binprog["pmem"] is not None else 0
        wmem_len = len(self.binprog["wmem"]) if self.binprog["wmem"] is not None else 0
        dmem_len = len(self.binprog["dmem"]) if self.binprog["dmem"] is not None else 0
        pmem_cap = self.tproccfg["pmem_size"]
        wmem_cap = self.tproccfg["wmem_size"]
        dmem_cap = self.tproccfg["dmem_size"]

        logger.debug(
            f"{self.__class__.__name__}.compile: "
            f"pmem={pmem_len}/{pmem_cap} ({(100 * pmem_len / pmem_cap if pmem_cap else 0):.1f}%), "
            f"wmem={wmem_len}/{wmem_cap} ({(100 * wmem_len / wmem_cap if wmem_cap else 0):.1f}%), "
            f"dmem={dmem_len}/{dmem_cap} ({(100 * dmem_len / dmem_cap if dmem_cap else 0):.1f}%)",
        )

    @overload
    def acquire(
        self,
        soc: object,
        *,
        rounds: int = 1,
        step_rounds: Literal[False] = False,
        **kwargs: object,
    ) -> list[NDArray[np.float64]]: ...

    @overload
    def acquire(
        self,
        soc: object,
        *,
        rounds: int = 1,
        step_rounds: Literal[True],
        **kwargs: object,
    ) -> None: ...

    @overload
    def acquire(
        self,
        soc: object,
        *args,
        rounds: int = 1,
        **kwargs,
    ) -> list[NDArray[np.float64]] | None: ...

    def acquire(
        self,
        soc,
        *args,
        rounds: int = 1,
        **kwargs,
    ) -> list[NDArray[np.float64]] | None:
        """Acquire integrated IQ from soc over software-averaged rounds.

        rounds defaults to one and must be a positive integer, excluding bool.
        Invalid rounds raises ValueError before transport or simulation admission.
        Native QICK threshold/angle returns excited fractions in I and zero Q.
        With shot masks, IQ is Σ m·x / N, where N counts every shot in completed
        rounds. A non-None threshold is incompatible with shot masks.
        charge_requirements is a tuple of simulate.boundary.ChargeDriveRequirement
        records. The simulation entry forwards it for environment validation;
        hardware consumes the keyword without checking physical roles or passing
        it to QICK. Operators remain responsible for real wiring.
        Other options follow QICK and the acquisition mixins; pass them by keyword.

        Omitted or False step_rounds runs to completion and returns one float64
        array per ADC in a list. Arrays contain averaged IQ with a final I/Q
        axis of length two, retaining readout and sweep axes. True prepares the
        first hardware round and returns None. The caller then owns stepping
        through finish_round/prepare_round and collecting finish_acquire.
        SimulationSoc rejects step_rounds=True with ValueError before admission.
        With only soc positional, overloads distinguish False from True;
        extra positional options or a dynamic bool retain the list-or-None type.

        cancel_flag stops further rounds while retaining completed-round data.
        Cancellation before any completed round raises StoppedPartialAcquireError.
        round_hook receives each completed round and may request cancellation.
        QICK argument/transport errors and hook errors propagate.
        """
        _require_positive_integer(rounds, "rounds")
        return super().acquire(soc, *args, rounds=rounds, **kwargs)

    @overload
    def acquire_decimated(
        self,
        soc: object,
        *,
        rounds: int = 1,
        step_rounds: Literal[False] = False,
        **kwargs: object,
    ) -> list[NDArray[np.float64]]: ...

    @overload
    def acquire_decimated(
        self,
        soc: object,
        *,
        rounds: int = 1,
        step_rounds: Literal[True],
        **kwargs: object,
    ) -> None: ...

    @overload
    def acquire_decimated(
        self,
        soc: object,
        *args,
        rounds: int = 1,
        **kwargs,
    ) -> list[NDArray[np.float64]] | None: ...

    def acquire_decimated(
        self,
        soc,
        *args,
        rounds: int = 1,
        **kwargs,
    ) -> list[NDArray[np.float64]] | None:
        """Acquire decimated IQ traces from soc, averaged over software rounds.

        Nonempty herald raises ValueError: post-selection requires integrated shots.
        rounds validation, keyword options, step_rounds ownership, cancellation,
        hooks and errors follow acquire(). Omitted or False step_rounds returns
        a list of float64 trace arrays, one per ADC, with a final length-two I/Q
        axis and a time-sample axis. Hardware step_rounds=True prepares only and
        returns None; SimulationSoc rejects it before admission.
        Software rounds average matching traces rather than concatenate them;
        repetitions and readouts retain QICK's trace layout.
        """
        _require_positive_integer(rounds, "rounds")
        return super().acquire_decimated(soc, *args, rounds=rounds, **kwargs)
