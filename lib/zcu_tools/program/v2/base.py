from __future__ import annotations

import logging
from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray
from qick import QickConfig
from qick.asm_v2 import AveragerProgramV2

from .ir import IRCompileMixin
from .macro import ImproveAsmV2
from .modules.registry import PulseRegistry
from .simulation import SimulationAcquireMixin

logger = logging.getLogger(__name__)


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
        self.pulse_registry = PulseRegistry()

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

    def _initialize(self, cfg: None) -> None: ...

    def _body(self, cfg: None) -> None: ...

    def _make_binprog(self) -> None:
        """Build QICK binaries with a NumPy-version-independent dmem payload."""
        super()._make_binprog()

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
        self, soc: object, *args, rounds: int = 1, **kwargs
    ) -> list[NDArray[np.float64]] | None: ...

    def acquire(
        self, soc, *args, rounds: int = 1, **kwargs
    ) -> list[NDArray[np.float64]] | None:
        """Acquire integrated IQ from soc over software-averaged rounds.

        rounds defaults to one and must be a positive integer, excluding bool.
        Invalid rounds raises ValueError before transport or simulation admission.
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
        self, soc: object, *args, rounds: int = 1, **kwargs
    ) -> list[NDArray[np.float64]] | None: ...

    def acquire_decimated(
        self, soc, *args, rounds: int = 1, **kwargs
    ) -> list[NDArray[np.float64]] | None:
        """Acquire decimated IQ traces from soc, averaged over software rounds.

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
