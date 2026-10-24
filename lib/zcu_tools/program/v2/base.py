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
        """Acquire averaged IQ over positive integer rounds (default one).

        bool is not a round count. Invalid rounds raises ValueError before
        transport/admission. Other options follow QICK; pass them by keyword.
        step_rounds=True prepares hardware and returns None; simulation rejects
        stepped ownership. Cancellation, hooks and partial results are unchanged.
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
        """Acquire decimated IQ over positive integer rounds (default one).

        Validation, options, stepped ownership and failures follow acquire().
        Software rounds average traces; they do not concatenate repetitions.
        """
        _require_positive_integer(rounds, "rounds")
        return super().acquire_decimated(soc, *args, rounds=rounds, **kwargs)
