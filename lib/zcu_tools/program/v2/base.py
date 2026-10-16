from __future__ import annotations

import logging
from typing import Annotated

import numpy as np
from qick import QickConfig
from qick.asm_v2 import AveragerProgramV2

from zcu_tools.cfg_model import ConfigBase, FieldMeta

from .ir import IRCompileMixin
from .macro import ImproveAsmV2
from .modules.registry import PulseRegistry
from .simulation import SimulationAcquireMixin

logger = logging.getLogger(__name__)


class ProgramV2Cfg(ConfigBase):
    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay",
            unit="us",
            description="Delay before the first repetition.",
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0


# QICK annotates single ndarrays; our existing mixin returns one array per ADC.
class MyProgramV2(  # pyright: ignore[reportIncompatibleMethodOverride]
    SimulationAcquireMixin, ImproveAsmV2, AveragerProgramV2, IRCompileMixin
):
    def __init__(self, soccfg: QickConfig, cfg: ProgramV2Cfg, **kwargs) -> None:

        # v2 program need to pass reps and final_delay to init
        self.cfg_model = cfg
        self.pulse_registry = PulseRegistry()

        super().__init__(
            soccfg,
            cfg=None,
            reps=cfg.reps,
            initial_delay=cfg.initial_delay,
            final_wait=0.0,
            final_delay=cfg.relax_delay,
            **kwargs,
        )

        logger.debug("ASM:\n%s", self.asm())

    def _initialize(self, cfg: ProgramV2Cfg) -> None: ...

    def _body(self, cfg: ProgramV2Cfg) -> None: ...

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

    def acquire(self, soc, *args, **kwargs):
        return super().acquire(soc, *args, rounds=self.cfg_model.rounds, **kwargs)

    def acquire_decimated(self, soc, *args, **kwargs):
        return super().acquire_decimated(
            soc, *args, rounds=self.cfg_model.rounds, **kwargs
        )
