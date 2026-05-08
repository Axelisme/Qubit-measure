from __future__ import annotations

import logging

from qick import QickConfig
from qick.asm_v2 import AveragerProgramV2

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.program.base import ImproveAcquireMixin

from .macro import ImproveAsmV2
from .modules.registry import PulseRegistry

logger = logging.getLogger(__name__)


class ProgramV2Cfg(ConfigBase):
    reps: int = 1
    rounds: int = 1
    initial_delay: float = 1.0
    relax_delay: float = 1.0


class MyProgramV2(ImproveAcquireMixin, ImproveAsmV2, AveragerProgramV2):  # type: ignore
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

    def acquire(self, *args, **kwargs):
        logger.debug(
            "MyProgramV2.acquire: reps=%s, rounds=%s",
            self.cfg_model.reps,
            self.cfg_model.rounds,
        )
        return super().acquire(*args, rounds=self.cfg_model.rounds, **kwargs)

    def acquire_decimated(self, *args, **kwargs):
        logger.debug(
            "MyProgramV2.acquire_decimated: reps=%s, rounds=%s",
            self.cfg_model.reps,
            self.cfg_model.rounds,
        )
        return super().acquire_decimated(*args, rounds=self.cfg_model.rounds, **kwargs)
