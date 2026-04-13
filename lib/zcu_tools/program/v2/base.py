from __future__ import annotations

import logging

from qick import QickConfig
from qick.asm_v2 import AveragerProgramV2
from typing_extensions import Any, Mapping, TypedDict, cast

from zcu_tools.program.base import MyProgram

from .modules.registry import PulseRegistry

logger = logging.getLogger(__name__)


class ProgramV2Cfg(TypedDict):
    reps: int
    rounds: int
    relax_delay: float


class MyProgramV2(MyProgram, AveragerProgramV2):  # type: ignore
    def __init__(self, soccfg: QickConfig, cfg: Mapping[str, Any], **kwargs) -> None:
        _cfg = cast(ProgramV2Cfg, cfg)

        # v2 program need to pass reps and final_delay to init
        self.pulse_registry = PulseRegistry()
        super().__init__(
            soccfg,
            cfg=dict(_cfg),
            reps=_cfg["reps"],
            initial_delay=0.0,
            final_wait=_cfg.get("final_wait", 0.0),
            final_delay=_cfg["relax_delay"],
            **kwargs,
        )

    def compile(self) -> None:
        super().compile()

        pmem_len = len(self.binprog["pmem"]) if self.binprog["pmem"] else 0
        wmem_len = len(self.binprog["wmem"]) if self.binprog["wmem"] else 0
        dmem_len = len(self.binprog["dmem"]) if self.binprog["dmem"] else 0
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
            self.cfg["reps"],
            self.cfg["rounds"],
        )
        return super().acquire(*args, rounds=self.cfg["rounds"], **kwargs)

    def acquire_decimated(self, *args, **kwargs):
        return super().acquire_decimated(*args, rounds=self.cfg["rounds"], **kwargs)
