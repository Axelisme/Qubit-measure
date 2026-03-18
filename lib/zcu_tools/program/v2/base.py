from __future__ import annotations

from qick import QickConfig
from qick.asm_v2 import AveragerProgramV2
from typing_extensions import Any, Mapping, TypedDict, cast

from zcu_tools.program.base import MyProgram

from .modules.registry import PulseRegistry


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

    def acquire(self, *args, **kwargs):
        # v2 program need to pass rounds to acquire
        return super().acquire(*args, rounds=self.cfg["rounds"], **kwargs)

    def acquire_decimated(self, *args, **kwargs):
        # v2 program need to pass rounds to acquire_decimated
        return super().acquire_decimated(*args, rounds=self.cfg["rounds"], **kwargs)
