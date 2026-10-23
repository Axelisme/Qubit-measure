from __future__ import annotations

import logging
from numbers import Integral

from qick.asm_v2 import AsmInst, TimedMacro

logger = logging.getLogger(__name__)


class DelayRegAuto(TimedMacro):
    """Auto-align to timeline, then increment by runtime cycles from a register.

    time_reg names a register holding tProc cycles (not microseconds). gens and
    ros select generator and readout timelines. preprocess raises ValueError before
    resolving the register if neither channel type is selected.
    """

    # fields: time_reg (str), gens (bool), ros (bool)
    def preprocess(self, prog) -> None:
        if not self.gens and not self.ros:
            raise ValueError("DelayRegAuto requires at least one channel: gens or ros")

        # Resolve early to fail fast on missing/invalid register names.
        prog._get_reg(self.time_reg)
        auto_t = prog.get_max_timestamp(gens=self.gens, ros=self.ros)
        auto_rounded = self.convert_time(prog, auto_t, "auto_t")
        prog.decrement_timestamps(auto_rounded)

    # tolerated: QICK Macro.expand is inferred as None, but translate iterates the returned macro list.
    def expand(self, prog):  # pyright: ignore[reportIncompatibleMethodOverride]
        insts = []
        auto_t_reg = self.t_regs["auto_t"]
        if isinstance(auto_t_reg, Integral):
            insts.append(
                AsmInst(
                    inst={
                        "CMD": "TIME",
                        "C_OP": "inc_ref",
                        "LIT": f"#{int(auto_t_reg)}",
                    },
                    addr_inc=1,
                )
            )
        elif auto_t_reg is not None:
            insts.append(
                AsmInst(
                    inst={
                        "CMD": "TIME",
                        "C_OP": "inc_ref",
                        "R1": prog._get_reg(auto_t_reg),
                    },
                    addr_inc=1,
                )
            )
        insts.append(
            AsmInst(
                inst={
                    "CMD": "TIME",
                    "C_OP": "inc_ref",
                    "R1": prog._get_reg(self.time_reg),
                },
                addr_inc=1,
            )
        )
        return insts
