from __future__ import annotations

from numbers import Integral

from qick.asm_v2 import AsmInst, TimedMacro


class PulseFromWmemReg(TimedMacro):
    """Play one or more pre-addressed wmem entries back-to-back at the same TIME.

    Emits exactly ``len(addr_regs)`` ``WPORT_WR`` instructions with no other
    instructions interleaved, so the hardware can chain a multi-wave pulse
    (e.g. flat_top: ramp_up / flat / ramp_down) without a gap. All address
    arithmetic must be done by the caller before this macro is appended.
    """

    # fields: ch (int), addr_regs (list[str]), t (float | QickParam)
    def preprocess(self, prog):
        self.convert_time(prog, self.t, "t")

    def expand(self, prog):  # type: ignore
        tproc_ch = prog.soccfg["gens"][self.ch]["tproc_ch"]
        t_reg = self.t_regs["t"]

        insts = []
        imm_time = isinstance(t_reg, Integral)
        if not imm_time:
            insts.append(self.set_timereg(prog, "t"))

        for reg_name in self.addr_regs:
            inst = {
                "CMD": "WPORT_WR",
                "DST": str(tproc_ch),
                "SRC": "wmem",
                "ADDR": f"&{prog._get_reg(reg_name)}",
            }
            if imm_time:
                inst["TIME"] = "@" + str(t_reg)
            insts.append(AsmInst(inst=inst, addr_inc=1))
        return insts
