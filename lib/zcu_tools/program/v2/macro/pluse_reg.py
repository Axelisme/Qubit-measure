from __future__ import annotations

from numbers import Integral

from qick.asm_v2 import AsmInst, TimedMacro


class PulseFromWmemReg(TimedMacro):
    """Play one waveform using a runtime-computed wmem address register."""

    # fields: ch (int), addr_reg (str), t (float | QickParam), tag (str | None)
    def preprocess(self, prog):  # type: ignore[override]
        self.convert_time(prog, self.t, "t")

    def expand(self, prog):  # type: ignore[override]
        tproc_ch = prog.soccfg["gens"][self.ch]["tproc_ch"]
        t_reg = self.t_regs["t"]
        addr = f"&{prog._get_reg(self.addr_reg)}"

        insts = []
        imm_time = isinstance(t_reg, Integral)
        if not imm_time:
            insts.append(self.set_timereg(prog, "t"))

        inst = {"CMD": "WPORT_WR", "DST": str(tproc_ch), "SRC": "wmem", "ADDR": addr}
        if imm_time:
            inst["TIME"] = "@" + str(t_reg)
        insts.append(AsmInst(inst=inst, addr_inc=1))
        return insts

