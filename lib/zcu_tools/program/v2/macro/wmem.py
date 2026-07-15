from __future__ import annotations

from numbers import Integral

from qick.asm_v2 import AsmInst, ConfigReadout, Pulse, QickParam

_W_FREQ = "w0"
_W_GAIN = "w3"


class _WaveFromRegs:
    name: str
    freq_reg: str | None = None
    gain_reg: str | None = None

    def _expand_wave_from_regs(self, prog, port: int) -> list[AsmInst]:  # type: ignore[no-untyped-def]
        if self.freq_reg is None and self.gain_reg is None:
            raise ValueError("runtime wave playback requires at least one register")
        if self.name not in prog.pulses:
            raise RuntimeError(f"pulse/readout config {self.name!r} is not registered")

        wave_names = prog.pulses[self.name].get_wavenames(exclude_special=True)
        if len(wave_names) != 1:
            raise NotImplementedError(
                f"runtime wmem patch for {self.name!r} requires exactly one "
                f"non-special waveform, got {len(wave_names)}"
            )

        addr = prog.wave2idx[wave_names[0]]
        time_reg = self.t_regs["t"]  # type: ignore[attr-defined]
        insts: list[AsmInst] = []
        if not isinstance(time_reg, Integral):
            insts.append(self.set_timereg(prog, "t"))  # type: ignore[attr-defined]
        insts.append(
            AsmInst(
                inst={
                    "CMD": "REG_WR",
                    "DST": "r_wave",
                    "SRC": "wmem",
                    "ADDR": f"&{addr}",
                },
                addr_inc=1,
            )
        )

        if self.freq_reg is not None:
            insts.append(_copy_reg_to_wave_field(prog, self.freq_reg, _W_FREQ))
        if self.gain_reg is not None:
            insts.append(_copy_reg_to_wave_field(prog, self.gain_reg, _W_GAIN))

        port_inst = {"CMD": "WPORT_WR", "DST": str(port), "SRC": "r_wave"}
        if isinstance(time_reg, Integral):
            port_inst["TIME"] = f"@{time_reg}"
        insts.append(AsmInst(inst=port_inst, addr_inc=1))
        return insts


class PulseFromRegs(_WaveFromRegs, Pulse):
    """Play a single-wave generator pulse directly from patched wave registers."""

    def __init__(
        self,
        *,
        ch: int,
        name: str,
        t: float | QickParam = 0.0,
        tag: str | None = None,
        freq_reg: str | None = None,
        gain_reg: str | None = None,
    ) -> None:
        super().__init__()
        self.ch = ch
        self.name = name
        self.t = t
        self.tag = tag
        self.freq_reg = freq_reg
        self.gain_reg = gain_reg

    def expand(self, prog) -> list[AsmInst]:  # type: ignore[no-untyped-def]
        port = int(prog.soccfg["gens"][self.ch]["tproc_ch"])
        return self._expand_wave_from_regs(prog, port)


class ConfigReadoutFromRegs(_WaveFromRegs, ConfigReadout):
    """Send a readout config directly from patched wave registers."""

    def __init__(
        self,
        *,
        ch: int,
        name: str,
        t: float | QickParam = 0.0,
        tag: str | None = None,
        freq_reg: str | None = None,
    ) -> None:
        super().__init__()
        self.ch = ch
        self.name = name
        self.t = t
        self.tag = tag
        self.freq_reg = freq_reg
        self.gain_reg = None

    def expand(self, prog) -> list[AsmInst]:  # type: ignore[no-untyped-def]
        port = int(prog.soccfg["readouts"][self.ch]["tproc_ctrl"])
        return self._expand_wave_from_regs(prog, port)


def _copy_reg_to_wave_field(prog, reg: str, wave_reg: str) -> AsmInst:  # type: ignore[no-untyped-def]
    resolved = prog._get_reg(reg)
    return AsmInst(
        inst={
            "CMD": "REG_WR",
            "DST": wave_reg,
            "SRC": "op",
            "OP": resolved,
        },
        addr_inc=1,
    )
