from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from qick.asm_v2 import AsmInst, ConfigReadout, Macro, Pulse, QickParam, TimedMacro

_W_FREQ = "w0"
_W_GAIN = "w3"
_W_LENGTH = "w4"


@dataclass
class PatchWmemFromRegs(Macro):
    """Persist selected runtime fields into one single-wave wmem entry."""

    name: str
    freq_reg: str | None = None
    gain_reg: str | None = None

    def __post_init__(self) -> None:
        if self.freq_reg is None and self.gain_reg is None:
            raise ValueError("PatchWmemFromRegs requires at least one runtime register")

    def expand(self, prog) -> list[AsmInst]:  # type: ignore[no-untyped-def]
        addr = _single_wave_addr(prog, self.name)
        insts = [_read_wmem(addr)]
        if self.freq_reg is not None:
            insts.append(_copy_reg_to_wave_field(prog, self.freq_reg, _W_FREQ))
        if self.gain_reg is not None:
            insts.append(_copy_reg_to_wave_field(prog, self.gain_reg, _W_GAIN))
        insts.append(AsmInst(inst={"CMD": "WMEM_WR", "DST": f"&{addr}"}, addr_inc=1))
        return insts


@dataclass
class PatchWmemFromDmem(Macro):
    """Load one table word and persist it into a single-wave wmem entry."""

    name: str
    idx_reg: str
    addr_reg: str
    val_reg: str
    dmem_offset: int

    def expand(self, prog) -> list[AsmInst]:  # type: ignore[no-untyped-def]
        idx_reg = prog._get_reg(self.idx_reg)
        addr_reg = prog._get_reg(self.addr_reg)
        val_reg = prog._get_reg(self.val_reg)
        insts = [
            AsmInst(
                inst={
                    "CMD": "REG_WR",
                    "DST": addr_reg,
                    "SRC": "op",
                    "OP": f"{idx_reg} + #{self.dmem_offset}",
                },
                addr_inc=1,
            ),
            AsmInst(
                inst={
                    "CMD": "REG_WR",
                    "DST": val_reg,
                    "SRC": "dmem",
                    "ADDR": f"&{addr_reg}",
                },
                addr_inc=1,
            ),
        ]
        insts.extend(
            PatchWmemFromRegs(name=self.name, freq_reg=self.val_reg).expand(prog)
        )
        return insts


class _WaveFromRegs:
    name: str
    freq_reg: str | None = None
    gain_reg: str | None = None

    def _expand_wave_from_regs(self, prog, port: int) -> list[AsmInst]:  # type: ignore[no-untyped-def]
        if self.freq_reg is None and self.gain_reg is None:
            raise ValueError("runtime wave playback requires at least one register")
        addr = _single_wave_addr(prog, self.name)
        time_reg = self.t_regs["t"]  # type: ignore[attr-defined]
        insts: list[AsmInst] = []
        if not isinstance(time_reg, Integral):
            insts.append(self.set_timereg(prog, "t"))  # type: ignore[attr-defined]
        insts.append(_read_wmem(addr))

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


class PulseFromLengthReg(TimedMacro):
    """Play one const/flat-top template with a runtime length register.

    Const pulses patch their only waveform. Flat-top pulses keep the two ramp
    entries unchanged and patch only the middle DDS (flat) entry. The emitted
    WPORT writes stay contiguous so the generator chains all flat-top segments
    as one pulse.
    """

    def __init__(
        self,
        *,
        ch: int,
        name: str,
        length_reg: str,
        flat_top: bool,
        t: float | QickParam = 0.0,
    ) -> None:
        super().__init__()
        self.ch = ch
        self.name = name
        self.length_reg = length_reg
        self.flat_top = flat_top
        self.t = t

    def preprocess(self, prog) -> None:  # type: ignore[no-untyped-def]
        if self.name not in prog.pulses:
            raise RuntimeError(
                f"trying to play pulse {self.name}, but it has not been defined"
            )
        pulse = prog.pulses[self.name]
        if pulse.gen_chs is None or self.ch not in pulse.gen_chs:
            raise RuntimeError(
                f"trying to play pulse {self.name} on generator {self.ch}, "
                f"but it was only defined for generators {pulse.gen_chs}"
            )
        # Runtime duration is accounted for by TableLengthPulse's duration
        # register. Do not advance QICK's compile-time generator timestamp here.
        self.convert_time(prog, self.t, "t")

    def expand(self, prog) -> list[Macro]:  # type: ignore[no-untyped-def]
        addrs = _pulse_wave_addrs(prog, self.name, flat_top=self.flat_top)
        port = int(prog.soccfg["gens"][self.ch]["tproc_ch"])
        time_reg = self.t_regs["t"]

        patch_addr = addrs[1] if self.flat_top else addrs[0]
        insts: list[Macro] = [
            _read_wmem(patch_addr),
            _copy_reg_to_wave_field(prog, self.length_reg, _W_LENGTH),
        ]
        if not isinstance(time_reg, Integral):
            insts.append(self.set_timereg(prog, "t"))

        for i, addr in enumerate(addrs):
            inst: dict[str, str] = {
                "CMD": "WPORT_WR",
                "DST": str(port),
            }
            if (self.flat_top and i == 1) or not self.flat_top:
                inst["SRC"] = "r_wave"
            else:
                inst["SRC"] = "wmem"
                inst["ADDR"] = f"&{addr}"
            if isinstance(time_reg, Integral):
                inst["TIME"] = f"@{time_reg}"
            insts.append(AsmInst(inst=inst, addr_inc=1))
        return insts


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


def _single_wave_addr(prog, name: str) -> int:  # type: ignore[no-untyped-def]
    if name not in prog.pulses:
        raise RuntimeError(f"pulse/readout config {name!r} is not registered")
    wave_names = prog.pulses[name].get_wavenames(exclude_special=True)
    if len(wave_names) != 1:
        raise NotImplementedError(
            f"runtime wmem patch for {name!r} requires exactly one "
            f"non-special waveform, got {len(wave_names)}"
        )
    return int(prog.wave2idx[wave_names[0]])


def _pulse_wave_addrs(prog, name: str, *, flat_top: bool) -> list[int]:  # type: ignore[no-untyped-def]
    if name not in prog.pulses:
        raise RuntimeError(f"pulse config {name!r} is not registered")
    wave_names = prog.pulses[name].get_wavenames(exclude_special=not flat_top)
    expected = (3, 4) if flat_top else (1,)
    if len(wave_names) not in expected:
        style = "flat_top" if flat_top else "const"
        raise NotImplementedError(
            f"runtime length pulse {name!r} ({style}) requires "
            f"{expected} waveform(s), got {len(wave_names)}"
        )
    return [int(prog.wave2idx[wave_name]) for wave_name in wave_names]


def _read_wmem(addr: int) -> AsmInst:
    return AsmInst(
        inst={
            "CMD": "REG_WR",
            "DST": "r_wave",
            "SRC": "wmem",
            "ADDR": f"&{addr}",
        },
        addr_inc=1,
    )


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
