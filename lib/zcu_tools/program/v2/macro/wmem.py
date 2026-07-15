from __future__ import annotations

from dataclasses import dataclass

from qick.asm_v2 import AsmInst, Macro

_W_FREQ = "w0"
_W_GAIN = "w3"


@dataclass
class PatchWmemFromRegs(Macro):
    """Patch a single-wave pulse/readout wmem entry from runtime registers."""

    name: str
    freq_reg: str | None = None
    gain_reg: str | None = None

    def __post_init__(self) -> None:
        if self.freq_reg is None and self.gain_reg is None:
            raise ValueError("PatchWmemFromRegs requires at least one runtime register")

    def expand(self, prog) -> list[AsmInst]:  # type: ignore[no-untyped-def]
        if self.name not in prog.pulses:
            raise RuntimeError(f"pulse/readout config {self.name!r} is not registered")

        wave_names = prog.pulses[self.name].get_wavenames(exclude_special=True)
        if len(wave_names) != 1:
            raise NotImplementedError(
                f"runtime wmem patch for {self.name!r} requires exactly one "
                f"non-special waveform, got {len(wave_names)}"
            )

        addr = prog.wave2idx[wave_names[0]]
        insts = [
            AsmInst(
                inst={
                    "CMD": "REG_WR",
                    "DST": "r_wave",
                    "SRC": "wmem",
                    "ADDR": f"&{addr}",
                },
                addr_inc=1,
            )
        ]

        if self.freq_reg is not None:
            insts.append(_copy_reg_to_wave_field(prog, self.freq_reg, _W_FREQ))
        if self.gain_reg is not None:
            insts.append(_copy_reg_to_wave_field(prog, self.gain_reg, _W_GAIN))

        insts.append(AsmInst(inst={"CMD": "WMEM_WR", "DST": f"&{addr}"}, addr_inc=1))
        return insts


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
