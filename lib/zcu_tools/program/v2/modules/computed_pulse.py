from __future__ import annotations

from copy import deepcopy

from qick.asm_v2 import QickParam
from typing_extensions import TYPE_CHECKING, Optional, Sequence, Union

from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg

if TYPE_CHECKING:
    from zcu_tools.program.v2.modular import ModularProgramV2


class ComputedPulse(Module):
    """Select one primitive pulse by register-computed wmem address."""

    def __init__(self, name: str, *, val_reg: str, pulses: Sequence[PulseCfg]) -> None:
        self.name = name
        self.val_reg = val_reg

        raw_cfgs = list(pulses)
        if len(raw_cfgs) < 2:
            raise ValueError("ComputedPulse requires at least 2 candidate pulses")

        self.ref_cfg = raw_cfgs[0]

        if any([cfg.ch != self.ref_cfg.ch for cfg in raw_cfgs]):
            raise ValueError("All candidate pulses must have the same channel")

        if any(cfg.pre_delay != self.ref_cfg.pre_delay for cfg in raw_cfgs):
            raise ValueError("All candidate pulses must have the same pre_delay")

        self.pulse_modules = [
            Pulse(f"{self.name}_w{i}", cfg, f"{self.name}_w{i}")
            for i, cfg in enumerate(raw_cfgs)
        ]

    def init(self, prog: ModularProgramV2) -> None:
        for pulse_mod in self.pulse_modules:
            pulse_mod.init(prog)

        wave_idxs = [
            prog.wave2idx[prog.list_pulse_waveforms(pulse_mod.pulse_id)[0]]
            for pulse_mod in self.pulse_modules
        ]

        expected = list(range(min(wave_idxs), max(wave_idxs) + 1))
        if wave_idxs != expected:
            raise ValueError(
                "ComputedPulse candidate waveform indices must be contiguous, "
                f"got {wave_idxs}"
            )

        self.wmem_offset = min(wave_idxs)

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        ref_cfg = self.ref_cfg
        with prog.acquire_temp_reg(1) as (addr_reg,):
            # addr_reg = wmem_offset + gate_idx
            prog.write_reg_op(addr_reg, self.val_reg, "+", self.wmem_offset)
            prog.pulse_wmem_reg(ref_cfg.ch, addr_reg, t=t + ref_cfg.pre_delay)

        return t + self.total_length(prog)

    def total_length(self, prog: ModularProgramV2) -> float:
        lengths = [pulse.total_length(prog) for pulse in self.pulse_modules]
        if any([isinstance(length, QickParam) for length in lengths]):
            raise ValueError(
                "ComputedPulse total length cannot be determined at compile time"
            )
        return max([float(length) for length in lengths])

    def allow_rerun(self) -> bool:
        return True
