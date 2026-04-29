from __future__ import annotations

from qick.asm_v2 import QickParam
from typing_extensions import TYPE_CHECKING, Sequence, Union

from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg

if TYPE_CHECKING:
    from zcu_tools.program.v2.ir.builder import IRBuilder
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

        # flat_top occupies 3 contiguous wmem entries (ramp_up, flat, ramp_down);
        # all other styles use 1. Mixing flat_top with non-flat_top breaks the
        # uniform-stride assumption used to address candidates by gate_idx.
        flat_tops = [cfg.waveform.style == "flat_top" for cfg in raw_cfgs]
        if any(flat_tops) and not all(flat_tops):
            raise ValueError(
                "ComputedPulse cannot mix flat_top with other waveform styles "
                "(wmem stride would differ across candidates)"
            )
        self._is_flat_top = flat_tops[0]
        self._stride = 3 if self._is_flat_top else 1

        self.pulse_modules = [
            Pulse(f"{self.name}_w{i}", cfg, f"{self.name}_w{i}")
            for i, cfg in enumerate(raw_cfgs)
        ]

    def init(self, prog: ModularProgramV2) -> None:
        if self._is_flat_top:
            gen_type = prog.soccfg["gens"][self.ref_cfg.ch].get("type", "")
            if gen_type in ("axis_sg_int4_v1", "axis_sg_int4_v2"):
                raise ValueError(
                    "ComputedPulse does not support flat_top on int4 generators "
                    f"(channel {self.ref_cfg.ch}, type {gen_type!r}); QICK appends "
                    "an extra dummy wave that breaks the wmem stride assumption"
                )

        for pulse_mod in self.pulse_modules:
            pulse_mod.init(prog)

        flat_idxs: list[int] = []
        for pulse_mod in self.pulse_modules:
            wave_names = prog.list_pulse_waveforms(
                pulse_mod.pulse_id, exclude_special=False
            )
            if len(wave_names) != self._stride:
                raise ValueError(
                    f"ComputedPulse candidate {pulse_mod.pulse_id!r} expected "
                    f"{self._stride} waveform(s) (style={self.ref_cfg.waveform.style}), "
                    f"got {len(wave_names)}"
                )
            flat_idxs.extend(prog.wave2idx[wn] for wn in wave_names)

        expected = list(range(min(flat_idxs), max(flat_idxs) + 1))
        if flat_idxs != expected:
            raise ValueError(
                "ComputedPulse candidate waveform indices must be contiguous, "
                f"got {flat_idxs}"
            )

        self.wmem_offset = min(flat_idxs)

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        ref_cfg = self.ref_cfg
        with prog.acquire_temp_reg(1) as (addr_reg,):
            # base = wmem_offset + gate_idx * stride
            if self._stride == 1:
                prog.write_reg_op(addr_reg, self.val_reg, "+", self.wmem_offset)
            else:
                # x*3 = (x<<1) + x; avoids relying on REG_WR multiplication.
                prog.write_reg_op(addr_reg, self.val_reg, "<<", 1)
                prog.write_reg_op(addr_reg, addr_reg, "+", self.val_reg)
                prog.write_reg_op(addr_reg, addr_reg, "+", self.wmem_offset)
            prog.pulse_wmem_reg(
                ref_cfg.ch,
                addr_reg,
                t=t + ref_cfg.pre_delay,
                flat_top_pulse=self._is_flat_top,
            )

        return t + self.total_length(prog)

    def ir_run(
        self,
        builder: IRBuilder,
        t: Union[float, QickParam],
        prog: ModularProgramV2,
    ) -> Union[float, QickParam]:
        ref_cfg = self.ref_cfg
        with prog.acquire_temp_reg(1) as (addr_reg,):
            # base = wmem_offset + gate_idx * stride
            if self._stride == 1:
                builder.ir_reg_op(addr_reg, self.val_reg, "+", self.wmem_offset)
            else:
                # x*3 = (x<<1) + x; avoid multiplication dependence.
                builder.ir_reg_op(addr_reg, self.val_reg, "SL", 1)
                builder.ir_reg_op(addr_reg, addr_reg, "+", self.val_reg)
                builder.ir_reg_op(addr_reg, addr_reg, "+", self.wmem_offset)
            builder.ir_pulse_wmem_reg(
                ref_cfg.ch,
                addr_reg,
                t=t + ref_cfg.pre_delay,
                flat_top_pulse=self._is_flat_top,
            )

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
