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

    def __init__(
        self,
        name: str,
        *,
        val_reg: str,
        pulses: Sequence[Optional[PulseCfg]],
    ) -> None:
        self.name = name
        self.val_reg = val_reg
        raw_pulses = list(pulses)
        if len(raw_pulses) < 2:
            raise ValueError("ComputedPulse requires at least 2 candidate pulses")

        first_cfg = next((cfg for cfg in raw_pulses if cfg is not None), None)
        if first_cfg is None:
            raise ValueError("ComputedPulse requires at least one non-None pulse cfg")

        self._wmem_base = 0
        self._pulse_modules: list[Pulse] = []
        self._ch: Optional[int] = None
        self._pre_delay: float = 0.0
        self._max_total_length = 0.0

        def to_float_scalar(value: Union[float, QickParam], field: str) -> float:
            if isinstance(value, QickParam):
                raise NotImplementedError(
                    f"ComputedPulse does not support swept {field} in candidate pulses"
                )
            return float(value)

        pulse_cfgs: list[PulseCfg] = []
        timings: list[tuple[float, float, float]] = []
        pre_delay_ref: Optional[float] = None
        for cfg in raw_pulses:
            pulse_cfg = first_cfg.with_updates(gain=0.0) if cfg is None else cfg

            if pulse_cfg.waveform.style == "flat_top":
                raise NotImplementedError(
                    "ComputedPulse does not support flat_top waveform"
                )

            if self._ch is None:
                self._ch = pulse_cfg.ch
            elif self._ch != pulse_cfg.ch:
                raise ValueError(
                    "All ComputedPulse candidates must use the same channel"
                )

            pre_delay = to_float_scalar(pulse_cfg.pre_delay, "pre_delay")
            post_delay = to_float_scalar(pulse_cfg.post_delay, "post_delay")
            length = to_float_scalar(pulse_cfg.waveform.length, "length")
            if pre_delay_ref is None:
                pre_delay_ref = pre_delay
            elif pre_delay_ref != pre_delay:
                raise ValueError(
                    "All ComputedPulse candidates must share identical pre_delay"
                )

            pulse_cfgs.append(pulse_cfg)
            timings.append((pre_delay, length, post_delay))

        assert self._ch is not None
        assert pre_delay_ref is not None

        self._pre_delay = pre_delay_ref
        total_lengths = [pre + length + post for pre, length, post in timings]
        self._max_total_length = max(total_lengths)
        self._pulse_modules = [
            Pulse(f"{self.name}_cand_{i}", cfg, block_mode=True)
            for i, cfg in enumerate(pulse_cfgs)
        ]

    def init(self, prog: ModularProgramV2) -> None:
        wave_idxs: list[int] = []
        for pulse_mod in self._pulse_modules:
            pulse_mod.init(prog)

            wave_names = prog.list_pulse_waveforms(pulse_mod.pulse_id)
            if len(wave_names) != 1:
                raise ValueError(
                    "ComputedPulse requires one waveform per candidate pulse "
                    f"(got {len(wave_names)} for {pulse_mod.pulse_id})"
                )
            wave_idxs.append(prog.wave2idx[wave_names[0]])

        sorted_idxs = sorted(wave_idxs)
        expected = list(range(sorted_idxs[0], sorted_idxs[0] + len(sorted_idxs)))
        if sorted_idxs != expected:
            raise ValueError(
                "ComputedPulse candidate waveform indices must be contiguous, "
                f"got {sorted_idxs}"
            )

        self._wmem_base = min(wave_idxs)

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        assert self._ch is not None

        with prog.acquire_temp_reg(1) as (addr_reg,):
            # addr_reg = base_wave_idx + gate_idx
            prog.write_reg_op(addr_reg, self.val_reg, "+", self._wmem_base)
            prog.pulse_wmem_reg(self._ch, addr_reg, t=t + self._pre_delay)

        return t + self._max_total_length

    def allow_rerun(self) -> bool:
        return True
