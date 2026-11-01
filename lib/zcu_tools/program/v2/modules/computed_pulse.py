from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.dmem import LoadValue
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg

from .._timeline_selected import PlaybackChoice, SelectedPlayback

if TYPE_CHECKING:
    from zcu_tools.program.v2.timeline import TimelineProgramV2

logger = logging.getLogger(__name__)


class ComputedPulse(Module):
    """Select one primitive pulse by register-computed wmem address."""

    def __init__(self, name: str, *, val_reg: str, pulses: Sequence[PulseCfg]) -> None:
        self.name = name
        self.val_reg = val_reg

        raw_cfgs = list(pulses)
        if len(raw_cfgs) < 2:
            raise ValueError("ComputedPulse requires at least 2 candidate pulses")

        self.ref_cfg = raw_cfgs[0]

        if any(cfg.ch != self.ref_cfg.ch for cfg in raw_cfgs):
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

    def declare(self, prog: TimelineProgramV2) -> None:
        """Declare candidate waveforms and their selector-indexed wmem table.

        Candidate channels, pre-delays and waveform strides must agree. Unsupported
        int4 flat-top layouts raise ValueError. The selector register is borrowed;
        its bounds and all address writes are audited when consumed as time.
        """
        base_idxs = self._declare_candidates(
            prog, declare_pulse=lambda pulse: pulse.declare(prog)
        )
        # Address words are consumed by this emission before any next module
        # lookup. Selected ends keep their own ADR-0083 runtime storage.
        address_register = prog.add_reg("_computed_wmem_index", allow_reuse=True)
        address_loader = LoadValue(
            f"{self.name}_address",
            base_idxs,
            idx_reg=self.val_reg,
            val_reg=address_register,
            use_existed=True,
        )
        address_loader.declare(prog)
        prog.set_module_state(self, address_loader)

    def lower(
        self, prog: TimelineProgramV2, *, timeline: str | None, tag: str | None
    ) -> None:
        """Play the selected declared waveform with its own post, without padding.

        The borrowed selector must have certified in-range producers on every
        path. Address/selector writes and same-channel overlap fail at compile.
        """
        del tag  # ModuleEmission owns the occurrence tag.
        address_loader = prog.get_module_state(self, LoadValue)
        # ModuleEmission finishes the paired selected post before another
        # ComputedPulse can overwrite this capture. It is not a temp arithmetic
        # register: lookup scratch and pulse address preparation must not alias it.
        captured = prog.add_reg("_computed_selected_index", allow_reuse=True)
        relation = address_loader.lower_selected(prog, captured_index=captured)
        choices = tuple(
            PlaybackChoice(pulse.pulse_name(prog), pulse.cfg.post_delay)
            for pulse in self.pulse_modules
            if pulse.cfg is not None
        )
        prog.pulse_by_reg(
            self.ref_cfg.ch,
            address_loader.val_reg,
            t=self.ref_cfg.pre_delay,
            flat_top_pulse=self._is_flat_top,
            timeline="default" if timeline is None else timeline,
            selection=SelectedPlayback(
                relation.selector, relation.register, choices, relation
            ),
        )

    def _declare_candidates(
        self,
        prog: TimelineProgramV2,
        *,
        declare_pulse: Callable[[Pulse], None],
    ) -> list[int]:
        if self._is_flat_top:
            gen_type = prog.soccfg["gens"][self.ref_cfg.ch].get("type", "")
            if gen_type in ("axis_sg_int4_v1", "axis_sg_int4_v2"):
                raise ValueError(
                    "ComputedPulse does not support flat_top on int4 generators "
                    f"(channel {self.ref_cfg.ch}, type {gen_type!r}); QICK appends "
                    "an extra dummy wave that breaks the wmem stride assumption"
                )

        for pulse_mod in self.pulse_modules:
            declare_pulse(pulse_mod)

        # Collect the starting wmem index for each candidate and store them in
        # a dmem table. This removes the contiguity requirement: candidates may
        # share waveforms (PulseRegistry dedup) or be non-contiguous without
        # causing errors.
        base_idxs: list[int] = []
        for pulse_mod in self.pulse_modules:
            wave_names = prog.list_pulse_waveforms(
                pulse_mod.pulse_name(prog), exclude_special=False
            )
            if len(wave_names) != self._stride:
                raise ValueError(
                    f"ComputedPulse candidate {pulse_mod.pulse_name(prog)!r} expected "
                    f"{self._stride} waveform(s) (style={self.ref_cfg.waveform.style}), "
                    f"got {len(wave_names)}"
                )
            base_idxs.append(prog.wave2idx[wave_names[0]])

        return base_idxs
