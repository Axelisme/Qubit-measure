from __future__ import annotations

import math
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias, TypedDict, Unpack

from pydantic import BeforeValidator
from qick.asm_v2 import QickParam

from .._timeline_selected import PlaybackChoice, SelectedPlayback
from .base import AbsModuleCfg, Module, apply_cfg_overrides
from .dmem import LoadValue
from .waveform import (
    AbsWaveform,
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    WaveformCfg,
    resolve_waveform_ref,
)

if TYPE_CHECKING:
    from zcu_tools.program.v2.timeline import TimelineProgramV2


class PulseCfg(AbsModuleCfg):
    """Native generator pulse; ADC matching is a Pulse runtime input.

    ch selects the generator, nqz is Nyquist zone 1 or 2. freq/mixer_freq are
    MHz; phase is degrees; gain is normalized. waveform is the native envelope;
    pre_delay/post_delay are us. Frequency, phase, gain and delays accept QICK
    sweeps. mixer_freq optionally configures mixing. desc is optional prose;
    type is pulse.
    """

    type: Literal["pulse"] = "pulse"
    waveform: Annotated[WaveformCfg, BeforeValidator(resolve_waveform_ref)]
    ch: int
    nqz: Literal[1, 2]
    freq: float | QickParam
    phase: float | QickParam = 0.0
    gain: float | QickParam
    pre_delay: float | QickParam = 0.0
    post_delay: float | QickParam = 0.0

    mixer_freq: float | None = None

    def build(self, name: str) -> Pulse:
        return Pulse(name, self)

    def set_param(self, name: str, value: float | QickParam) -> None:
        if name == "length":
            self.waveform.set_param(name, value)
        elif name in {"gain", "freq", "phase"}:
            setattr(self, name, value)
        else:
            raise ValueError(f"Unknown parameter: {name}")


class PulseOverrides(TypedDict, total=False):
    """Optional generator knobs; every field accepts a scalar, sweep or None.

    freq is the carrier in MHz, gain the normalized amplitude, phase degrees,
    and length the waveform duration in us. None keeps the cfg value. Length
    sweeps are accepted only by waveforms that support set_param length sweeps.
    """

    freq: float | QickParam | None
    gain: float | QickParam | None
    phase: float | QickParam | None
    length: float | QickParam | None


@dataclass(frozen=True)
class _PulseDeclaration:
    """Program-local QICK pulse name and its built envelope (including length)."""

    pulse_id: str
    waveform: AbsWaveform


class Pulse(Module):
    """Emit a generator pulse; zcu_lab declares its settings as ``Pulse.Cfg``."""

    Cfg: TypeAlias = PulseCfg

    def __init__(
        self,
        name: str,
        cfg: PulseCfg | None,
        pulse_id: str | None = None,
        *,
        ro_ch: int | None = None,
        **overrides: Unpack[PulseOverrides],
    ) -> None:
        """Capture a pulse and optional runtime ADC channel for frequency matching.

        ro_ch is None for unpaired playback or an ADC channel number. It affects
        QICK frequency rounding, mixer declaration and pulse registry identity.
        cfg is copied; None emits no pulse. Explicit pulse_id disables sharing.
        freq (MHz), gain (normalized), phase (degrees) and length (us) override
        the copied cfg via set_param; None keeps cfg. Waveform length restrictions
        apply, including ValueError for unsupported sweeps. With cfg=None,
        non-None overrides raise ValueError because there is no pulse to update.
        Unknown override keys or nonnumeric values raise TypeError.
        """
        self.name = name
        self.cfg = deepcopy(cfg) if cfg is not None else None
        apply_cfg_overrides(self.cfg, overrides, PulseOverrides.__annotations__)
        self._requested_pulse_id = pulse_id
        self.ro_ch = ro_ch

    def _add_pulse(
        self,
        prog: TimelineProgramV2,
        pulse_id: str,
        waveform: AbsWaveform,
    ) -> None:
        cfg = self.cfg
        assert cfg is not None

        ro_ch = self.ro_ch if cfg.mixer_freq is not None else None
        prog.declare_gen(
            cfg.ch,
            nqz=cfg.nqz,
            mixer_freq=cfg.mixer_freq,
            ro_ch=ro_ch,
        )

        waveform.create(prog, cfg.ch)
        prog.add_pulse(
            cfg.ch,
            pulse_id,
            ro_ch=self.ro_ch,
            freq=cfg.freq,
            phase=cfg.phase,
            gain=cfg.gain,
            **waveform.to_wav_kwargs(),
        )

    def declare(self, prog: TimelineProgramV2) -> None:
        """Register this program's pulse template; QICK errors propagate."""
        if self.cfg is None:
            return
        pulse_id = self._requested_pulse_id
        if pulse_id is None:
            pulse_id = prog.pulse_registry.calc_name(self.cfg, ro_ch=self.ro_ch)
            create = prog.pulse_registry.register(self.name, self.cfg, ro_ch=self.ro_ch)
        else:
            create = True
        waveform = self.cfg.waveform.build(f"{pulse_id}_waveform")
        prog.set_module_state(self, _PulseDeclaration(pulse_id, waveform))
        if create:
            self._add_pulse(prog, pulse_id, waveform)

    def pulse_name(self, prog: TimelineProgramV2) -> str:
        """Return the pulse identifier declared on prog for native QICK calls.

        prog must have declared this nonempty pulse.
        Missing declaration or empty cfg raises ValueError.
        """
        return prog.get_module_state(self, _PulseDeclaration).pulse_id

    def lower(
        self, prog: TimelineProgramV2, *, timeline: str | None, tag: str | None
    ) -> None:
        """Play at timeline plus pre_delay; ceil actual end plus post_delay once.

        Empty cfg has zero duration and no playback.
        QICK resource/timing errors propagate.
        """
        timeline = timeline if timeline is not None else "default"
        self.play(prog, timeline=timeline, tag=tag)
        if self.cfg is not None:
            prog.timeline_inc(self.cfg.post_delay, timeline=timeline)

    def play(
        self,
        prog: TimelineProgramV2,
        *,
        timeline: str | None,
        tag: str | None = None,
        t: float | QickParam = 0.0,
    ) -> None:
        """Lower playback at t plus pre_delay, without adding owner duration.

        Composite owners call this inside one emission, which holds their events
        at the same timeline origin. Outside emission, timeline pulse placement
        advances its end normally. t is relative us. Empty cfg emits nothing;
        undeclared pulses raise ValueError. tag names only this playback occurrence.
        """
        if self.cfg is None:
            return
        pulse_id = self.pulse_name(prog)
        prog.pulse(
            self.cfg.ch,
            pulse_id,
            t=t + self.cfg.pre_delay,
            tag=tag,
            timeline=timeline,
        )


class TableLengthPulse(Module):
    """Sweep const/flat-top pulse length from dmem with one wmem template.

    The hardware loop counter selects generator-length and tProcessor-duration
    tables. Every entry must be strictly positive and satisfy the generator's
    three-cycle minimum. Readout alignment uses the
    runtime duration table, so shorter pulses are not padded to the longest point.
    """

    def __init__(
        self,
        name: str,
        cfg: PulseCfg,
        *,
        lengths: Iterable[float],
        idx_reg: str,
    ) -> None:
        self.name = name
        self.cfg = deepcopy(cfg)
        self.lengths = [float(length) for length in lengths]
        self.idx_reg = idx_reg

        if not self.lengths:
            raise ValueError("TableLengthPulse requires at least one length")
        if any(not math.isfinite(length) or length <= 0.0 for length in self.lengths):
            raise ValueError("TableLengthPulse lengths must be finite and > 0")
        if not isinstance(self.cfg.waveform, (ConstWaveformCfg, FlatTopWaveformCfg)):
            raise NotImplementedError(
                "TableLengthPulse supports only const and flat_top waveforms"
            )

        prefix = f"{name}_table_length"
        self._length_reg = f"{prefix}_gen_cycles"
        self._template = Pulse(f"{name}_template", self.cfg)

    def declare(self, prog: TimelineProgramV2) -> None:
        """Declare one template and the actual generator-length table, not durations.

        The caller owns idx_reg. Entries must satisfy generator minimum/maximum
        lengths and flat-top ramp constraints. No legacy duration table is built;
        native playback timing is resolved from the selected actual waveform.
        """
        wave_cycles = [
            self._wave_length_cycles(prog, length) for length in self.lengths
        ]
        template_cfg = self._template.cfg
        if template_cfg is None:
            raise ValueError(f"{self.name}: missing runtime pulse template")
        template_cfg.set_param("length", self.lengths[0])
        self._template.declare(prog)
        length_loader = LoadValue(
            f"{self.name}_load_gen_length",
            wave_cycles,
            idx_reg=self.idx_reg,
            val_reg=self._length_reg,
        )
        length_loader.declare(prog)
        prog.set_module_state(self, length_loader)

    def lower(
        self, prog: TimelineProgramV2, *, timeline: str | None, tag: str | None
    ) -> None:
        """Play the selected actual generator length; count pre/post once.

        The caller initializes idx_reg with certified in-range producers.
        Timeline owner resolves flat-top ramps, exact end rounding and audits.
        """
        del tag  # ModuleEmission owns the occurrence tag.
        relation = prog.get_module_state(self, LoadValue).lower_selected(prog)
        name = self._template.pulse_name(prog)
        prog.pulse_from_length_reg(
            self.cfg.ch,
            name,
            length_reg=self._length_reg,
            flat_top=isinstance(self.cfg.waveform, FlatTopWaveformCfg),
            t=self.cfg.pre_delay,
            timeline="default" if timeline is None else timeline,
            selection=SelectedPlayback(
                relation.selector,
                relation.register,
                tuple(
                    PlaybackChoice(name, length_cycles=cycles)
                    for cycles in relation.table.values
                ),
                relation,
            ),
        )
        prog.timeline_inc(
            self.cfg.post_delay, timeline="default" if timeline is None else timeline
        )

    def _wave_length_cycles(self, prog: TimelineProgramV2, length: float) -> int:
        wave_length = length
        if isinstance(self.cfg.waveform, FlatTopWaveformCfg):
            wave_length -= float(self.cfg.waveform.raise_waveform.length)
            if wave_length <= 0.0:
                raise ValueError(
                    f"TableLengthPulse flat_top length {length} us must exceed "
                    f"the fixed ramp length {self.cfg.waveform.raise_waveform.length} us"
                )

        cycles = int(prog.us2cycles(gen_ch=self.cfg.ch, us=wave_length))
        if cycles < 3:
            raise ValueError(
                f"TableLengthPulse non-zero length {length} us resolves to "
                f"{cycles} generator cycles; hardware requires at least 3"
            )
        if cycles >= 2**16:
            raise ValueError(
                f"TableLengthPulse length {length} us resolves to {cycles} "
                "generator cycles; hardware limit is less than 2**16"
            )
        return cycles
