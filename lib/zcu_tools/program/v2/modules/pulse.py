from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias, TypedDict, Unpack

from pydantic import BeforeValidator
from qick.asm_v2 import QickParam

from .base import AbsModuleCfg, Module, apply_cfg_overrides
from .waveform import (
    AbsWaveform,
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
