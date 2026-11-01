# tolerated(file-size): Readout families share ADC routes, pulse templates, and acquisition contracts.
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias, TypedDict, Unpack

import numpy as np
from numpy.typing import NDArray
from pydantic import BeforeValidator
from qick.asm_v2 import Macro, QickParam

from zcu_tools.cfg_model import (
    Centroid3Classifier,
    CircleClassifier,
    LinearClassifier,
    ReadoutClassifier,
)
from zcu_tools.cfg_model import MuxReadoutCfg as CoreMuxReadoutCfg
from zcu_tools.utils.shot_classification import classify_shots, nearest_centroid3

from ..macro.readout import ReadoutEmission
from ..macro.wmem import PatchWmemFromDmem
from .base import AbsModuleCfg, Module, apply_cfg_overrides
from .dmem import LoadWord
from .pulse import Pulse, PulseCfg
from .waveform import WaveformCfg, resolve_waveform_ref

if TYPE_CHECKING:
    from zcu_tools.program.v2.timeline import TimelineProgramV2


class AbsReadoutCfg(AbsModuleCfg):
    @abstractmethod
    def build(self, name: str) -> AbsReadout: ...


class PulseReadoutCfg(AbsReadoutCfg):
    """Native generator/ADC readout, with one shared semantic frequency.

    gen_ch/ro_ch select generator/ADC channels; nqz is Nyquist zone 1 or 2.
    freq/mixer_freq are MHz, gain is normalized, phase is degrees. waveform is
    an expanded native envelope (arb holds its absolute asset path). pre_delay
    and post_delay are in us. pre_delay offsets generator playback; post_delay
    follows generator completion once. ro_length is the independent ADC
    window and trig_offset the trigger delay in us. Scalar knobs except mixer_freq accept
    QICK sweeps. desc is optional prose. type is readout/pulse. classifier stores
    offset-free normalized ADC calibration (including radians for linear angle);
    None means no calibration is available. The module does not apply it.
    """

    type: Literal["readout/pulse"] = "readout/pulse"
    gen_ch: int
    nqz: Literal[1, 2]
    freq: float | QickParam
    gain: float | QickParam
    phase: float | QickParam = 0.0
    waveform: Annotated[WaveformCfg, BeforeValidator(resolve_waveform_ref)]
    mixer_freq: float | None = None
    pre_delay: float | QickParam = 0.0
    post_delay: float | QickParam = 0.0
    ro_ch: int
    ro_length: float | QickParam
    trig_offset: float | QickParam = 0.0
    classifier: ReadoutClassifier | None = None

    def build(self, name: str) -> PulseReadout:
        """Capture this cfg as a named readout ending at generator plus post-delay."""
        return PulseReadout(name, self)

    def nominal_duration_us(self) -> float:
        """Return scalar readout completion time in us without clock rounding.

        pre_delay, post_delay and waveform length are finite nonnegative scalar us.
        Return pre-delay + generator waveform length + post-delay once.
        ADC trigger and integration do not extend this duration. No program or
        hardware is needed, and this cfg is unchanged. This nominal estimate
        does not quantize playback or the final duration to hardware clocks.
        Non-scalar timing raises TypeError during float conversion; waveform
        length lookup errors (for example an unreadable arb asset) propagate.
        """
        return (
            float(self.pre_delay) + float(self.waveform.length) + float(self.post_delay)
        )

    def to_pulse_cfg(self) -> PulseCfg:
        """Return a fresh outer PulseCfg containing only generator settings.

        gen_ch maps to ch; ADC channel, window and trigger settings are omitted.
        Pass ro_ch separately to Pulse when ADC frequency matching is needed.
        The waveform and any QickParam values are borrowed, not deep-copied;
        callers must not mutate these shared nested objects through the result.
        PulseCfg validation runs as usual and its errors propagate unchanged.
        """
        return PulseCfg(
            ch=self.gen_ch,
            nqz=self.nqz,
            freq=self.freq,
            gain=self.gain,
            phase=self.phase,
            waveform=self.waveform,
            mixer_freq=self.mixer_freq,
            pre_delay=self.pre_delay,
            post_delay=self.post_delay,
        )

    def set_param(self, name: str, value: float | QickParam) -> None:
        """Set shared freq (MHz), gain, waveform length or ro_length (us).

        Unsupported names raise ValueError; waveform length restrictions apply.
        """
        if name == "length":
            self.waveform.set_param(name, value)
        elif name in {"freq", "gain", "ro_length"}:
            setattr(self, name, value)
        else:
            raise ValueError(f"Unknown parameter: {name}")


class MuxReadoutCfg(CoreMuxReadoutCfg, AbsReadoutCfg):
    """Native fixed mux settings; numeric fields and validation belong to cfg_model.

    Inherited fields retain MHz, degrees, normalized gain and us units. build
    captures a copy for mixmux8/static PFBv4; every set_param call is rejected.
    Each tone's optional classifier retains its offset-free normalized ADC
    calibration; None means unavailable. The module does not apply it.
    """

    def build(self, name: str) -> MuxReadout:
        """Capture these numeric settings as a named fixed mux readout module."""
        return MuxReadout(name, self)


ReadoutCfg: TypeAlias = PulseReadoutCfg | MuxReadoutCfg


def _classifier_entries(
    cfg: ReadoutCfg,
) -> list[tuple[int, ReadoutClassifier | None]]:
    """List ADC numbers and borrowed classifiers in pulse/tone order."""
    return (
        [(cfg.ro_ch, cfg.classifier)]
        if isinstance(cfg, PulseReadoutCfg)
        else [(tone.ro_ch, tone.classifier) for tone in cfg.tones]
    )


def _classifier_for_adc(
    cfg: ReadoutCfg, ro_ch: int, *, calibration_hint: str
) -> ReadoutClassifier:
    """Find one ADC calibration and detach it, preserving caller-specific guidance."""
    for ch, classifier in _classifier_entries(cfg):
        if ch == ro_ch:
            if classifier is None:
                raise ValueError(f"ADC {ch}: classifier missing; {calibration_hint}")
            return classifier.model_copy(deep=True)
    raise ValueError(f"ADC {ro_ch}: not declared by readout")


def circle_classifier(cfg: ReadoutCfg, *, ro_ch: int) -> CircleClassifier:
    """Return detached circle calibration for the specified ADC channel number.

    cfg is a native Pulse or Mux readout; ro_ch identifies its declared ADC,
    not a tone position. Centers use raw integrated ADC / readout length,
    without IQ offset subtraction. Radius and confusion matrix may be absent.
    Raise ValueError for an unknown ADC, missing classifier, or non-circle mode. No hardware or program registry is read.
    """
    classifier = _classifier_for_adc(cfg, ro_ch, calibration_hint="run singleshot GE")
    if not isinstance(classifier, CircleClassifier):
        raise ValueError(
            f"ADC {ro_ch}: {classifier.mode} classifier is not circle; use circle GE centers"
        )
    return classifier


def require_circle_classifiers(cfg: ReadoutCfg) -> None:
    """Require circle classification with a radius on every declared ADC.

    cfg is a native Pulse or Mux readout. Check the pulse ADC or all mux tones
    in declaration order, not just the tone selected for acquisition. Missing
    classifiers, non-circle mode or absent radius raise ValueError naming the
    ADC; missing radius requests singleshot GE Post calibration. Circle emits
    g/e/other, not the g/e/f labels of centroid3. The cfg is unchanged; no
    hardware, program or live library is read. Returns None on success.
    """
    for ro_ch, _ in _classifier_entries(cfg):
        if circle_classifier(cfg, ro_ch=ro_ch).radius is None:
            raise ValueError(f"ADC {ro_ch}: radius missing; run singleshot GE Post")


def centroid3_classifier(cfg: ReadoutCfg, *, ro_ch: int) -> Centroid3Classifier:
    """Return detached three-state calibration for the declared ADC number.

    cfg is native Pulse or Mux readout; ro_ch is an ADC channel, not a tone
    position. Centers use raw integrated ADC / rounded integration sample count
    without IQ-offset subtraction. Training confusion rows are prepared g/e/f,
    columns measured g/e/f. Unknown ADC, missing or non-centroid3 classifier
    raises ValueError. No hardware, registry or live library is read.
    """
    classifier = _classifier_for_adc(
        cfg, ro_ch, calibration_hint="run qutrit reference calibration"
    )
    if not isinstance(classifier, Centroid3Classifier):
        raise ValueError(
            f"ADC {ro_ch}: {classifier.mode} classifier; require centroid3"
        )
    return classifier


def classification_processor(
    classifier: ReadoutClassifier,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Snapshot calibration as a normalized-IQ to one-hot shot processor.

    classifier is circle (g/e/other), linear (g/e), or centroid3 (g/e/f).
    Input float64 (..., 2) contains raw ADC integrals divided by readout length,
    without offset removal. Output float64 (..., k) preserves all leading axes.
    Circle uses strict radius and nearest-center regions; ties are other.
    Missing circle radius raises ValueError requesting GE Post calibration.
    Linear rotates by exp(+1j*angle), with equality assigned ground. Centroid3
    chooses nearest center, breaking ties in g/e/f order. No cfg is mutated.
    """
    if isinstance(classifier, CircleClassifier) and classifier.radius is None:
        raise ValueError("circle radius missing; run GE Post analysis")
    classifier = classifier.model_copy(deep=True)

    def process(iq: NDArray[np.float64]) -> NDArray[np.float64]:
        signals = iq[..., 0] + 1j * iq[..., 1]
        if isinstance(classifier, CircleClassifier):
            if classifier.radius is None:
                raise ValueError("circle radius missing; run GE Post analysis")
            return np.stack(
                classify_shots(
                    signals, classifier.g_center, classifier.e_center, classifier.radius
                ),
                axis=-1,
            ).astype(np.float64)
        if isinstance(classifier, LinearClassifier):
            excited = (
                signals * np.exp(1j * classifier.angle)
            ).real > classifier.threshold
            return np.stack([~excited, excited], axis=-1).astype(np.float64)
        labels = nearest_centroid3(
            signals, classifier.g_center, classifier.e_center, classifier.f_center
        )
        return np.eye(3, dtype=np.float64)[labels]

    return process


class AbsReadout(Module):
    """Base declaration for readout modules emitted on a TimelineProgramV2."""


class PulseReadoutOverrides(TypedDict, total=False):
    """Optional readout knobs, with None retaining each cfg value.

    freq is shared generator/ADC carrier MHz, gain normalized generator
    amplitude, length waveform duration us and ro_length ADC window us.
    Each accepts a scalar or QICK sweep. Waveform length restrictions and
    QICK ADC-window sweep compile limitations remain unchanged.
    """

    freq: float | QickParam | None
    gain: float | QickParam | None
    length: float | QickParam | None
    ro_length: float | QickParam | None


class Readout:
    """Dispatch cfg to its readout module; zcu_lab uses ``Readout.Cfg`` fields.

    Calling Readout(name, cfg) returns cfg.build(name), not a Readout instance.
    name labels the module; cfg supplies validated native readout settings.
    """

    Cfg: TypeAlias = ReadoutCfg

    def __new__(
        cls,
        name: str,
        cfg: AbsReadoutCfg,
        **overrides: Unpack[PulseReadoutOverrides],
    ) -> AbsReadout:
        """Build a detached readout with optional MHz/gain/us overrides.

        None keeps cfg. Pulse readout accepts freq, gain, length and ro_length;
        mux rejects non-None overrides with ValueError. Waveform set_param
        restrictions propagate. Unknown keys or nonnumeric values raise TypeError.
        The caller's cfg is never modified.
        """
        effective = deepcopy(cfg)
        apply_cfg_overrides(effective, overrides, PulseReadoutOverrides.__annotations__)
        return effective.build(name)


@dataclass(frozen=True)
class _ReadoutDeclaration:
    """Optional register-driven generator template and ADC configuration name."""

    runtime_pulse: Pulse | None
    runtime_ro_name: str | None


class PulseReadout(AbsReadout):
    """Play and acquire a pulse readout configured with ``PulseReadout.Cfg``.

    Timeline completion is generator end plus post-delay once. ADC integration
    stays on its channel, independent of the author timeline, until shot closure.
    A later timeline trigger on that ADC must prove the previous window complete.
    """

    Cfg: TypeAlias = PulseReadoutCfg

    def __init__(
        self,
        name: str,
        cfg: PulseReadoutCfg,
        *,
        gain_val: str | None = None,
        freq_val: str | None = None,
        ro_freq_val: str | None = None,
        **overrides: Unpack[PulseReadoutOverrides],
    ) -> None:
        """Capture a flat readout and optional hardware-word register inputs.

        name identifies ADC and pulse templates. cfg is copied and not mutated.
        gain_val names a hardware gain-word register; freq_val/ro_freq_val are
        paired generator/ADC frequency-word registers, already lowered for mixer
        and downconversion sign. Unspecified registers use cfg scalar/sweep values.
        freq (MHz), gain (normalized), length and ro_length (us) override the
        copied cfg via set_param. None keeps cfg; unsupported waveform length
        sweeps raise ValueError. init must precede run. Invalid QICK settings
        fail during initialization. Unknown keys or nonnumeric overrides raise TypeError.
        """
        self.name = name
        self.cfg = deepcopy(cfg)
        apply_cfg_overrides(self.cfg, overrides, PulseReadoutOverrides.__annotations__)
        self.gain_val = gain_val
        self.freq_val = freq_val
        self.ro_freq_val = ro_freq_val
        pulse_cfg = self.cfg.to_pulse_cfg()
        self.pulse = Pulse(name=f"{name}_pulse", cfg=pulse_cfg, ro_ch=self.cfg.ro_ch)

    @property
    def adc_name(self) -> str:
        """Registered ADC template name, also used for table-backed word updates."""
        return f"{self.name}_adc"

    def _declare_resources(
        self,
        prog: TimelineProgramV2,
        *,
        declare_pulse: Callable[[Pulse], None],
    ) -> _ReadoutDeclaration:
        declare_pulse(self.pulse)
        prog.declare_readout(ch=self.cfg.ro_ch, length=self.cfg.ro_length)
        prog.add_readoutconfig(
            ch=self.cfg.ro_ch,
            name=self.adc_name,
            freq=self.cfg.freq,
            gen_ch=self.cfg.gen_ch,
        )
        runtime_pulse = None
        runtime_ro_name = None
        if self._uses_runtime_pulse:
            runtime_pulse = Pulse(
                name=f"{self.name}_runtime_pulse",
                cfg=self.pulse.cfg,
                ro_ch=self.cfg.ro_ch,
                pulse_id=f"{self.name}_runtime_pulse",
            )
            declare_pulse(runtime_pulse)
        if self.ro_freq_val is not None:
            runtime_ro_name = f"{self.name}_adc_runtime"
            prog.add_readoutconfig(
                ch=self.cfg.ro_ch,
                name=runtime_ro_name,
                freq=self.cfg.freq,
                gen_ch=self.cfg.gen_ch,
            )
        return _ReadoutDeclaration(runtime_pulse, runtime_ro_name)

    @property
    def _uses_runtime_pulse(self) -> bool:
        return self.gain_val is not None or self.freq_val is not None

    def declare(self, prog: TimelineProgramV2) -> None:
        """Declare generator/ADC templates on prog before emission."""
        state = self._declare_resources(
            prog, declare_pulse=lambda pulse: pulse.declare(prog)
        )
        prog.set_module_state(self, state)

    def lower(
        self, prog: TimelineProgramV2, *, timeline: str | None, tag: str | None
    ) -> None:
        """Configure/trigger/play at one origin, then ceil generator end plus post_delay.

        All times are relative to the selected timeline. tag identifies the
        configuration event; generator pre_delay and ADC trig_offset are retained.
        Declaration and QICK timing errors propagate.
        """
        timeline = timeline if timeline is not None else "default"
        state = prog.get_module_state(self, _ReadoutDeclaration)
        prog.append_macro(ReadoutEmission(self, self.cfg.ro_ch, tag))
        if self.ro_freq_val is None:
            prog.send_readoutconfig(
                self.cfg.ro_ch, self.adc_name, t=0, tag=tag, timeline=timeline
            )
        else:
            runtime_ro_name = state.runtime_ro_name
            if runtime_ro_name is None:
                raise ValueError(f"{self.name}: runtime ADC template is not declared")
            prog.send_readoutconfig_from_regs(
                self.cfg.ro_ch,
                runtime_ro_name,
                t=0,
                tag=tag,
                freq_reg=self.ro_freq_val,
                timeline=timeline,
            )
        prog.trigger([self.cfg.ro_ch], t=self.cfg.trig_offset, timeline=timeline)
        if self._uses_runtime_pulse:
            runtime_pulse = state.runtime_pulse
            if runtime_pulse is None:
                raise ValueError(f"{self.name}: runtime pulse template is not declared")
            prog.pulse_from_regs(
                self.cfg.gen_ch,
                runtime_pulse.pulse_name(prog),
                t=self.cfg.pre_delay,
                freq_reg=self.freq_val,
                gain_reg=self.gain_val,
                timeline=timeline,
            )
        else:
            self.pulse.play(prog, timeline=timeline)
        prog.timeline_inc(self.cfg.post_delay, timeline=timeline)


class TablePulseReadout(AbsReadout):
    """Sweep a pulse readout through exact dmem frequency-word tables.

    The current point is played through QICK's ordinary wmem-backed pulse and
    readout-config path. At the sweep loop's exec-after hook, the next point is
    loaded from rotated dmem tables and persisted into dedicated templates. The
    final table entry wraps back to the first point for the next outer repetition.
    Timeline completion is generator end plus post-delay once; ADC windows do
    not extend it. A later timeline trigger must prove its ADC window disjoint.
    """

    def __init__(
        self,
        name: str,
        cfg: PulseReadoutCfg,
        *,
        idx_reg: str,
        freq_words: Sequence[int],
        ro_freq_words: Sequence[int],
    ) -> None:
        """Capture flat cfg and paired uint32 words for a named sweep loop.

        idx_reg must match exactly one declared loop with the same point count.
        Empty, unequal, or invalid uint32 tables raise ValueError. The first pair
        must encode cfg.freq; each later pair represents one semantic frequency.
        Dedicated templates isolate updates from shared pulses. The final update
        wraps to the first pair before the next outer repetition.
        """
        self.name = name
        self.idx_reg = idx_reg
        self.freq_words = [int(word) for word in freq_words]
        self.ro_freq_words = [int(word) for word in ro_freq_words]
        if len(self.freq_words) == 0:
            raise ValueError("TablePulseReadout requires at least one frequency word")
        if len(self.freq_words) != len(self.ro_freq_words):
            raise ValueError(
                "TablePulseReadout generator/readout tables must have equal length"
            )

        self._readout = PulseReadout(name, cfg)
        self.cfg = self._readout.cfg
        self._readout.pulse = Pulse(
            name=f"{name}_pulse",
            cfg=self._readout.pulse.cfg,
            ro_ch=self.cfg.ro_ch,
            pulse_id=f"{name}_table_pulse",
        )
        self._addr_reg = f"{name}_table_addr"

        self._freq_loader = LoadWord(
            name=f"{name}_next_freq",
            values=self.freq_words[1:] + self.freq_words[:1],
            idx_reg=idx_reg,
            val_reg=f"{name}_next_freq_word",
        )
        self._ro_freq_loader = LoadWord(
            name=f"{name}_next_ro_freq",
            values=self.ro_freq_words[1:] + self.ro_freq_words[:1],
            idx_reg=idx_reg,
            val_reg=f"{name}_next_ro_freq_word",
        )

    @property
    def initial_template(self) -> PulseReadout:
        """Borrow the initial scalar/swept template used before exec-after updates.

        The returned Module is owned by this table readout and must not be
        mutated by consumers. init(prog) initializes it and its dedicated pulse
        identity. It is not the current mutable hardware wmem contents; semantic
        interpreters own their own per-execution template-word update state.
        """
        return self._readout

    def declare(self, prog: TimelineProgramV2) -> None:
        """Declare independent templates and paired words for a public sweep loop.

        idx_reg must name a top-level native loop with exactly the table count;
        missing loops or count mismatches raise ValueError before allocation.
        Updates execute after each shot and wrap to the first pair.
        """
        matches = [loop for loop in prog.sweep_loops if loop.name == self.idx_reg]
        if len(matches) != 1:
            raise ValueError(
                f"TablePulseReadout idx_reg={self.idx_reg!r} requires one sweep loop"
            )
        if matches[0].count != len(self.freq_words):
            raise ValueError(
                f"TablePulseReadout table length {len(self.freq_words)} does not "
                f"match loop {self.idx_reg!r} count {matches[0].count}"
            )
        self._readout.declare(prog)
        self._freq_loader.declare(prog)
        self._ro_freq_loader.declare(prog)
        prog.add_reg(self._addr_reg)
        self._append_updates(prog)

    def lower(
        self, prog: TimelineProgramV2, *, timeline: str | None, tag: str | None
    ) -> None:
        """Play paired templates, ending at generator completion plus post-delay.

        Hook updates affect only frequency words, not placement or duration.
        timeline=None selects default; tag names this occurrence's config.
        """
        self._readout.lower(prog, timeline=timeline, tag=tag)

    def _append_updates(self, prog: TimelineProgramV2) -> None:
        pulse_id = self._readout.pulse.pulse_name(prog)
        prog.append_loop_after(
            self.idx_reg,
            PatchWmemFromDmem(
                name=pulse_id,
                idx_reg=self.idx_reg,
                addr_reg=self._addr_reg,
                val_reg=self._freq_loader.val_reg,
                dmem_offset=self._freq_loader.dmem_offset(prog),
            ),
            PatchWmemFromDmem(
                name=self._readout.adc_name,
                idx_reg=self.idx_reg,
                addr_reg=self._addr_reg,
                val_reg=self._ro_freq_loader.val_reg,
                dmem_offset=self._ro_freq_loader.dmem_offset(prog),
            ),
        )

    def _validate_loop_count(self, prog: TimelineProgramV2) -> None:
        sweep_dict = prog.sweep_dict
        if sweep_dict is None:
            raise ValueError(
                f"TablePulseReadout idx_reg={self.idx_reg!r} requires a sweep loop"
            )
        matches = [spec for name, spec in sweep_dict if name == self.idx_reg]
        if len(matches) != 1:
            raise ValueError(
                f"TablePulseReadout idx_reg={self.idx_reg!r} must match exactly "
                "one sweep loop"
            )
        spec = matches[0]
        count = spec if isinstance(spec, int) else spec.expts
        if count != len(self.freq_words):
            raise ValueError(
                f"TablePulseReadout table length {len(self.freq_words)} does not "
                f"match loop {self.idx_reg!r} count {count}"
            )


class MuxReadout(AbsReadout):
    """Fixed mixmux8 playback and simultaneous static PFBv4 acquisition.

    zcu_lab declares native settings as ``MuxReadout.Cfg``. Timeline completion
    is generator end plus post-delay once. Each ADC retains its own window for
    shot closure and overlap checks, without extending the author timeline.

    Only explicit tone channels are used, in their original order. Initialization
    rejects incompatible hardware, duplicate routes and sub-cycle windows. No
    envelope, tProc readout configuration or hardware parameter update is used.
    """

    Cfg: TypeAlias = MuxReadoutCfg

    def __init__(self, name: str, cfg: MuxReadoutCfg) -> None:
        """Copy numeric cfg; name identifies the const mask pulse and diagnostics."""
        self.name = name
        self.cfg = deepcopy(cfg)

    def declare(self, prog: TimelineProgramV2) -> None:
        """Declare tones/routes for playback and compiled-emission IQ collection.

        Incompatible wiring and illegal rounded durations raise ValueError.
        ADC/generator declarations follow QICK: later declarations overwrite
        earlier ones, and authors own their consistency. Read gen_chs/ro_chs
        for settings. QICK frequency/format errors propagate.
        """
        self._declare_resources(prog)

    def lower(
        self, prog: TimelineProgramV2, *, timeline: str | None, tag: str | None
    ) -> None:
        """Play all tones and trigger all ADCs at one origin, then add post once.

        The end covers the all-tone generator waveform plus post-delay once.
        ADC windows remain on their channels and do not extend this timeline.
        timeline=None selects default. Collection validates this compiled emission.
        """
        timeline = timeline if timeline is not None else "default"
        prog.append_macro(ReadoutEmission(self, self.cfg.tones[0].ro_ch, tag))
        prog.pulse(
            self.cfg.gen_ch, self.name, t=self.cfg.pre_delay, tag=tag, timeline=timeline
        )
        prog.trigger(
            [tone.ro_ch for tone in self.cfg.tones],
            t=self.cfg.trig_offset,
            timeline=timeline,
        )
        prog.timeline_inc(self.cfg.post_delay, timeline=timeline)

    def _declare_resources(self, prog: TimelineProgramV2) -> None:
        self._validate_generator(prog)
        self._validate_readouts(prog)
        cfg = self.cfg
        prog.declare_gen(
            ch=cfg.gen_ch,
            nqz=cfg.nqz,
            mixer_freq=cfg.mixer_freq,
            ro_ch=cfg.tones[0].ro_ch,
            mux_freqs=[tone.freq for tone in cfg.tones],
            mux_gains=[tone.gain for tone in cfg.tones],
            mux_phases=[tone.phase for tone in cfg.tones],
        )
        for tone in cfg.tones:
            prog.declare_readout(
                ch=tone.ro_ch,
                length=cfg.ro_length,
                freq=tone.freq,
                phase=0,
                gen_ch=cfg.gen_ch,
            )
        prog.add_pulse(
            ch=cfg.gen_ch,
            name=self.name,
            style="const",
            mask=list(range(len(cfg.tones))),
            length=cfg.length,
        )

    def _validate_generator(self, prog: TimelineProgramV2) -> None:
        ch = self.cfg.gen_ch
        gens = prog.soccfg["gens"]
        if ch >= len(gens):
            raise ValueError(f"{self.name}.gen_ch: missing generator {ch}")
        gen = gens[ch]
        if (
            gen["type"] != "axis_sg_mixmux8_v1"
            or not gen["has_mixer"]
            or not gen["has_gain"]
            or not gen["has_phase"]
            or len(self.cfg.tones) > gen["n_tones"]
        ):
            raise ValueError(
                f"{self.name}.gen_ch: generator {ch} requires mixmux8 mixer/gain/phase capacity"
            )
        cycles = prog.us2cycles(self.cfg.length, gen_ch=ch)
        if not 3 <= cycles < 2**32:
            raise ValueError(
                f"{self.name}.length: generator {ch} rounded length must be 3..2**32-1 cycles"
            )

    def _validate_readouts(self, prog: TimelineProgramV2) -> None:
        channels: set[int] = set()
        ports: set[int] = set()
        path: str | None = None
        readouts = prog.soccfg["readouts"]
        for i, tone in enumerate(self.cfg.tones):
            ch = tone.ro_ch
            where = f"{self.name}.tones[{i}].ro_ch: ADC {ch}"
            if ch >= len(readouts) or ch in channels:
                raise ValueError(f"{where} missing or duplicate readout route")
            ro = readouts[ch]
            if (
                ro["ro_type"] != "axis_pfb_readout_v4"
                or "tproc_ctrl" in ro
                or not ro.get("ro_fullpath")
                or not ro["pfb_dds_on_output"]
            ):
                raise ValueError(f"{where} requires static PFBv4")
            if path is not None and path != ro["ro_fullpath"]:
                raise ValueError(f"{where} must use the same PFB ro_fullpath")
            path = ro["ro_fullpath"]
            port = ro["pfb_port"]
            if (
                len(self.cfg.tones) > ro["pfb_nout"]
                or not 0 <= port < ro["pfb_nout"]
                or port in ports
            ):
                raise ValueError(f"{where} invalid or duplicate PFB output/capacity")
            if prog.us2cycles(self.cfg.ro_length, ro_ch=ch) <= 0:
                raise ValueError(
                    f"{self.name}.ro_length: ADC {ch} rounded window must be positive"
                )
            channels.add(ch)
            ports.add(port)

    def collect_raw_iq(
        self,
        prog: TimelineProgramV2,
        *,
        read_index: int | None = None,
    ) -> NDArray[np.float64]:
        """Return last-round raw IQ in tone order, normalized by ADC length.

        prog must contain this module's compiled emission.
        Call after its
        integrated acquire(), or in a completed-round hook. This reads get_raw()
        from that program, never acquires or samples a new round. A simulation
        getter may materialize retained shots using that round's fixed identity.
        Multiple rounds are NOT concatenated: get_raw() retains only the last
        round. Collect in each completed-round hook to retain every round.

        Input buffers must be int64 QICK sums shaped (*prog.loop_dims,reads,2)
        in actual ADC declaration order. The returned independent float64 array
        has shape (*prog.loop_dims,reads,N,2), with N in cfg.tones order and the
        final axis [I,Q]. Every loop axis, including reps at prog.avg_level, keeps
        its original position and shot order. No preparation meaning is assigned.
        read_index=None keeps every read; a nonnegative int selects the same read
        on all tones and returns (*prog.loop_dims,N,2). No shots are averaged.

        Values are ADC units per integration sample: sums divided by each ADC's
        declared, rounded length in samples. DC offsets remain, unlike QICK's
        default averaged acquisition. No rotation or classification is applied.
        Hardware and simulation get_raw() share this normalization contract.

        ValueError names this module for the wrong/uninitialized program, absent
        raw buffers/channels, count/dtype/loop/read/IQ shape mismatches or invalid
        read_index. Unrelated channels' shapes are ignored. Getter errors propagate.
        """
        self._validate_collection_program(prog)
        if read_index is not None and (type(read_index) is not int or read_index < 0):
            raise ValueError(
                f"{self.name}: read_index must be None or a nonnegative integer"
            )
        positions = {ch: i for i, ch in enumerate(prog.ro_chs)}
        for i, tone in enumerate(self.cfg.tones):
            if tone.ro_ch not in positions:
                raise ValueError(f"{self.name}: tone {i} missing ro_ch {tone.ro_ch}")
        raw = prog.get_raw()
        if raw is None:
            raise ValueError(
                f"{self.name}: no raw buffers; acquire integrated IQ first"
            )
        if len(raw) != len(positions):
            raise ValueError(
                f"{self.name}: raw item count {len(raw)} != channels {len(positions)}"
            )
        if prog.loop_dims is None:
            raise ValueError(f"{self.name}: raw loop dimensions are unavailable")
        reads = prog.ro_chs[self.cfg.tones[0].ro_ch]["trigs"]
        expected_shape = (*prog.loop_dims, reads, 2)
        selected: list[NDArray[np.float64]] = []
        for i, tone in enumerate(self.cfg.tones):
            pos = positions[tone.ro_ch]
            array = raw[pos]
            ro = prog.ro_chs[tone.ro_ch]
            where = f"{self.name}: tone {i} ro_ch {tone.ro_ch} raw[{pos}]"
            if array.dtype != np.int64:
                raise ValueError(f"{where} must be int64 QICK sums")
            if reads <= 0 or ro["trigs"] != reads or array.shape != expected_shape:
                raise ValueError(f"{where} shape {array.shape} != {expected_shape}")
            if read_index is not None and read_index >= reads:
                raise ValueError(f"{where} read_index {read_index} out of bounds")
            iq = array.astype(np.float64) / ro["length"]
            selected.append(iq if read_index is None else iq[..., read_index, :])
        return np.stack(selected, axis=-2)

    def _validate_collection_program(self, prog: TimelineProgramV2) -> None:
        from .._timeline_macros import (
            ModuleEmission,
            PreparedTimelineMacro,
            TimelineBranch,
            TimeProducer,
        )

        def emitted(macros: Sequence[Macro]) -> bool:
            for macro in macros:
                source = (
                    macro.source if isinstance(macro, PreparedTimelineMacro) else macro
                )
                if isinstance(source, ReadoutEmission) and source.owner is self:
                    return True
                if isinstance(source, (ModuleEmission, TimeProducer)) and emitted(
                    source.macros
                ):
                    return True
                if isinstance(source, TimelineBranch) and any(
                    emitted(case) for case in source.cases
                ):
                    return True
            return False

        if not emitted(prog.macro_list):
            raise ValueError(
                f"{self.name}: module has no compiled readout emission on this program"
            )

    def collect_iq(
        self,
        prog: TimelineProgramV2,
        acquired: Sequence[NDArray[np.float64]],
        *,
        read_index: int = 0,
    ) -> NDArray[np.float64]:
        """Select one averaged read in tone order, returning (*sweep_shape,N,2).

        prog must contain this module's compiled emission.
        Caller must supply that program's same acquire result:
        an unclassified averaged list of float64 arrays (reads,*sweep_shape,2).
        The list carries no source identity, so another acquisition's same-shaped
        arrays cannot be detected. Raw, single-shot, classified and decimated
        outputs are not supported. Values (including NaN), normalization and
        units are preserved; no complex conversion, interpretation or I/O occurs.

        ValueError names this module and the invalid position/tone/channel for
        missing initialization/channels, list count, read_index, dtype or shape.
        Unrelated channels' shapes are ignored.
        """
        self._validate_collection_program(prog)
        positions = {ch: i for i, ch in enumerate(prog.ro_chs)}
        for i, tone in enumerate(self.cfg.tones):
            if tone.ro_ch not in positions:
                raise ValueError(f"{self.name}: tone {i} missing ro_ch {tone.ro_ch}")
        if len(acquired) != len(positions):
            raise ValueError(
                f"{self.name}: acquired item count {len(acquired)} != channels {len(positions)}"
            )
        if type(read_index) is not int or read_index < 0:
            raise ValueError(f"{self.name}: read_index must be a nonnegative integer")
        selected: list[NDArray[np.float64]] = []
        shape: tuple[int, ...] | None = None
        for i, tone in enumerate(self.cfg.tones):
            pos = positions[tone.ro_ch]
            array = acquired[pos]
            where = f"{self.name}: tone {i} ro_ch {tone.ro_ch} acquired[{pos}]"
            if array.dtype != np.float64 or array.ndim < 2 or array.shape[-1] != 2:
                raise ValueError(f"{where} must be float64 (reads,*sweep_shape,2)")
            if shape is not None and array.shape != shape:
                raise ValueError(f"{where} read/sweep shape {array.shape} != {shape}")
            if read_index >= array.shape[0]:
                raise ValueError(f"{where} read_index {read_index} out of bounds")
            shape = array.shape
            selected.append(array[read_index])
        return np.stack(selected, axis=-2)
