"""Persistent module Specs and materialized Cfgs, without program execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, field_validator
from qick.asm_v2 import QickParam

from .base import ConfigBase
from .field_meta import FieldMeta
from .waveform import WaveformCfg, WaveformSpec


class _ModuleFields(ConfigBase):
    """Common module metadata: type is the family discriminator, desc is optional prose."""

    type: str
    desc: str | None = None


class AbsModuleCfg(_ModuleFields, ABC):
    """Materialized module parameters; mutation validates assignments."""

    @abstractmethod
    def set_param(self, name: str, value: float | QickParam) -> None:
        """Update a family-specific knob; unsupported names or values raise ValueError."""


class _PulseFields[WaveformT](_ModuleFields):
    """Pulse waveform and scalar knobs shared by persistent and materialized schemas.

    type is pulse. ch is the explicit generator channel; nqz is Nyquist zone 1
    or 2. freq and mixer_freq are MHz, phase is degrees, gain is normalized,
    and pre_delay/post_delay are µs. freq, phase, gain and delays accept QICK
    sweeps. Optional mixer_freq configures mixing; None leaves it unspecified.
    Readout-channel matching is supplied at runtime, not persisted.
    """

    type: Literal["pulse"] = "pulse"
    waveform: Annotated[WaveformT, FieldMeta(label="Waveform")]
    ch: Annotated[int, FieldMeta(label="Gen ch")]
    nqz: Annotated[Literal[1, 2], FieldMeta(label="NQZ", group="Advanced")]
    freq: Annotated[float | QickParam, FieldMeta(label="Freq", unit="MHz", decimals=3)]
    phase: Annotated[
        float | QickParam,
        FieldMeta(label="Phase", unit="deg", decimals=2, group="Advanced"),
    ] = 0.0
    gain: Annotated[float | QickParam, FieldMeta(label="Gain", decimals=4)]
    pre_delay: Annotated[
        float | QickParam,
        FieldMeta(label="Pre-delay", unit="us", decimals=3, group="Advanced"),
    ] = 0.0
    post_delay: Annotated[
        float | QickParam,
        FieldMeta(label="Post-delay", unit="us", decimals=3, group="Advanced"),
    ] = 0.0
    mixer_freq: Annotated[
        float | None,
        FieldMeta(label="Mixer freq", unit="MHz", decimals=3, group="Advanced"),
    ] = None


class PulseSpec(_PulseFields[str | WaveformSpec]):
    """Persistent pulse; waveform is a concrete library key or inline WaveformSpec."""


class PulseCfg(_PulseFields[WaveformCfg], AbsModuleCfg):
    """Materialized pulse; waveform is an expanded, validated WaveformCfg."""

    def set_param(self, name: str, value: float | QickParam) -> None:
        """Set gain/freq/phase or delegate length to the waveform; otherwise raise ValueError."""
        if name == "length":
            self.waveform.set_param(name, value)
        elif name in {"gain", "freq", "phase"}:
            setattr(self, name, value)
        else:
            raise ValueError(f"Unknown parameter: {name}")


class AbsReadoutCfg(AbsModuleCfg):
    """Materialized readout family with generator-driven acquisition."""


class _PulseReadoutFields[WaveformT](_ModuleFields):
    """Independent generator and ADC settings for type readout/pulse.

    gen_ch selects the generator and ro_ch the ADC. nqz is Nyquist zone 1 or 2.
    freq is the shared generator/ADC frequency in MHz; mixer_freq is an optional
    generator mixer frequency in MHz. gain is normalized and phase is degrees.
    pre_delay/post_delay are generator delays in us; ro_length is the integration
    window in us and trig_offset is the trigger delay in us. Frequency, gain,
    phase and timing accept QICK sweeps in materialized cfgs, not persistent Specs.
    """

    type: Literal["readout/pulse"] = "readout/pulse"
    waveform: Annotated[WaveformT, FieldMeta(label="Waveform")]
    gen_ch: Annotated[int, FieldMeta(label="Gen ch")]
    nqz: Annotated[Literal[1, 2], FieldMeta(label="NQZ", group="Advanced")]
    freq: Annotated[float | QickParam, FieldMeta(label="Freq", unit="MHz", decimals=3)]
    gain: Annotated[float | QickParam, FieldMeta(label="Gain", decimals=4)]
    phase: Annotated[
        float | QickParam,
        FieldMeta(label="Phase", unit="deg", decimals=2, group="Advanced"),
    ] = 0.0
    mixer_freq: Annotated[
        float | None,
        FieldMeta(label="Mixer freq", unit="MHz", decimals=3, group="Advanced"),
    ] = None
    pre_delay: Annotated[
        float | QickParam,
        FieldMeta(label="Pre-delay", unit="us", decimals=3, group="Advanced"),
    ] = 0.0
    post_delay: Annotated[
        float | QickParam,
        FieldMeta(label="Post-delay", unit="us", decimals=3, group="Advanced"),
    ] = 0.0
    ro_ch: Annotated[int, FieldMeta(label="RO ch")]
    ro_length: Annotated[
        float | QickParam, FieldMeta(label="RO length", unit="us", decimals=3)
    ]
    trig_offset: Annotated[
        float | QickParam, FieldMeta(label="Trig offset", unit="us", decimals=3)
    ] = 0.0


class PulseReadoutSpec(_PulseReadoutFields[str | WaveformSpec]):
    """Persistent readout; waveform is a concrete key or inline WaveformSpec."""


class PulseReadoutCfg(_PulseReadoutFields[WaveformCfg], AbsReadoutCfg):
    """Materialized readout; waveform is an expanded, validated WaveformCfg."""

    def set_param(self, name: str, value: float | QickParam) -> None:
        """Set freq (MHz), gain, length or ro_length (us); otherwise raise ValueError.

        freq supplies both generator and ADC. Waveform length restrictions apply.
        """
        if name == "length":
            self.waveform.set_param(name, value)
        elif name in {"freq", "gain", "ro_length"}:
            setattr(self, name, value)
        else:
            raise ValueError(f"Unknown parameter: {name}")


class MuxReadoutTone(ConfigBase):
    """One ordered, numeric DAC tone and its explicit logical ADC channel.

    freq is finite absolute MHz (negative values are permitted), gain is finite
    normalized amplitude in [-1, 1], phase is any finite degree value, and ro_ch
    is a nonnegative integer, never bool. Equal frequencies are permitted. Tone
    order, not frequency or channel order, defines identity. label is optional
    display text, defaults to the empty string, and may repeat; it is not a
    component identity and program execution does not interpret it. No rescaling
    occurs; the author owns summed-signal headroom.
    """

    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    freq: float
    gain: Annotated[float, Field(ge=-1.0, le=1.0)]
    phase: float = 0.0
    ro_ch: Annotated[int, Field(ge=0)]
    label: str = ""


class _MuxReadoutFields(_ModuleFields):
    """Numeric readout/mux settings shared by persistent Spec and runtime cfg.

    gen_ch is the explicit nonnegative integer generator channel (not bool).
    nqz must be integer 1 or 2, and mixer_freq is required finite MHz. tones is
    an ordered list of 1..8 MuxReadoutTone values, each carrying its own ro_ch.
    length is the common positive const-playback duration in us; ro_length is
    the common positive ADC window in us. pre_delay/post_delay bracket playback,
    and trig_offset delays the common ADC trigger, all finite nonnegative us.
    desc is optional prose; tone labels are display-only strings. Execution
    fields accept only numeric scalars, not references, expressions, waveforms,
    QICK sweeps or register words.
    """

    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    type: Literal["readout/mux"] = "readout/mux"
    gen_ch: Annotated[int, Field(ge=0)]
    nqz: Literal[1, 2]
    mixer_freq: float
    tones: Annotated[list[MuxReadoutTone], Field(min_length=1, max_length=8)]
    length: Annotated[float, Field(gt=0)]
    ro_length: Annotated[float, Field(gt=0)]
    pre_delay: Annotated[float, Field(ge=0)] = 0.0
    post_delay: Annotated[float, Field(ge=0)] = 0.0
    trig_offset: Annotated[float, Field(ge=0)] = 0.0

    @field_validator("nqz", mode="before")
    @classmethod
    def _integer_nqz(cls, value: object) -> object:
        # Literal validation alone accepts True and 1.0 as zone 1.
        if type(value) is not int:
            raise ValueError("nqz must be integer 1 or 2, not bool or float")
        return value


class MuxReadoutSpec(_MuxReadoutFields):
    """Persistent numeric mux module; only the entire module may use a library key."""


class MuxReadoutCfg(_MuxReadoutFields, AbsReadoutCfg):
    """Materialized numeric mux module with no assets, resolver or runtime objects."""

    def set_param(self, name: str, value: float | QickParam) -> None:
        """Reject every knob with ValueError; rebuild numeric cfg and program instead."""
        raise ValueError(
            f"MuxReadout does not support set_param({name!r}, {value!r}); "
            "rebuild numeric cfg and program instead"
        )


class AbsResetCfg(AbsModuleCfg):
    """Materialized reset family, with zero, one, two, or bath pulse content."""


class NoneResetCfg(AbsResetCfg):
    """Empty reset; type is reset/none and no sweep knobs are supported."""

    type: Literal["reset/none"] = "reset/none"

    def set_param(self, name: str, value: float | QickParam) -> None:
        """Raise ValueError for every name; this reset has no parameters."""
        raise ValueError(f"NoneReset does not support set_param: {name}={value}")


class _PulseResetFields(_ModuleFields):
    """One-pulse reset family; type is reset/pulse."""

    type: Literal["reset/pulse"] = "reset/pulse"


class PulseResetSpec(_PulseResetFields):
    """Persistent one-pulse reset; pulse_cfg is a concrete key or inline PulseSpec."""

    pulse_cfg: Annotated[str | PulseSpec, FieldMeta(label="Pulse")]


class PulseResetCfg(_PulseResetFields, AbsResetCfg):
    """Materialized one-pulse reset; pulse_cfg is an expanded PulseCfg."""

    pulse_cfg: PulseCfg

    def set_param(self, name: str, value: float | QickParam) -> None:
        """Delegate the knob to the pulse, including its units and ValueError failures."""
        self.pulse_cfg.set_param(name, value)


class _TwoPulseResetFields(_ModuleFields):
    """Two-pulse reset family; type is reset/two_pulse."""

    type: Literal["reset/two_pulse"] = "reset/two_pulse"


class TwoPulseResetSpec(_TwoPulseResetFields):
    """Persistent two-pulse reset; pulse1_cfg/pulse2_cfg are keys or inline PulseSpecs."""

    pulse1_cfg: Annotated[str | PulseSpec, FieldMeta(label="Pulse 1")]
    pulse2_cfg: Annotated[str | PulseSpec, FieldMeta(label="Pulse 2")]


class TwoPulseResetCfg(_TwoPulseResetFields, AbsResetCfg):
    """Materialized two-pulse reset; pulse1_cfg/pulse2_cfg are expanded PulseCfgs."""

    pulse1_cfg: PulseCfg
    pulse2_cfg: PulseCfg

    def set_param(self, name: str, value: float | QickParam) -> None:
        """Set gain1/freq1, gain2/freq2, or length on both pulses; otherwise raise ValueError."""
        if name in ["gain1", "freq1"]:
            self.pulse1_cfg.set_param(name.replace("1", ""), value)
        elif name in ["gain2", "freq2"]:
            self.pulse2_cfg.set_param(name.replace("2", ""), value)
        elif name == "length":
            self.pulse1_cfg.set_param("length", value)
            self.pulse2_cfg.set_param("length", value)
        else:
            raise ValueError(f"Unknown parameter: {name}")


class _BathResetFields(_ModuleFields):
    """Bath reset family; type is reset/bath."""

    type: Literal["reset/bath"] = "reset/bath"


class BathResetSpec(_BathResetFields):
    """Persistent bath reset; cavity_tone_cfg/qubit_tone_cfg/pi2_cfg are keys or PulseSpecs."""

    cavity_tone_cfg: Annotated[str | PulseSpec, FieldMeta(label="Cavity Tone")]
    qubit_tone_cfg: Annotated[str | PulseSpec, FieldMeta(label="Qubit Tone")]
    pi2_cfg: Annotated[str | PulseSpec, FieldMeta(label="Pi/2 Pulse")]


class BathResetCfg(_BathResetFields, AbsResetCfg):
    """Materialized bath reset; all three children are expanded PulseCfgs."""

    cavity_tone_cfg: PulseCfg
    qubit_tone_cfg: PulseCfg
    pi2_cfg: PulseCfg

    def set_param(self, name: str, value: float | QickParam) -> None:
        """Set qub_gain/freq/length, res_gain/freq/length, or pi2_phase.

        Qubit/cavity knobs delegate to the corresponding tone; pi2_phase sets the
        third pulse's phase in degrees. Unsupported names/values raise ValueError.
        """
        if name in ["qub_gain", "qub_freq"]:
            self.qubit_tone_cfg.set_param(name.replace("qub_", ""), value)
        elif name in ["res_gain", "res_freq"]:
            self.cavity_tone_cfg.set_param(name.replace("res_", ""), value)
        elif name == "qub_length":
            self.qubit_tone_cfg.set_param("length", value)
        elif name == "res_length":
            self.cavity_tone_cfg.set_param("length", value)
        elif name == "pi2_phase":
            self.pi2_cfg.set_param("phase", value)
        else:
            raise ValueError(f"Unknown parameter: {name}")


type ReadoutSpec = Annotated[
    PulseReadoutSpec | MuxReadoutSpec, Field(discriminator="type")
]
type ReadoutCfg = Annotated[
    PulseReadoutCfg | MuxReadoutCfg, Field(discriminator="type")
]
type ResetSpec = Annotated[
    NoneResetCfg | PulseResetSpec | TwoPulseResetSpec | BathResetSpec,
    Field(discriminator="type"),
]
type ResetCfg = Annotated[
    NoneResetCfg | PulseResetCfg | TwoPulseResetCfg | BathResetCfg,
    Field(discriminator="type"),
]
type ModuleSpec = Annotated[
    PulseSpec | ReadoutSpec | ResetSpec, Field(discriminator="type")
]
type ModuleCfg = Annotated[
    PulseCfg | ReadoutCfg | ResetCfg, Field(discriminator="type")
]
