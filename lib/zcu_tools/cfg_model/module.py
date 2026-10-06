"""Persistent module Specs and materialized Cfgs, without program execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Annotated, Literal

from pydantic import Field, model_validator
from qick.asm_v2 import QickParam

from .base import ConfigBase
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


class _PulseFields(_ModuleFields):
    """Pulse scalar knobs, shared by persistent and materialized schemas.

    type is pulse. ch is the explicit generator channel; nqz is Nyquist zone 1
    or 2. freq/mixer_freq/mux_freqs are MHz, phase/mux_phases are degrees,
    gain/mux_gains are normalized gains, pre_delay/post_delay are µs.
    freq, phase, gain and delays accept QICK sweeps. Optional mixer_freq and mux
    arrays configure mixing/multiplexing; None leaves the option unspecified.
    mask selects mux tones by index. outsel is the optional hardware output
    selector. ro_ch is the optional readout channel for generator configuration.
    """

    type: Literal["pulse"] = "pulse"
    ch: int
    nqz: Literal[1, 2]
    freq: float | QickParam
    phase: float | QickParam = 0.0
    gain: float | QickParam
    pre_delay: float | QickParam = 0.0
    post_delay: float | QickParam = 0.0
    mixer_freq: float | None = None
    mux_freqs: list[float] | None = None
    mux_gains: list[float] | None = None
    mux_phases: list[float] | None = None
    mask: list[int] | None = None
    outsel: int | None = None
    ro_ch: int | None = None


class PulseSpec(_PulseFields):
    """Persistent pulse; waveform is a concrete library key or inline WaveformSpec."""

    waveform: str | WaveformSpec


class PulseCfg(_PulseFields, AbsModuleCfg):
    """Materialized pulse; waveform is an expanded, validated WaveformCfg."""

    waveform: WaveformCfg

    def set_param(self, name: str, value: float | QickParam) -> None:
        """Set gain/freq/phase or delegate length to the waveform; otherwise raise ValueError."""
        if name == "length":
            self.waveform.set_param(name, value)
        elif name in {"gain", "freq", "phase"}:
            setattr(self, name, value)
        else:
            raise ValueError(f"Unknown parameter: {name}")


class AbsReadoutCfg(AbsModuleCfg):
    """Materialized readout family with direct or pulse-driven acquisition."""


class DirectReadoutCfg(AbsReadoutCfg):
    """Direct readout: ro_ch is the ADC channel, gen_ch an optional associated generator.

    ro_length/trig_offset are µs and ro_freq is MHz; all accept QICK sweeps.
    type is readout/direct.
    """

    type: Literal["readout/direct"] = "readout/direct"
    ro_ch: int
    ro_length: float | QickParam
    ro_freq: float | QickParam
    trig_offset: float | QickParam = 0.0
    gen_ch: int | None = None

    def set_param(self, name: str, value: float | QickParam) -> None:
        """Set ro_freq (MHz) or ro_length (µs); other names raise ValueError."""
        if name == "ro_freq":
            self.ro_freq = value
        elif name == "ro_length":
            self.ro_length = value
        else:
            raise ValueError(f"Unknown parameter: {name}")


class _PulseReadoutFields(_ModuleFields):
    """Pulse readout scalar schema; ro_cfg is the associated direct ADC configuration.

    type is readout/pulse. Inline pulse mappings supply missing gen_ch/ro_freq
    from ch/freq. Explicit ro_cfg values win. String pulse references are resolved
    by materialization before this derivation; this schema does not look up keys.
    """

    type: Literal["readout/pulse"] = "readout/pulse"
    ro_cfg: DirectReadoutCfg

    @model_validator(mode="before")
    @classmethod
    def _autoderive_ro_fields(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        result = deepcopy(data)
        pulse = result.get("pulse_cfg")
        ro = result.get("ro_cfg")
        if isinstance(pulse, dict) and isinstance(ro, dict):
            if "ch" in pulse:
                ro.setdefault("gen_ch", pulse["ch"])
            if "freq" in pulse:
                ro.setdefault("ro_freq", pulse["freq"])
        return result


class PulseReadoutSpec(_PulseReadoutFields):
    """Persistent pulse readout; pulse_cfg is a concrete key or inline PulseSpec."""

    pulse_cfg: str | PulseSpec


class PulseReadoutCfg(_PulseReadoutFields, AbsReadoutCfg):
    """Materialized pulse readout; pulse_cfg is an expanded PulseCfg."""

    pulse_cfg: PulseCfg

    def set_param(self, name: str, value: float | QickParam) -> None:
        """Set gain/freq/length on the pulse or ro_freq/ro_length on ADC configuration.

        freq also updates ro_freq. Unsupported names/values raise ValueError.
        """
        if name == "gain":
            self.pulse_cfg.set_param("gain", value)
        elif name == "freq":
            self.pulse_cfg.set_param("freq", value)
            self.ro_cfg.set_param("ro_freq", value)
        elif name == "ro_freq":
            self.ro_cfg.set_param("ro_freq", value)
        elif name == "length":
            self.pulse_cfg.set_param("length", value)
        elif name == "ro_length":
            self.ro_cfg.set_param("ro_length", value)
        else:
            raise ValueError(f"Unknown parameter: {name}")


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

    pulse_cfg: str | PulseSpec


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

    pulse1_cfg: str | PulseSpec
    pulse2_cfg: str | PulseSpec


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

    cavity_tone_cfg: str | PulseSpec
    qubit_tone_cfg: str | PulseSpec
    pi2_cfg: str | PulseSpec


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
    DirectReadoutCfg | PulseReadoutSpec, Field(discriminator="type")
]
type ReadoutCfg = Annotated[
    DirectReadoutCfg | PulseReadoutCfg, Field(discriminator="type")
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
