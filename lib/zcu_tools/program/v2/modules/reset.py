from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Annotated,
    Literal,
    TypeAlias,
    TypedDict,
    Unpack,
    overload,
)

from pydantic import BeforeValidator, Field
from qick.asm_v2 import QickParam

from .base import AbsModuleCfg, Module, apply_cfg_overrides, resolve_module_ref
from .pulse import Pulse, PulseCfg, PulseOverrides
from .util import calc_max_length

if TYPE_CHECKING:
    from zcu_tools.program.v2.timeline import TimelineProgramV2


PulseOrRef: TypeAlias = Annotated[PulseCfg, BeforeValidator(resolve_module_ref)]


class AbsResetCfg(AbsModuleCfg):
    @abstractmethod
    def build(self, name: str) -> AbsReset: ...


class PulseResetCfg(AbsResetCfg):
    type: Literal["reset/pulse"] = "reset/pulse"
    pulse_cfg: PulseOrRef

    def build(self, name: str) -> PulseReset:
        return PulseReset(name, self)

    def set_param(self, name: str, value: float | QickParam) -> None:
        self.pulse_cfg.set_param(name, value)


class TwoPulseResetCfg(AbsResetCfg):
    type: Literal["reset/two_pulse"] = "reset/two_pulse"
    pulse1_cfg: PulseOrRef
    pulse2_cfg: PulseOrRef

    def build(self, name: str) -> TwoPulseReset:
        return TwoPulseReset(name, self)

    def set_param(self, name: str, value: float | QickParam) -> None:
        if name in ["gain1", "freq1"]:
            self.pulse1_cfg.set_param(name.replace("1", ""), value)
        elif name in ["gain2", "freq2"]:
            self.pulse2_cfg.set_param(name.replace("2", ""), value)
        elif name == "length":
            self.pulse1_cfg.set_param("length", value)
            self.pulse2_cfg.set_param("length", value)
        else:
            raise ValueError(f"Unknown parameter: {name}")


class BathResetCfg(AbsResetCfg):
    type: Literal["reset/bath"] = "reset/bath"
    cavity_tone_cfg: PulseOrRef
    qubit_tone_cfg: PulseOrRef
    pi2_cfg: PulseOrRef

    def build(self, name: str) -> BathReset:
        return BathReset(name, self)

    def set_param(self, name: str, value: float | QickParam) -> None:
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


ResetCfg: TypeAlias = Annotated[
    PulseResetCfg | TwoPulseResetCfg | BathResetCfg,
    Field(discriminator="type"),
]


class AbsReset(Module):
    """Base declaration for reset modules emitted on a TimelineProgramV2."""



class PulseResetOverrides(PulseOverrides, total=False):
    """Optional pulse-reset knobs; None keeps each cfg value.

    freq sets the pulse carrier in MHz, gain the normalized amplitude,
    phase the pulse phase in degrees, and length the waveform duration in us.
    Every field accepts a float scalar, QickParam sweep or None.
    Waveform length sweep restrictions remain unchanged.
    """


class TwoPulseResetOverrides(TypedDict, total=False):
    """Optional two-pulse knobs, with None keeping each cfg value.

    gain1/gain2 are normalized amplitudes of pulse1/pulse2. freq1/freq2 are
    their carriers in MHz. length sets both waveform durations in us.
    All accept scalars or QICK sweeps, subject to waveform length restrictions.
    """

    gain1: float | QickParam | None
    gain2: float | QickParam | None
    freq1: float | QickParam | None
    freq2: float | QickParam | None
    length: float | QickParam | None


class BathResetOverrides(TypedDict, total=False):
    """Optional bath-reset knobs; every field accepts a scalar, sweep or None.

    qub_gain/qub_freq/qub_length set qubit-tone normalized amplitude, carrier
    MHz and waveform duration us. res_gain/res_freq/res_length set the same
    quantities for the cavity tone. pi2_phase sets the final pi/2 phase in
    degrees. None keeps cfg. Waveform length sweep restrictions still apply.
    """

    qub_gain: float | QickParam | None
    qub_freq: float | QickParam | None
    qub_length: float | QickParam | None
    res_gain: float | QickParam | None
    res_freq: float | QickParam | None
    res_length: float | QickParam | None
    pi2_phase: float | QickParam | None


class ResetOverrides(PulseResetOverrides, BathResetOverrides, total=False):
    """Factory keyword union; only the selected cfg variant's keys are valid.

    freq sets the pulse-reset carrier in MHz, gain its normalized amplitude,
    and phase its phase in degrees. length sets waveform duration in us for
    pulse reset, or both waveform durations for two-pulse reset.
    gain1/gain2 set normalized pulse1/pulse2 amplitudes; freq1/freq2 set
    their carriers in MHz. qub_gain/qub_freq/qub_length set qubit-tone
    normalized amplitude, carrier MHz and waveform duration us.
    res_gain/res_freq/res_length set those quantities for the cavity tone.
    pi2_phase sets the final pi/2 phase in degrees. Every field accepts a
    float scalar, QickParam sweep or None to keep cfg. Waveform length
    sweep restrictions remain unchanged.
    """

    gain1: float | QickParam | None
    gain2: float | QickParam | None
    freq1: float | QickParam | None
    freq2: float | QickParam | None


class Reset:
    """Dispatch cfg to its reset module; zcu_lab uses ``Reset.Cfg`` fields.

    Calling Reset(name, cfg) returns cfg.build(name), not a Reset instance.
    name labels the module. None means no reset and returns None.
    """

    Cfg: TypeAlias = ResetCfg

    @overload
    def __new__(
        cls, name: str, cfg: AbsResetCfg, **overrides: Unpack[ResetOverrides]
    ) -> AbsReset: ...

    @overload
    def __new__(
        cls, name: str, cfg: None, **overrides: Unpack[ResetOverrides]
    ) -> None: ...

    def __new__(
        cls, name: str, cfg: AbsResetCfg | None, **overrides: Unpack[ResetOverrides]
    ) -> AbsReset | None:
        """Build the reset with keyword-only overrides on a detached cfg.

        Pulse reset accepts freq/gain/phase/length. Two-pulse reset accepts
        gain1/gain2/freq1/freq2/length. Bath reset accepts qub_gain/qub_freq/
        qub_length/res_gain/res_freq/res_length/pi2_phase. Frequencies are MHz,
        lengths us, phases degrees and gains normalized; values may be sweeps.
        None keeps each cfg value. Unknown or wrong-variant keys and nonnumeric
        values raise TypeError. Unsupported waveform sweeps raise ValueError.
        cfg=None returns None without reset.
        """
        unknown = overrides.keys() - ResetOverrides.__annotations__.keys()
        if unknown:
            raise TypeError(f"Unknown override keywords: {sorted(unknown)}")
        if cfg is None:
            return None
        if isinstance(cfg, PulseResetCfg):
            allowed = PulseResetOverrides.__annotations__
        elif isinstance(cfg, TwoPulseResetCfg):
            allowed = TwoPulseResetOverrides.__annotations__
        elif isinstance(cfg, BathResetCfg):
            allowed = BathResetOverrides.__annotations__
        else:
            allowed = {}
        effective = deepcopy(cfg)
        apply_cfg_overrides(effective, overrides, allowed)
        return effective.build(name)


class PulseReset(AbsReset):
    """Build this reset variant with settings declared as ``PulseReset.Cfg``."""

    Cfg: TypeAlias = PulseResetCfg

    def __init__(
        self,
        name: str,
        cfg: PulseResetCfg,
        **overrides: Unpack[PulseResetOverrides],
    ) -> None:
        """Capture named reset settings without changing the caller's cfg.

        Override freq (MHz), gain (normalized), phase (degrees), length (us).
        Unknown keys or nonnumeric overrides raise TypeError.
        Values accept scalars or QICK sweeps; None keeps cfg. Existing set_param
        waveform restrictions apply and unsupported length sweeps raise ValueError.
        """
        self.name = name
        self.cfg = deepcopy(cfg)
        apply_cfg_overrides(self.cfg, overrides, PulseResetOverrides.__annotations__)
        self.reset_pulse = Pulse(name=f"{name}_pulse", cfg=self.cfg.pulse_cfg)




    def declare(self, prog: TimelineProgramV2) -> None:
        """Declare the reset pulse's resources on prog."""
        self.reset_pulse.declare(prog)

    def lower(
        self, prog: TimelineProgramV2, *, timeline: str | None, tag: str | None
    ) -> None:
        """Play the reset pulse and ceil its actual end plus its post-delay."""
        self.reset_pulse.lower(prog, timeline=timeline, tag=tag)


class TwoPulseReset(AbsReset):
    """Build this reset variant with settings declared as ``TwoPulseReset.Cfg``."""

    Cfg: TypeAlias = TwoPulseResetCfg

    def __init__(
        self,
        name: str,
        cfg: TwoPulseResetCfg,
        **overrides: Unpack[TwoPulseResetOverrides],
    ) -> None:
        """Capture named reset settings without changing the caller's cfg.

        Override gain1/gain2 (normalized), freq1/freq2 (MHz), shared length (us).
        Unknown keys or nonnumeric overrides raise TypeError.
        Values accept scalars or QICK sweeps; None keeps cfg. Existing set_param
        waveform restrictions apply and unsupported length sweeps raise ValueError.
        """
        self.name = name
        self.cfg = deepcopy(cfg)
        apply_cfg_overrides(self.cfg, overrides, TwoPulseResetOverrides.__annotations__)
        self.reset_pulse1 = Pulse(name=f"{name}_pulse1", cfg=self.cfg.pulse1_cfg)
        self.reset_pulse2 = Pulse(name=f"{name}_pulse2", cfg=self.cfg.pulse2_cfg)




    def declare(self, prog: TimelineProgramV2) -> None:
        """Declare both reset pulses on prog."""
        self.reset_pulse1.declare(prog)
        self.reset_pulse2.declare(prog)

    def lower(
        self, prog: TimelineProgramV2, *, timeline: str | None, tag: str | None
    ) -> None:
        """Start both pulses together and merge actual ends plus each child's post."""
        timeline = timeline if timeline is not None else "default"
        first = prog.emission_name("reset_first")
        second = prog.emission_name("reset_second")
        with prog.reference_guard(timeline):
            prog.fork(timeline, into=[first, second])
            self.reset_pulse1.lower(prog, timeline=first, tag=tag)
            self.reset_pulse2.lower(prog, timeline=second, tag=None)
            prog.join(first, second, into=timeline)


class BathReset(AbsReset):
    """Build this reset variant with settings declared as ``BathReset.Cfg``."""

    Cfg: TypeAlias = BathResetCfg

    def __init__(
        self,
        name: str,
        cfg: BathResetCfg,
        **overrides: Unpack[BathResetOverrides],
    ) -> None:
        """Capture named reset settings without changing the caller's cfg.

        Override qub/res gain (normalized), freq (MHz), length (us), pi2_phase (degrees).
        Unknown keys or nonnumeric overrides raise TypeError.
        Values accept scalars or QICK sweeps; None keeps cfg. Existing set_param
        waveform restrictions apply and unsupported length sweeps raise ValueError.
        """
        self.name = name
        self.cfg = deepcopy(cfg)
        apply_cfg_overrides(self.cfg, overrides, BathResetOverrides.__annotations__)
        self.res_pulse = Pulse(name=f"{name}_res_pulse", cfg=self.cfg.cavity_tone_cfg)
        self.qub_pulse = Pulse(name=f"{name}_qub_pulse", cfg=self.cfg.qubit_tone_cfg)
        self.pi2_pulse = Pulse(name=f"{name}_pi2_pulse", cfg=self.cfg.pi2_cfg)




    def declare(self, prog: TimelineProgramV2) -> None:
        """Declare the cavity, qubit and final pi/2 pulses on prog."""
        self.res_pulse.declare(prog)
        self.qub_pulse.declare(prog)
        self.pi2_pulse.declare(prog)

    def lower(
        self, prog: TimelineProgramV2, *, timeline: str | None, tag: str | None
    ) -> None:
        """Overlap cavity/qubit, then place pi/2 after the ceiled cavity end+post.

        Merge the qubit and pi/2 ends including their individual post-delays.
        tag identifies the cavity start.
        """
        timeline = timeline if timeline is not None else "default"
        cavity = prog.emission_name("reset_cavity")
        qubit = prog.emission_name("reset_qubit")
        with prog.reference_guard(timeline):
            prog.fork(timeline, into=[cavity, qubit])
            self.res_pulse.lower(prog, timeline=cavity, tag=tag)
            self.qub_pulse.lower(prog, timeline=qubit, tag=None)
            self.pi2_pulse.lower(prog, timeline=cavity, tag=None)
            prog.join(cavity, qubit, into=timeline)
