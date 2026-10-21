"""Detached author readout values and the sole FTF YAML-to-readout adapter."""

from dataclasses import dataclass

import numpy as np

from zcu_tools.resources.ftf import FTFSettings
from zcu_tools.resources.ftf.schema import Receiver


@dataclass(frozen=True)
class FTFReceiverConfig:
    """One bound receiver, independent of the persisted settings shape.

    ro_ch: nonnegative integer logical ADC channel with no fixed upper bound,
        not its output position.
    gain: signed field-to-ADC amplitude gain; phase_deg: additive IQ phase.
    noise_psd_adc2_ns: symmetric positive-semidefinite two-sided noise PSD.
    offset_adc: per-sample I/Q offset. time_of_flight_ns: nonnegative line delay.
    All numeric values must be finite. Invalid calibration raises ValueError.
    """

    ro_ch: int
    gain: float
    noise_psd_adc2_ns: tuple[tuple[float, float], tuple[float, float]]
    phase_deg: float = 0.0
    offset_adc: tuple[float, float] = (0.0, 0.0)
    time_of_flight_ns: float = 0.0

    def __post_init__(self) -> None:
        if type(self.ro_ch) is not int or self.ro_ch < 0:
            raise ValueError("receiver ro_ch must be a nonnegative integer")
        if not np.isfinite([self.gain, self.phase_deg, self.time_of_flight_ns]).all():
            raise ValueError("receiver calibration must be finite")
        if self.time_of_flight_ns < 0:
            raise ValueError("receiver time_of_flight_ns must be nonnegative")
        offset = np.asarray(self.offset_adc)
        noise = np.asarray(self.noise_psd_adc2_ns)
        if offset.shape != (2,) or not np.isfinite(offset).all():
            raise ValueError("receiver offset_adc must be a finite I/Q pair")
        if noise.shape != (2, 2) or not np.isfinite(noise).all():
            raise ValueError("receiver noise PSD must be a finite 2x2 matrix")
        roundoff = 8 * np.finfo(float).eps * float(np.max(np.abs(noise)))
        if (
            not np.array_equal(noise, noise.T)
            or np.linalg.eigvalsh(noise).min() < -roundoff
        ):
            raise ValueError(
                "receiver noise PSD must be symmetric positive semidefinite"
            )


@dataclass(frozen=True)
class FTFMuxReadoutConfig:
    """Detached mux source and bound receivers; invalid values raise ValueError.

    gen_ch: nonnegative integer generator index, with no fixed upper bound.
    input_per_gain: signed finite incident sqrt(photons/ns) per program gain.
    receivers: 1..8 uniquely bound ADCs; programs may select a subset. Program
    tone order determines output order, not this tuple. Both readout paths share
    the physical resonator pair; this value does not map tones to resonators.
    """

    gen_ch: int
    input_per_gain: float
    receivers: tuple[FTFReceiverConfig, ...]

    def __post_init__(self) -> None:
        if type(self.gen_ch) is not int or self.gen_ch < 0:
            raise ValueError("mux gen_ch must be a nonnegative integer")
        if not np.isfinite(self.input_per_gain):
            raise ValueError("mux input_per_gain must be finite")
        if not 1 <= len(self.receivers) <= 8:
            raise ValueError("mux requires 1..8 receivers")
        if len({r.ro_ch for r in self.receivers}) != len(self.receivers):
            raise ValueError("duplicate mux receiver channel")


@dataclass(frozen=True)
class FTFReadoutConfig:
    """Detached pulse receiver plus an optional independent mux capability.

    pulse: receiver for the single-carrier pulse path; its generator and incident
    calibration remain in settings.readout.drive. mux: optional mux source and
    receivers; None disables mux without disabling pulse. Receiver channels must
    be disjoint across paths, otherwise construction raises ValueError.
    """

    pulse: FTFReceiverConfig
    mux: FTFMuxReadoutConfig | None = None

    def __post_init__(self) -> None:
        if self.mux is not None and any(
            receiver.ro_ch == self.pulse.ro_ch for receiver in self.mux.receivers
        ):
            raise ValueError("duplicate readout receiver channel across pulse and mux")


def _receiver_configuration(adc: Receiver) -> FTFReceiverConfig:
    return FTFReceiverConfig(
        adc.ro_ch,
        adc.gain,
        adc.noise_psd_adc2_ns,
        adc.phase_deg,
        adc.offset_adc,
        adc.time_of_flight_ns,
    )


def readout_configuration(settings: FTFSettings) -> FTFReadoutConfig:
    """Adapt validated FTF settings into independent pulse and mux capabilities.

    Preserve every receiver calibration and mux generator/incident conversion.
    Omitted mux yields mux=None. This is the sole settings-shape adapter below
    resource parsing; explicit author configurations use the same detached values.
    """
    mux = settings.readout.mux
    return FTFReadoutConfig(
        pulse=_receiver_configuration(settings.readout.adc),
        mux=None
        if mux is None
        else FTFMuxReadoutConfig(
            gen_ch=mux.gen_ch,
            input_per_gain=mux.input_per_gain,
            receivers=tuple(_receiver_configuration(adc) for adc in mux.adc),
        ),
    )
