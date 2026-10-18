"""Detached author readout values and the sole FTF YAML-to-readout adapter."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from zcu_tools.resources.ftf import FTFSettings


@dataclass(frozen=True)
class FTFReceiverConfig:
    """One bound receiver, independent of the persisted settings shape.

    ro_ch: nonnegative logical ADC channel, not its output position.
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
class FTFReadoutConfig:
    """Detached readout capability consumed below FTF settings.

    kind: pulse for one active incident carrier; mux for 1..8 simultaneous tones.
    receivers: nonempty unique bound ADCs. Pulse requires exactly one receiver;
        mux permits a program to use a subset. Output order comes from the
        program, never this tuple. Invalid kind/count/channels raise ValueError.
    """

    kind: Literal["pulse", "mux"]
    receivers: tuple[FTFReceiverConfig, ...]

    def __post_init__(self) -> None:
        if self.kind not in ("pulse", "mux"):
            raise ValueError("readout kind must be pulse or mux")
        limit = 1 if self.kind == "pulse" else 8
        if not 1 <= len(self.receivers) <= limit:
            raise ValueError(f"{self.kind} requires 1..{limit} receivers")
        if len({r.ro_ch for r in self.receivers}) != len(self.receivers):
            raise ValueError("duplicate readout receiver channel")


def readout_configuration(settings: FTFSettings) -> FTFReadoutConfig:
    """Adapt current single-ADC FTF settings without changing values or defaults.

    Mux settings are deliberately not parsed here until their YAML shape is
    decided. Author callers may supply FTFReadoutConfig directly to planning
    and FTFSource without adding persisted fields.
    """
    adc = settings.readout.adc
    return FTFReadoutConfig(
        "pulse",
        (
            FTFReceiverConfig(
                adc.ro_ch,
                adc.gain,
                adc.noise_psd_adc2_ns,
                adc.phase_deg,
                adc.offset_adc,
                adc.time_of_flight_ns,
            ),
        ),
    )
