"""Accepted pulse scalars shared by calibration and downstream experiments."""

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Calibrated:
    """Accepted scalar calibration, also usable as deeply read-only host seed.

    f01_mhz/readout_freq_mhz are positive finite frequencies in MHz.
    pi_length_us is a positive finite pi-pulse duration at drive_gain.
    readout_gain is a finite nonnegative native readout gain.
    drive_gain is the acquisition gain in [0, 1]; None retains the template gain.
    Invalid values raise ValueError at construction.
    """

    f01_mhz: float
    pi_length_us: float
    readout_freq_mhz: float
    readout_gain: float
    drive_gain: float | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        for name, value in (
            ("f01_mhz", self.f01_mhz),
            ("pi_length_us", self.pi_length_us),
            ("readout_freq_mhz", self.readout_freq_mhz),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.readout_gain) or self.readout_gain < 0:
            raise ValueError("readout_gain must be nonnegative and finite")
        if self.drive_gain is not None and (
            not math.isfinite(self.drive_gain) or not 0 <= self.drive_gain <= 1
        ):
            raise ValueError("drive_gain must be finite and in [0, 1]")


@dataclass(frozen=True)
class CalibratedPi2(Calibrated):
    """Successful calibration with a fitted pi/2 duration for Echo and Ramsey.

    The original four Calibrated scalars retain their meanings and validation.
    pi2_length_us is the phase-aware Len fit duration, or the fixed Amp duration,
    in microseconds at the selected waveform. It is finite and at least 0.03 us.
    pi2_drive_gain is the independently fitted gain for AmpRabi; None uses
    drive_gain for Length-Rabi. Neither pi2 scalar is inferred by halving pi.
    Invalid original scalars or pi2 duration raise ValueError at construction.
    """

    pi2_length_us: float
    pi2_drive_gain: float | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.pi2_drive_gain is not None and (
            not math.isfinite(self.pi2_drive_gain) or not 0 <= self.pi2_drive_gain <= 1
        ):
            raise ValueError("pi2_drive_gain must be finite and in [0, 1]")
        if not math.isfinite(self.pi2_length_us) or self.pi2_length_us < 0.03:
            raise ValueError("pi2_length_us must be finite and at least 0.03 us")
