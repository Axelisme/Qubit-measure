"""Generation and acceptance knobs shared by Coherence and MIST calibration."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator
from zcu_tools.cfg_model import Gain

from zcu_lab.workflows.common.frequency_tracking import FrequencyTunables


class CalibrationTunables(FrequencyTunables):
    """Generation and acceptance knobs captured once per flux-point invocation.

    qf_widen specifies the finite sequence of spectroscopy spans, not retries
    after adapter errors. Error and residual upper bounds are strict; readout
    SNR uses an inclusive lower bound. ro_every counts flux points.
    drive_gain_mode controls spectroscopy; rabi_drive_gain_mode controls Rabi.
    Acceptance/cadence/range defaults are provisional simulator settings, not
    hardware-validated physics defaults; users confirm them at CP-B/CP-C.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    qf_reps: int = Field(default=1000, ge=1, description="Spectroscopy averages")
    qf_rounds: int = Field(default=10, ge=1, description="Spectroscopy rounds")
    qf_acquire_retry: int = Field(
        default=3, ge=0, description="Retries after a transient acquisition failure"
    )
    qf_earlystop_snr: float | None = Field(
        default=50.0,
        gt=0,
        description="Completed-round trace SNR threshold; None disables early stop",
    )
    rabi_reps: int = Field(default=1000, ge=1, description="Rabi averages")
    rabi_rounds: int = Field(default=10, ge=1, description="Rabi rounds")
    rabi_acquire_retry: int = Field(
        default=3, ge=0, description="Retries after a transient acquisition failure"
    )
    rabi_earlystop_snr: float | None = Field(
        default=30.0,
        gt=0,
        description="Completed-round trace SNR threshold; None disables early stop",
    )
    ro_reps: int = Field(default=100, ge=1, description="Readout averages")
    ro_rounds: int = Field(default=1, ge=1, description="Readout rounds")
    ro_acquire_retry: int = Field(
        default=3, ge=0, description="Retries after a transient acquisition failure"
    )
    max_frequency_rel_err: float = Field(
        default=0.3, gt=0, le=1, description="Strict center-error/FWHM upper bound"
    )
    max_rabi_rel_err: float = Field(
        default=0.3,
        gt=0,
        le=1,
        description="Strict pi-error/pi-coordinate upper bound (duration or gain)",
    )
    rabi_max_mean_residual_ratio: float = Field(
        default=0.1,
        gt=0,
        description="Strict mean absolute Rabi residual/fit-range upper bound",
    )
    max_fit_residual: float = Field(
        default=0.2,
        gt=0,
        le=1,
        description="Strict normalized residual RMS upper bound",
    )
    min_readout_snr: float = Field(
        default=5.0, gt=0, description="Inclusive minimum readout SNR"
    )
    ro_every: int = Field(
        default=5, ge=1, description="Flux-point readout scan cadence"
    )
    rabi_start_factor: float = Field(
        default=0.1, gt=0, description="Sweep start divided by prior pi duration"
    )
    rabi_stop_factor: float = Field(
        default=3.0, gt=0, description="Sweep stop divided by prior pi duration"
    )

    rabi_mode: Literal["len_then_amp", "amp_only", "len_only"] = Field(
        default="len_then_amp",
        description="Length then gain, gain only, or length only",
    )
    amp_gain_max: Gain = Field(
        default=1.0, gt=0, description="Amp-only gain sweep upper limit"
    )
    amp_gain_half_width: Gain = Field(
        default=0.2, gt=0, description="Amp-only tracking window half width"
    )
    amp_min_gain: Gain = Field(
        default=1e-4, gt=0, description="Minimum trusted pi and pi/2 gain"
    )
    amp_max_window_fraction: float = Field(
        default=0.9,
        gt=0,
        le=1,
        description="Pi gain must not exceed this fraction of sweep stop",
    )

    rabi_drive_gain_mode: Literal["auto_pi_product", "fixed"] = Field(
        default="auto_pi_product",
        description="Predict/control gain or retain template gain",
    )
    target_pi_length_us: float | None = Field(
        default=None,
        gt=0,
        description="Pi setpoint in us; None uses min(seed pi, Rabi stop floor / stop factor)",
    )
    rabi_min_start_us: float = Field(
        default=0.05, ge=0, description="Minimum Rabi sweep start in us"
    )
    rabi_min_stop_us: float = Field(
        default=0.5, gt=0, description="Minimum Rabi sweep stop in us"
    )
    rabi_points: int = Field(default=101, ge=3, description="Rabi sweep point count")
    rabi_relax_mode: Literal["auto", "fixed"] = Field(
        default="auto", description="T1-derived or fixed Rabi relaxation"
    )
    rabi_relax_factor: float = Field(
        default=3.0, gt=0, description="Rabi relaxation divided by smoothed T1"
    )
    rabi_relax_min_us: float = Field(
        default=0.0, ge=0, description="Rabi relaxation floor in us"
    )
    rabi_relax_delay_us: float = Field(
        default=30.0, ge=0, description="Fixed Rabi relaxation in us"
    )
    ro_relax_mode: Literal["auto", "fixed"] = Field(
        default="auto", description="T1-derived or fixed readout relaxation"
    )
    ro_relax_factor: float = Field(
        default=3.0, gt=0, description="Readout relaxation divided by smoothed T1"
    )
    ro_relax_min_us: float = Field(
        default=0.0, ge=0, description="Readout relaxation floor in us"
    )
    ro_relax_delay_us: float = Field(
        default=30.0, ge=0, description="Fixed readout relaxation in us"
    )
    ro_freq_half_width_mhz: float = Field(
        default=1.0, gt=0, description="Readout frequency search half width in MHz"
    )
    ro_gain_half_width: Gain = Field(
        default=0.1, gt=0, description="Readout gain search half width"
    )
    ro_skew_penalty: float = Field(
        default=0.0, ge=0, description="Readout IQ skewness score penalty"
    )

    def resolve_target_pi_length_us(self, seed_pi_length_us: float) -> float:
        """Return the explicit target, or min(host seed, stop floor / stop factor).

        seed_pi_length_us is the validated positive host pi duration in us.
        The fallback keeps the requested length window wide enough to resolve
        several oscillations, including when Amp-only skips length calibration.
        """
        if self.target_pi_length_us is not None:
            return self.target_pi_length_us
        return min(seed_pi_length_us, self.rabi_min_stop_us / self.rabi_stop_factor)

    @model_validator(mode="after")
    def _validate_rabi_range(self) -> CalibrationTunables:
        if self.rabi_stop_factor <= self.rabi_start_factor:
            raise ValueError("Rabi stop factor must exceed start factor")
        if self.rabi_min_stop_us <= self.rabi_min_start_us:
            raise ValueError("Rabi stop floor must exceed start floor")
        return self
