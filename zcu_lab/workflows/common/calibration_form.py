"""common calibration_form helpers for lab workflows."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator
from zcu_tools.gui.form import Display, field_error

from zcu_lab.workflows.common.calibration import CalibrationTunables
from zcu_lab.workflows.common.frequency_form import FrequencyTunablesForm


class CalibrationTunablesForm(FrequencyTunablesForm):
    """User-editable calibration generation, acceptance and cadence knobs.

    Field descriptions carry the ordinary knob meanings. Acceptance/cadence and
    range defaults remain provisional simulator settings for CP-B/CP-C, not
    hardware-validated physics defaults. No hidden fixed value is edited here.
    Inherited drive_gain_mode controls spectroscopy; rabi_drive_gain_mode controls
    the separate Rabi gain feedback.
    """

    qf_reps: Annotated[int, Display(label="Spectroscopy averages")] = Field(
        default=100, ge=1, description="Spectroscopy averages"
    )
    qf_rounds: Annotated[int, Display(label="Spectroscopy rounds")] = Field(
        default=1, ge=1, description="Spectroscopy averaging rounds"
    )
    qf_acquire_retry: int = Field(
        default=3, ge=0, description="Retries after a transient acquisition failure"
    )
    qf_earlystop_snr: float | None = Field(
        default=50.0,
        gt=0,
        description="Completed-round trace SNR threshold; None disables early stop",
    )
    rabi_reps: Annotated[int, Display(label="Rabi averages")] = Field(
        default=100, ge=1, description="Rabi averages"
    )
    rabi_rounds: Annotated[int, Display(label="Rabi rounds")] = Field(
        default=1, ge=1, description="Rabi averaging rounds"
    )
    rabi_acquire_retry: int = Field(
        default=3, ge=0, description="Retries after a transient acquisition failure"
    )
    rabi_earlystop_snr: float | None = Field(
        default=30.0,
        gt=0,
        description="Completed-round trace SNR threshold; None disables early stop",
    )
    ro_reps: Annotated[int, Display(label="Readout averages")] = Field(
        default=100, ge=1, description="Readout averages"
    )
    ro_rounds: Annotated[int, Display(label="Readout rounds")] = Field(
        default=1, ge=1, description="Readout averaging rounds"
    )
    ro_acquire_retry: int = Field(
        default=3, ge=0, description="Retries after a transient acquisition failure"
    )
    span_mhz: Annotated[float, Display(label="Spectroscopy span", unit="MHz")] = Field(
        default=30.0, gt=0, description="Base spectroscopy span"
    )
    qf_widen: Annotated[
        tuple[Annotated[float, Field(gt=0)], ...],
        Display(label="Span multipliers"),
    ] = Field(
        default=(1.0, 2.0, 4.0),
        min_length=1,
        description="Ordered spectroscopy span multipliers",
    )
    max_frequency_rel_err: Annotated[
        float, Display(label="Max frequency relative error")
    ] = Field(
        default=0.3, gt=0, le=1, description="Strict center-error/FWHM upper bound"
    )
    max_rabi_rel_err: Annotated[float, Display(label="Max Rabi relative error")] = (
        Field(
            default=0.3,
            gt=0,
            le=1,
            description="Strict pi-error/pi-duration upper bound",
        )
    )
    rabi_max_mean_residual_ratio: float = Field(
        default=0.1,
        gt=0,
        description="Strict mean absolute Rabi residual/fit-range upper bound",
    )
    max_fit_residual: Annotated[float, Display(label="Max fit residual")] = Field(
        default=0.2,
        gt=0,
        le=1,
        description="Strict normalized residual RMS upper bound",
    )
    min_readout_snr: Annotated[float, Display(label="Min readout SNR")] = Field(
        default=5.0, gt=0, description="Inclusive minimum readout SNR"
    )
    ro_every: Annotated[
        int, Display(label="Readout scan interval", unit="flux points")
    ] = Field(default=5, ge=1, description="Flux-point readout scan cadence")
    rabi_start_factor: Annotated[float, Display(label="Rabi start factor")] = Field(
        default=0.1, gt=0, description="Sweep start divided by prior pi duration"
    )
    rabi_stop_factor: Annotated[float, Display(label="Rabi stop factor")] = Field(
        default=3.0, gt=0, description="Sweep stop divided by prior pi duration"
    )

    rabi_drive_gain_mode: Annotated[
        Literal["auto_pi_product", "fixed"], Display(label="Rabi drive gain mode")
    ] = Field(default="auto_pi_product", description="Rabi drive gain mode")
    target_pi_length_us: Annotated[
        float | None,
        Display(
            label="Target pi duration",
            unit="µs",
            placeholder="min(seed π, stop floor / stop factor); default ≤ 0.166667 µs",
        ),
    ] = Field(
        default=None,
        gt=0,
        description="Target pi duration; blank uses min(seed π, Rabi stop floor / stop factor)",
    )
    rabi_min_start_us: Annotated[
        float, Display(label="Rabi start floor", unit="µs")
    ] = Field(default=0.05, ge=0, description="Rabi start floor")
    rabi_min_stop_us: Annotated[float, Display(label="Rabi stop floor", unit="µs")] = (
        Field(default=0.5, gt=0, description="Rabi stop floor")
    )
    rabi_points: Annotated[int, Display(label="Rabi sweep points")] = Field(
        default=101, ge=3, description="Rabi sweep points"
    )
    rabi_relax_mode: Annotated[
        Literal["auto", "fixed"], Display(label="Rabi relaxation mode")
    ] = Field(default="auto", description="Rabi relaxation mode")
    rabi_relax_factor: Annotated[float, Display(label="Rabi relaxation / T1")] = Field(
        default=3.0, gt=0, description="Rabi relaxation / T1"
    )
    rabi_relax_min_us: Annotated[
        float, Display(label="Rabi relaxation floor", unit="µs")
    ] = Field(default=0.0, ge=0, description="Rabi relaxation floor")
    rabi_relax_delay_us: Annotated[
        float, Display(label="Fixed Rabi relaxation", unit="µs")
    ] = Field(default=30.0, ge=0, description="Fixed Rabi relaxation")
    ro_relax_mode: Annotated[
        Literal["auto", "fixed"], Display(label="Readout relaxation mode")
    ] = Field(default="auto", description="Readout relaxation mode")
    ro_relax_factor: Annotated[float, Display(label="Readout relaxation / T1")] = Field(
        default=3.0, gt=0, description="Readout relaxation / T1"
    )
    ro_relax_min_us: Annotated[
        float, Display(label="Readout relaxation floor", unit="µs")
    ] = Field(default=0.0, ge=0, description="Readout relaxation floor")
    ro_relax_delay_us: Annotated[
        float, Display(label="Fixed readout relaxation", unit="µs")
    ] = Field(default=30.0, ge=0, description="Fixed readout relaxation")
    ro_freq_half_width_mhz: Annotated[
        float, Display(label="Readout frequency half width", unit="MHz")
    ] = Field(default=1.0, gt=0, description="Readout frequency half width")
    ro_gain_half_width: Annotated[float, Display(label="Readout gain half width")] = (
        Field(default=0.1, gt=0, le=1, description="Readout gain half width")
    )
    ro_skew_penalty: Annotated[float, Display(label="Readout skew penalty")] = Field(
        default=0.0, ge=0, description="Readout skew penalty"
    )

    @model_validator(mode="after")
    def _validate_rabi_range(self) -> Self:
        if self.rabi_stop_factor <= self.rabi_start_factor:
            raise field_error(
                ("rabi_stop_factor",), "Rabi stop factor must exceed start factor"
            )
        if self.rabi_min_stop_us <= self.rabi_min_start_us:
            raise field_error(
                ("rabi_min_stop_us",), "Rabi stop floor must exceed start floor"
            )
        return self

    def to_cfg(self) -> CalibrationTunables:
        """Copy validated ordinary knobs into the native calibration model.

        No expression parsing, context access or device action occurs. Native
        validation failures propagate if the caller bypassed form validation.
        """
        return CalibrationTunables(
            qf_reps=self.qf_reps,
            qf_rounds=self.qf_rounds,
            qf_acquire_retry=self.qf_acquire_retry,
            qf_earlystop_snr=self.qf_earlystop_snr,
            rabi_reps=self.rabi_reps,
            rabi_rounds=self.rabi_rounds,
            rabi_acquire_retry=self.rabi_acquire_retry,
            rabi_earlystop_snr=self.rabi_earlystop_snr,
            ro_reps=self.ro_reps,
            ro_rounds=self.ro_rounds,
            ro_acquire_retry=self.ro_acquire_retry,
            frequency_mode=self.frequency_mode,
            coarse_gain=self.coarse_gain,
            coarse_span_factor=self.coarse_span_factor,
            fine_gain=self.fine_gain,
            fine_span_mhz=self.fine_span_mhz,
            peak_min_prominence=self.peak_min_prominence,
            peak_min_distance_mhz=self.peak_min_distance_mhz,
            peak_min_snr=self.peak_min_snr,
            peak_min_width_samples=self.peak_min_width_samples,
            max_candidates=self.max_candidates,
            span_mhz=self.span_mhz,
            qf_widen=self.qf_widen,
            detune_center_mhz=self.detune_center_mhz,
            qf_points=self.qf_points,
            frequency_edge_margin=self.frequency_edge_margin,
            linewidth_max_mean_residual_ratio=self.linewidth_max_mean_residual_ratio,
            rabi_max_mean_residual_ratio=self.rabi_max_mean_residual_ratio,
            drive_gain_mode=self.drive_gain_mode,
            target_kappa=self.target_kappa,
            drive_gain_cap=self.drive_gain_cap,
            physical_recovery_mode=self.physical_recovery_mode,
            max_frequency_rel_err=self.max_frequency_rel_err,
            max_rabi_rel_err=self.max_rabi_rel_err,
            max_fit_residual=self.max_fit_residual,
            min_readout_snr=self.min_readout_snr,
            ro_every=self.ro_every,
            rabi_start_factor=self.rabi_start_factor,
            rabi_stop_factor=self.rabi_stop_factor,
            rabi_drive_gain_mode=self.rabi_drive_gain_mode,
            target_pi_length_us=self.target_pi_length_us,
            rabi_min_start_us=self.rabi_min_start_us,
            rabi_min_stop_us=self.rabi_min_stop_us,
            rabi_points=self.rabi_points,
            rabi_relax_mode=self.rabi_relax_mode,
            rabi_relax_factor=self.rabi_relax_factor,
            rabi_relax_min_us=self.rabi_relax_min_us,
            rabi_relax_delay_us=self.rabi_relax_delay_us,
            ro_relax_mode=self.ro_relax_mode,
            ro_relax_factor=self.ro_relax_factor,
            ro_relax_min_us=self.ro_relax_min_us,
            ro_relax_delay_us=self.ro_relax_delay_us,
            ro_freq_half_width_mhz=self.ro_freq_half_width_mhz,
            ro_gain_half_width=self.ro_gain_half_width,
            ro_skew_penalty=self.ro_skew_penalty,
        )
