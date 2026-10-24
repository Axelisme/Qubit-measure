"""common calibration_form helpers for lab workflows."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, model_validator
from zcu_tools.gui.form import Display, field_error

from zcu_lab.workflows.common.calibration import CalibrationTunables
from zcu_lab.workflows.common.frequency_form import FrequencyTunablesForm


class CalibrationTunablesForm(FrequencyTunablesForm):
    """User-editable calibration generation, acceptance and cadence knobs.

    Field descriptions carry the ordinary knob meanings. Acceptance/cadence and
    range defaults remain provisional simulator settings for CP-B/CP-C, not
    hardware-validated physics defaults. No hidden fixed value is edited here.
    """

    qf_reps: Annotated[int, Display(label="Spectroscopy averages")] = Field(
        default=100, ge=1, description="Spectroscopy averages"
    )
    qf_rounds: Annotated[int, Display(label="Spectroscopy rounds")] = Field(
        default=1, ge=1, description="Spectroscopy averaging rounds"
    )
    rabi_reps: Annotated[int, Display(label="Rabi averages")] = Field(
        default=100, ge=1, description="Rabi averages"
    )
    rabi_rounds: Annotated[int, Display(label="Rabi rounds")] = Field(
        default=1, ge=1, description="Rabi averaging rounds"
    )
    ro_reps: Annotated[int, Display(label="Readout averages")] = Field(
        default=100, ge=1, description="Readout averages"
    )
    ro_rounds: Annotated[int, Display(label="Readout rounds")] = Field(
        default=1, ge=1, description="Readout averaging rounds"
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

    @model_validator(mode="after")
    def _validate_rabi_range(self) -> Self:
        if self.rabi_stop_factor <= self.rabi_start_factor:
            raise field_error(
                ("rabi_stop_factor",), "Rabi stop factor must exceed start factor"
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
            rabi_reps=self.rabi_reps,
            rabi_rounds=self.rabi_rounds,
            ro_reps=self.ro_reps,
            ro_rounds=self.ro_rounds,
            span_mhz=self.span_mhz,
            qf_widen=self.qf_widen,
            detune_center_mhz=self.detune_center_mhz,
            qf_points=self.qf_points,
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
        )
