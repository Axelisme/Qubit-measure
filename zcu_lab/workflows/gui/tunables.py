"""T1 knob forms and explicit conversion to native workflow models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Self

from pydantic import Field, JsonValue, model_validator
from zcu_tools.gui.form import Display, FormEnv, FormModel, field_error

from zcu_lab.workflows.calibration import CalibrationTunables
from zcu_lab.workflows.t1 import T1Tunables


class CalibrationTunablesForm(FormModel):
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
    qf_widen: tuple[Annotated[float, Field(gt=0)], ...] = Field(
        default=(1.0, 2.0, 4.0),
        min_length=1,
        description="Ordered spectroscopy span multipliers",
    )
    max_frequency_rel_err: float = Field(
        default=0.3, gt=0, le=1, description="Strict center-error/FWHM upper bound"
    )
    max_rabi_rel_err: float = Field(
        default=0.3, gt=0, le=1, description="Strict pi-error/pi-duration upper bound"
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
        default=5, ge=1, description="Successful readout scan cadence"
    )
    rabi_start_factor: float = Field(
        default=0.1, gt=0, description="Sweep start divided by prior pi duration"
    )
    rabi_stop_factor: float = Field(
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
            max_frequency_rel_err=self.max_frequency_rel_err,
            max_rabi_rel_err=self.max_rabi_rel_err,
            max_fit_residual=self.max_fit_residual,
            min_readout_snr=self.min_readout_snr,
            ro_every=self.ro_every,
            rabi_start_factor=self.rabi_start_factor,
            rabi_stop_factor=self.rabi_stop_factor,
        )


class T1TunablesForm(FormModel):
    """User-facing T1 and calibration knobs; instances hold ordinary values only.

    cal owns calibration knobs. Other fields control T1 generation/acceptance as
    described in their Fields. defaults read only captured parameters; these
    knobs need neither library nor device infos. Submission belongs to the host.
    """

    cal: Annotated[
        CalibrationTunablesForm, Display(label="Calibration", collapsed=True)
    ] = Field(
        default_factory=CalibrationTunablesForm,
        description="Calibration generation and acceptance knobs",
    )
    reps: Annotated[int, Display(label="T1 averages")] = Field(
        default=100, ge=1, description="T1 averages"
    )
    rounds: Annotated[int, Display(label="T1 rounds")] = Field(
        default=10, ge=1, description="T1 averaging rounds"
    )
    guess_us: Annotated[
        float,
        Display(
            label="Initial lifetime", unit="us", placeholder="general.ext.guess_us"
        ),
    ] = Field(default=20.0, gt=0, description="Initial lifetime in us")
    sweep_factor: float = Field(
        default=5.0, gt=0, description="Sweep end divided by lifetime guess"
    )
    relax_factor: float = Field(
        default=3.0, gt=0, description="Relax delay divided by lifetime guess"
    )
    max_rel_err: float = Field(
        default=0.3, gt=0, le=1, description="Strict T1 error/lifetime upper bound"
    )
    max_fit_residual: float = Field(
        default=0.2, gt=0, le=1, description="Strict normalized residual RMS bound"
    )

    @classmethod
    def defaults(
        cls, env: FormEnv, previous: Mapping[str, JsonValue]
    ) -> dict[str, JsonValue]:
        """Build defaults, retaining a captured guess expression for initial input.

        When previous is empty and parameters contain general.ext.guess_us
        (even explicit null),
        retain that path as an expression rather than copying its value. With
        no capture or an absent leaf, keep the provisional ordinary 20 us.
        Invalid referenced values remain result errors, not a fallback. Reads
        use the fixed point only, never materialization, devices or I/O.
        Unavailable captured sources propagate their source error.
        """
        data = super().defaults(env, previous)
        if (
            not previous
            and env.parameters is not None
            and env.parameters.lookup_point("general.ext.guess_us").present
        ):
            data["guess_us"] = {"expr": "general.ext.guess_us"}
        return data

    def to_cfg(self) -> T1Tunables:
        """Copy validated ordinary form values to native T1Tunables.

        Expression input stays in FormInput, not in the native values or Engine
        journal. No owner update or source access occurs. Native validation
        failures propagate after invalid direct edits.
        """
        return T1Tunables(
            cal=self.cal.to_cfg(),
            reps=self.reps,
            rounds=self.rounds,
            guess_us=self.guess_us,
            sweep_factor=self.sweep_factor,
            relax_factor=self.relax_factor,
            max_rel_err=self.max_rel_err,
            max_fit_residual=self.max_fit_residual,
        )
