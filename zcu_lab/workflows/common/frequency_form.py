"""Captured physics and editable spectroscopy knobs shared by both workflows."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, JsonValue, model_validator
from zcu_tools.cfg_model import Gain
from zcu_tools.gui.form import Display, FormEnv, FormModel, field_error

from zcu_lab.v2.support.measure.flux_defaults import captured_flux_seed, captured_qubit
from zcu_lab.workflows.common.frequency_tracking import FrequencyModel


class FrequencyModelForm(FormModel):
    """Choose fluxonium energies (GHz) and the plan's native device affine map.

    Defaults use the fixed $qubit capture and detached device values. The
    selected flux_dev half accounts for other contributors; its period
    stays signed. Missing source/calibration leaves coordinates required.
    Invalid frames raise named errors. flux_bias defaults to zero; explicit
    values can replace every captured value before Start.
    """

    EJ: float = Field(gt=0, description="Josephson energy (GHz)")
    EC: float = Field(gt=0, description="Charging energy (GHz)")
    EL: float = Field(gt=0, description="Inductive energy (GHz)")
    flux_half: float = Field(description="Half-flux device value")
    flux_period: float = Field(description="Signed device span per flux quantum")
    flux_bias: float = Field(default=0.0, description="Additive device offset")

    @classmethod
    def defaults(cls, env: FormEnv) -> dict[str, JsonValue]:
        """Seed complete numeric calibrations from the fixed capture, never live I/O."""
        data = super().defaults(env)
        if env.parameters is not None:
            for name in ("EJ", "EC", "EL"):
                data[name] = {"expr": f"$qubit.{name}"}
            qubit = captured_qubit(env, "$qubit")
            source = qubit.flux_dev
            seed = captured_flux_seed(env, "$qubit", source)
            if seed.points is not None and source is not None:
                data["flux_half"] = seed.points.half
                data["flux_period"] = qubit.flux[source].period
        return data

    @model_validator(mode="after")
    def _nonzero_period(self) -> Self:
        if self.flux_period == 0:
            raise field_error(("flux_period",), "Flux period must be nonzero")
        return self

    def to_cfg(self) -> FrequencyModel:
        """Return a frozen native plan model without creating or fitting a predictor."""
        return FrequencyModel(**self.model_dump())


class FrequencyTunablesForm(FormModel):
    """Edit two-step/single scanning, peak selection and linewidth/recovery knobs.

    The inherited calibration form supplies base span and coarse widening.
    Coarse gain/span and fine gain/span are independently editable. Offset
    shifts coarse/single windows only; fine scans center on the coarse fit.
    """

    frequency_mode: Literal["two_step", "single"] = "two_step"
    coarse_gain: Gain | None = Field(
        default=None,
        gt=0,
        description="Coarse probe gain; None uses the probe pulse gain",
    )
    coarse_span_mhz: Annotated[float, Display(label="Coarse span", unit="MHz")] = Field(
        default=100.0,
        gt=0,
        description="Two-step coarse span in MHz; multiplied by widen during widening",
    )
    fine_gain: Gain | None = Field(
        default=None,
        gt=0,
        description="Fine probe gain; None uses half the adaptive gain, capped at half coarse gain",
    )
    fine_span_mhz: float | None = Field(
        default=None, gt=0, description="Fine span in MHz; None uses span_mhz"
    )
    peak_min_prominence: float = Field(
        default=0.2,
        ge=0,
        le=1,
        description="Minimum candidate prominence / projected signal range",
    )
    peak_min_distance_mhz: float = Field(
        default=0.0, ge=0, description="Minimum candidate separation in MHz"
    )

    peak_min_snr: float = Field(
        default=5.0, ge=0, description="Minimum peak height above median / noise sigma"
    )
    peak_min_width_samples: float = Field(
        default=3.0,
        ge=0,
        description="Minimum observable half-prominence width in samples",
    )
    max_candidates: int = Field(
        default=3,
        ge=1,
        description="Maximum coarse candidates tried per frequency point",
    )

    detune_center_mhz: float = Field(
        default=0.0, description="Detune window offset (MHz)"
    )
    qf_points: int = Field(default=201, ge=2, description="Spectroscopy sample count")
    frequency_edge_margin: float = Field(
        default=0.1,
        ge=0,
        lt=0.5,
        description="Minimum fitted-center distance from either edge as a scan-span fraction",
    )
    linewidth_max_mean_residual_ratio: float = Field(
        default=0.1,
        gt=0,
        description="Strict mean absolute residual/fit-range bound for linewidth feedback",
    )
    drive_gain_mode: Literal["adaptive", "fixed"] = "adaptive"
    target_kappa: float = Field(default=6.5, gt=0, description="Target linewidth (MHz)")
    drive_gain_cap: Gain = Field(
        default=1.0, gt=0, description="Maximum adaptive probe gain"
    )
    physical_recovery_mode: Literal["off", "fail_triggered_fit"] = "fail_triggered_fit"
