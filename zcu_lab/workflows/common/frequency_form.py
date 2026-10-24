"""Captured physics and editable spectroscopy knobs shared by both workflows."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, JsonValue, model_validator
from zcu_tools.gui.form import FormEnv, FormModel, field_error

from zcu_lab.workflows.common.frequency_tracking import FrequencyModel


class FrequencyModelForm(FormModel):
    """Choose fluxonium energies (GHz) and the plan's native device affine map.

    Defaults use only the fixed $qubit parameter capture. Missing calibrations
    stay required; flux_bias defaults to zero because the point has no bias
    field. Explicit values can replace every captured value before Start.
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
            for name in ("EJ", "EC", "EL", "flux_half", "flux_period"):
                data[name] = {"expr": f"$qubit.{name}"}
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
    """Edit detune offset, point count, linewidth gain and physical recovery mode.

    The inherited calibration form supplies span and widening. Offset plus/minus
    half-span supports asymmetric windows, e.g. offset 15, span 70 gives -20..50.
    """

    detune_center_mhz: float = Field(
        default=0.0, description="Detune window offset (MHz)"
    )
    qf_points: int = Field(default=201, ge=2, description="Spectroscopy sample count")
    drive_gain_mode: Literal["adaptive", "fixed"] = "adaptive"
    target_kappa: float = Field(default=6.5, gt=0, description="Target linewidth (MHz)")
    drive_gain_cap: float = Field(
        default=1.0, gt=0, le=1, description="Maximum adaptive probe gain"
    )
    physical_recovery_mode: Literal["off", "fail_triggered_fit"] = "fail_triggered_fit"
