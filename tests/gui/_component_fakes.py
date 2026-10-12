"""Small native component models for GUI binding and predictor collaborators."""

from pydantic import BaseModel, Field
from zcu_tools.format_version import YamlMap
from zcu_tools.resources.entry import ComponentView
from zcu_tools.resources.entry.schema import ComponentSchema
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams


class Connections(BaseModel):
    ch: int | None = None
    ro_ch: int | None = None


class Resonator(ComponentSchema):
    ext: YamlMap = Field(default_factory=dict)
    freq: float | None = None
    kappa: float | None = None
    wiring: Connections = Field(default_factory=Connections)


class Qubit(ComponentSchema):
    ext: YamlMap = Field(default_factory=dict)
    freq: float | None = None
    kappa: float | None = None
    EJ: float | None = None
    EC: float | None = None
    EL: float | None = None
    flux_half: float | None = None
    flux_int: float | None = None
    flux_period: float | None = None


def params_for(
    component: ComponentSchema | ComponentView,
) -> FluxoniumParams:
    """Project only the caller's synthetic predictor inputs from its fixed capture."""
    if not isinstance(component, Qubit):
        raise ValueError("Expected a synthetic qubit")
    bias = component.ext.get("flux_bias")
    if bias is not None and not isinstance(bias, float):
        raise ValueError("Synthetic flux bias must be a float")
    return FluxoniumParams(
        component.EJ,
        component.EC,
        component.EL,
        component.flux_half,
        component.flux_period,
        bias,
    )
