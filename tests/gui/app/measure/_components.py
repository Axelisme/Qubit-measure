"""Small component shapes for fixed captures, bindings and predictor RPC tests."""

from typing import Annotated

from pydantic import BaseModel, Field
from zcu_tools.format_version import YamlMap
from zcu_tools.resources.entry.registry import ComponentRegistry, RoleSpec
from zcu_tools.resources.entry.schema import ComponentSchema
from zcu_tools.resources.entry.views import ComponentView
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams
from zcu_tools.simulate.transmon import TransmonParams

Number = Annotated[float | None, Field(allow_inf_nan=False)]


class WiringSchema(BaseModel):
    ch: int | None = None
    ro_ch: int | None = None


class ResonatorSchema(ComponentSchema):
    freq: Number = None
    kappa: Number = None
    wiring: WiringSchema = Field(default_factory=WiringSchema)
    ext: YamlMap = Field(default_factory=dict)
    module: dict[str, str] = Field(default_factory=dict)


class QubitSchema(ComponentSchema):
    freq: Number = None
    t1: Number = None
    flux_half: Number = None
    flux_int: Number = None
    flux_period: Number = None
    resonator: str | None = None
    EJ: Number = None
    EC: Number = None
    EL: Number = None
    ext: YamlMap = Field(default_factory=dict)
    module: dict[str, str] = Field(default_factory=dict)


class TransmonSchema(QubitSchema):
    EJmax: Number = None
    d: Number = None


def params_for(
    component: ComponentSchema | ComponentView,
) -> FluxoniumParams | TransmonParams:
    """Fixture-owned input projection; the framework receives this callback."""
    if not isinstance(component, QubitSchema):
        raise ValueError("Fixture requires a predictor component")
    bias = component.ext.get("flux_bias")
    if bias is not None and not isinstance(bias, (float, int)):
        raise ValueError("Fixture bias must be numeric")
    if isinstance(component, TransmonSchema):
        return TransmonParams(
            EJmax=component.EJmax,
            EC=component.EC,
            d=component.d,
            flux_half=component.flux_half,
            flux_period=component.flux_period,
            flux_bias=bias,
        )
    return FluxoniumParams(
        EJ=component.EJ,
        EC=component.EC,
        EL=component.EL,
        flux_half=component.flux_half,
        flux_period=component.flux_period,
        flux_bias=bias,
    )


def register_all(registry: ComponentRegistry) -> None:
    registry.register("resonator", ResonatorSchema)
    registry.register("qubit/fluxonium", QubitSchema)
    registry.register("qubit/transmon", TransmonSchema)
    registry.roles.register("qubit", RoleSpec("qubit/*"))
    registry.roles.register("resonator", RoleSpec("resonator"))
    registry.roles.configure(shorthand=("qubit", "resonator"), focus_kinds=("qubit/*",))
