"""Explicitly bootstrapped lab definitions; no import-time registration."""

from math import isfinite
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from zcu_tools.format_version import YamlMap
from zcu_tools.resources.entry.registry import ComponentRegistry, RoleSpec
from zcu_tools.resources.entry.schema import ComponentSchema
from zcu_tools.resources.entry.views import ComponentView, FieldView
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams

Frequency = Annotated[
    float | None,
    Field(strict=True, allow_inf_nan=False, json_schema_extra={"unit": "MHz"}),
]
Duration = Annotated[
    float | None,
    Field(strict=True, allow_inf_nan=False, json_schema_extra={"unit": "us"}),
]
Energy = Annotated[
    float | None,
    Field(strict=True, allow_inf_nan=False, json_schema_extra={"unit": "GHz"}),
]
Flux = Annotated[float | None, Field(strict=True, allow_inf_nan=False)]
PumpPower = Annotated[
    float | None,
    Field(strict=True, allow_inf_nan=False, json_schema_extra={"unit": "dBm"}),
]


class WiringSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ch: int | None = Field(default=None, ge=0, strict=True)
    ro_ch: int | None = Field(default=None, ge=0, strict=True)

    @field_validator("ch", "ro_ch", mode="before")
    @classmethod
    def reject_null_channels(cls, value: object) -> object:
        if value is None:
            raise ValueError("a supplied wiring channel must be a non-negative integer")
        return value


class ChannelSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ch: int = Field(ge=0, strict=True)


class QubitWiringSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    L01: ChannelSchema | None = None
    L14: ChannelSchema | None = None
    L45: ChannelSchema | None = None
    L56: ChannelSchema | None = None
    flux: ChannelSchema | None = None


class BuiltinComponent(ComponentSchema):
    model_config = ConfigDict(extra="forbid")

    ext: YamlMap = Field(default_factory=dict)
    wiring: WiringSchema = Field(default_factory=WiringSchema)
    module: dict[str, str] = Field(default_factory=dict)


class ResonatorSchema(BuiltinComponent):
    freq: Frequency = None
    kappa: Frequency = None
    amplifier: str | None = None


class QubitSchema(BuiltinComponent):
    wiring: QubitWiringSchema = Field(default_factory=QubitWiringSchema)
    freq: Frequency = None
    kappa: Frequency = None
    t1: Duration = None
    t2r: Duration = None
    t2e: Duration = None
    EJ: Energy = None
    EC: Energy = None
    EL: Energy = None
    # These three values belong to global flux, interpreted by general.ext.flux_unit.
    flux_half: Flux = None
    flux_period: Flux = None
    flux_int: Flux = None
    flux_unit: Literal["A", "V"] | None = None
    resonator: str | None = None
    flux_source: str | None = None


class JpaSchema(BuiltinComponent):
    pump_freq: Frequency = None
    pump_power: PumpPower = None
    flux: Flux = None
    flux_unit: Literal["A", "V"] | None = None


def params_for(component: ComponentSchema | ComponentView) -> FluxoniumParams:
    """Read independent model inputs from a fluxonium component's current values.

    Accept a notebook Point/Setup ComponentView or an isolated ComponentSchema
    from a GUI context capture. Read EJ/EC/EL in GHz, flux_half/flux_period and
    ext.flux_bias in native device units. Missing/unset fields stay None.
    No read refreshes, edits, adds defaults or builds a model. Unsupported kinds
    and nonnumeric/nonfinite supplied fields raise ValueError naming the field.
    Only fluxonium is supported here; SQUID transmon is added with its own schema.
    """
    if component.kind != "qubit/fluxonium":
        raise ValueError(f"Unsupported predictor kind: {component.kind}")
    ext = getattr(component, "ext", None)
    if isinstance(ext, dict):
        bias = ext.get("flux_bias")
    elif isinstance(ext, FieldView):
        bias = getattr(ext, "flux_bias", None)
    else:
        bias = None
    return FluxoniumParams(
        EJ=_optional_number(getattr(component, "EJ", None), "EJ"),
        EC=_optional_number(getattr(component, "EC", None), "EC"),
        EL=_optional_number(getattr(component, "EL", None), "EL"),
        flux_half=_optional_number(getattr(component, "flux_half", None), "flux_half"),
        flux_period=_optional_number(
            getattr(component, "flux_period", None), "flux_period"
        ),
        flux_bias=_optional_number(bias, "flux_bias"),
    )


def _optional_number(value: object, name: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number or None")
    return float(value)


def register_all(registry: ComponentRegistry) -> None:
    """Register definitions into the caller's registry; duplicates fail normally."""
    registry.register("resonator", ResonatorSchema)
    registry.register("device/current_source", BuiltinComponent)
    registry.register("amplifier/jpa", JpaSchema)
    registry.register("qubit/fluxonium", QubitSchema)
    registry.register("qubit/transmon", QubitSchema)
    for name, kind in (
        ("qubit", "qubit/*"),
        ("resonator", "resonator"),
        ("control", "qubit/*"),
        ("target", "qubit/*"),
        ("coupler", "coupler/*"),
    ):
        registry.roles.register(name, RoleSpec(kind))
    registry.roles.configure(shorthand=("qubit", "resonator"), focus_kinds=("qubit/*",))
