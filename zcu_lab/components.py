"""Explicitly bootstrapped lab definitions; no import-time registration."""

from math import isfinite
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from zcu_tools.format_version import YamlMap
from zcu_tools.resources.entry.registry import ComponentRegistry, RoleSpec
from zcu_tools.resources.entry.schema import ComponentSchema
from zcu_tools.resources.entry.views import ComponentView, FieldView
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams
from zcu_tools.simulate.predictor import PredictorInputError
from zcu_tools.simulate.transmon import TransmonParams

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

    ch: int | None = Field(
        default=None, ge=0, strict=True, description="Generator (output) channel index."
    )
    ro_ch: int | None = Field(
        default=None, ge=0, strict=True, description="Readout (ADC) channel index."
    )

    @field_validator("ch", "ro_ch", mode="before")
    @classmethod
    def reject_null_channels(cls, value: object) -> object:
        if value is None:
            raise ValueError("a supplied wiring channel must be a non-negative integer")
        return value


class ChannelSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ch: int = Field(ge=0, strict=True, description="Generator (output) channel index.")


class QubitWiringSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    L01: ChannelSchema | None = Field(
        default=None, description="Drive line for the 0-1 transition."
    )
    L14: ChannelSchema | None = Field(
        default=None, description="Drive line for the 1-4 transition."
    )
    L45: ChannelSchema | None = Field(
        default=None, description="Drive line for the 4-5 transition."
    )
    L56: ChannelSchema | None = Field(
        default=None, description="Drive line for the 5-6 transition."
    )
    flux: ChannelSchema | None = Field(default=None, description="Local flux line.")


class BuiltinComponent(ComponentSchema):
    model_config = ConfigDict(extra="forbid")

    ext: YamlMap = Field(
        default_factory=dict, description="Extra values without a fixed schema."
    )
    wiring: WiringSchema = Field(
        default_factory=WiringSchema, description="Hardware channel wiring."
    )
    module: dict[str, str] = Field(
        default_factory=dict,
        description="Module slots: slot name to a module library key.",
    )


class ResonatorSchema(BuiltinComponent):
    freq: Frequency = Field(default=None, description="Resonator frequency.")
    kappa: Frequency = Field(default=None, description="Resonator linewidth.")
    amplifier: str | None = Field(
        default=None, description="Name of the readout amplifier component."
    )


class QubitBaseSchema(BuiltinComponent):
    """Fields shared by every qubit kind; physics energies belong to subclasses."""

    wiring: QubitWiringSchema = Field(
        default_factory=QubitWiringSchema,
        description="Drive and flux lines per transition.",
    )
    freq: Frequency = Field(default=None, description="Qubit 0-1 transition frequency.")
    kappa: Frequency = Field(default=None, description="Qubit transition linewidth.")
    t1: Duration = Field(default=None, description="Energy relaxation time T1.")
    t2r: Duration = Field(default=None, description="Ramsey dephasing time T2*.")
    t2e: Duration = Field(default=None, description="Echo dephasing time T2E.")
    # These three values belong to global flux, interpreted by general.ext.flux_unit.
    flux_half: Flux = Field(
        default=None,
        description="Global flux at the half flux quantum, in general.ext.flux_unit.",
    )
    flux_period: Flux = Field(
        default=None,
        description="Global flux span of one flux quantum, in general.ext.flux_unit.",
    )
    flux_int: Flux = Field(
        default=None,
        description="Global flux at an integer flux quantum, in general.ext.flux_unit.",
    )
    flux_unit: Literal["A", "V"] | None = Field(
        default=None, description="Unit of the local flux bias: A or V."
    )
    resonator: str | None = Field(
        default=None, description="Name of the readout resonator component."
    )
    flux_source: str | None = Field(
        default=None, description="Name of the component that supplies flux bias."
    )


class QubitSchema(QubitBaseSchema):
    """Fluxonium qubit: EJ, EC and EL energies in GHz."""

    EJ: Energy = Field(default=None, description="Josephson energy.")
    EC: Energy = Field(default=None, description="Charging energy.")
    EL: Energy = Field(default=None, description="Inductive energy.")


class TransmonSchema(QubitBaseSchema):
    """SQUID transmon: EJmax and EC in GHz, plus dimensionless asymmetry d."""

    EJmax: Energy = Field(default=None, description="Maximum SQUID Josephson energy.")
    EC: Energy = Field(default=None, description="Charging energy.")
    d: Annotated[float | None, Field(strict=True, allow_inf_nan=False)] = Field(
        default=None, description="Dimensionless SQUID junction asymmetry."
    )


class JpaSchema(BuiltinComponent):
    pump_freq: Frequency = Field(default=None, description="JPA pump frequency.")
    pump_power: PumpPower = Field(default=None, description="JPA pump power.")
    flux: Flux = Field(default=None, description="JPA flux bias, in flux_unit.")
    flux_unit: Literal["A", "V"] | None = Field(
        default=None, description="Unit of the JPA flux bias: A or V."
    )


def params_for(
    component: ComponentSchema | ComponentView,
) -> FluxoniumParams | TransmonParams:
    """Read independent model inputs from a qubit component's current values.

    Accept a notebook Point/Setup ComponentView or an isolated ComponentSchema
    from a GUI context capture. Fluxonium reads EJ/EC/EL and SQUID transmon reads
    EJmax/EC/d; energies are GHz and d is dimensionless. Both read flux_half,
    flux_period and ext.flux_bias in native device units. Missing/unset fields
    stay None. No read refreshes, edits, adds defaults or builds a model.
    Unsupported kinds and nonnumeric/nonfinite supplied fields raise PredictorInputError
    naming the field.
    """
    if component.kind not in ("qubit/fluxonium", "qubit/transmon"):
        raise PredictorInputError(f"Unsupported predictor kind: {component.kind}")
    ext = getattr(component, "ext", None)
    if isinstance(ext, dict):
        bias = ext.get("flux_bias")
    elif isinstance(ext, FieldView):
        bias = getattr(ext, "flux_bias", None)
    else:
        bias = None

    def read(name: str) -> float | None:
        return _optional_number(getattr(component, name, None), name)

    affine = {
        "flux_half": read("flux_half"),
        "flux_period": read("flux_period"),
        "flux_bias": _optional_number(bias, "flux_bias"),
    }
    if component.kind == "qubit/transmon":
        return TransmonParams(EJmax=read("EJmax"), EC=read("EC"), d=read("d"), **affine)
    return FluxoniumParams(EJ=read("EJ"), EC=read("EC"), EL=read("EL"), **affine)


def _optional_number(value: object, name: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not isfinite(value)
    ):
        raise PredictorInputError(f"{name} must be a finite number or None")
    return float(value)


def register_all(registry: ComponentRegistry) -> None:
    """Register definitions into the caller's registry; duplicates fail normally."""
    registry.register("resonator", ResonatorSchema)
    registry.register("device/current_source", BuiltinComponent)
    registry.register("amplifier/jpa", JpaSchema)
    registry.register("qubit/fluxonium", QubitSchema)
    registry.register("qubit/transmon", TransmonSchema)
    for name, kind in (
        ("qubit", "qubit/*"),
        ("resonator", "resonator"),
        ("control", "qubit/*"),
        ("target", "qubit/*"),
        ("coupler", "coupler/*"),
    ):
        registry.roles.register(name, RoleSpec(kind))
    registry.roles.configure(shorthand=("qubit", "resonator"), focus_kinds=("qubit/*",))
