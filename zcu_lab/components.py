"""Explicitly bootstrapped lab definitions; no import-time registration."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from zcu_tools.format_version import YamlMap
from zcu_tools.resources.entry.registry import ComponentRegistry, RoleSpec
from zcu_tools.resources.entry.schema import ComponentSchema

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


class QubitSchema(BuiltinComponent):
    wiring: QubitWiringSchema = Field(
        default_factory=QubitWiringSchema,
        description="Drive and flux lines per transition.",
    )
    freq: Frequency = Field(default=None, description="Qubit 0-1 transition frequency.")
    kappa: Frequency = Field(default=None, description="Qubit transition linewidth.")
    t1: Duration = Field(default=None, description="Energy relaxation time T1.")
    t2r: Duration = Field(default=None, description="Ramsey dephasing time T2*.")
    t2e: Duration = Field(default=None, description="Echo dephasing time T2E.")
    EJ: Energy = Field(default=None, description="Josephson energy.")
    EC: Energy = Field(default=None, description="Charging energy.")
    EL: Energy = Field(default=None, description="Inductive energy.")
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


class JpaSchema(BuiltinComponent):
    pump_freq: Frequency = Field(default=None, description="JPA pump frequency.")
    pump_power: PumpPower = Field(default=None, description="JPA pump power.")
    flux: Flux = Field(default=None, description="JPA flux bias, in flux_unit.")
    flux_unit: Literal["A", "V"] | None = Field(
        default=None, description="Unit of the JPA flux bias: A or V."
    )


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
