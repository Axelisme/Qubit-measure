"""Explicitly bootstrapped lab definitions; no import-time registration."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .registry import ComponentRegistry, RoleSpec
from .schema import ComponentSchema, ModuleSlot, Ref, UnitSpec

Frequency = Annotated[
    float | None, UnitSpec("MHz"), Field(strict=True, allow_inf_nan=False)
]
Duration = Annotated[
    float | None, UnitSpec("µs"), Field(strict=True, allow_inf_nan=False)
]
Energy = Annotated[
    float | None, UnitSpec("GHz"), Field(strict=True, allow_inf_nan=False)
]
Flux = Annotated[float | None, UnitSpec("A/V"), Field(strict=True, allow_inf_nan=False)]
PumpPower = Annotated[
    float | None, UnitSpec("dBm"), Field(strict=True, allow_inf_nan=False)
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
    wiring: WiringSchema = Field(default_factory=WiringSchema)
    module: Annotated[dict[str, str], ModuleSlot()] = Field(default_factory=dict)


class ResonatorSchema(BuiltinComponent):
    freq: Frequency = None
    kappa: Frequency = None
    amplifier: Annotated[str | None, Ref()] = None


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
    # These three values belong to global flux, interpreted by general.flux_unit.
    flux_half: Flux = None
    flux_period: Flux = None
    flux_int: Flux = None
    flux_unit: Literal["A", "V"] | None = None
    resonator: Annotated[str | None, Ref()] = None
    flux_source: Annotated[str | None, Ref()] = None


class JpaSchema(BuiltinComponent):
    pump_freq: Frequency = None
    pump_power: PumpPower = None
    flux: Flux = None
    flux_unit: Literal["A", "V"] | None = None


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
