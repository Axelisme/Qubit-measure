"""Local schemas for container behavior, independent of lab definitions."""

from copy import deepcopy
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator
from zcu_tools.format_version import YamlMap
from zcu_tools.resources.entry import (
    ComponentSchema,
    RoleSpec,
    component_registry,
    role_registry,
)

type Number = Annotated[float | None, Field(strict=True, allow_inf_nan=False)]


class Connections(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ch: int | None = Field(default=None, ge=0, strict=True)
    ro_ch: int | None = Field(default=None, ge=0, strict=True)
    bias_ch: int | None = Field(default=None, ge=0, strict=True)
    delay: Annotated[float | None, Field(ge=0, strict=True, allow_inf_nan=False)] = None

    @field_validator("ch", "ro_ch", "bias_ch", mode="before")
    @classmethod
    def reject_null_channels(cls, value: object) -> object:
        if value is None:
            raise ValueError("a supplied channel must be a non-negative integer")
        return value


class Connected(ComponentSchema):
    model_config = ConfigDict(extra="forbid")
    ext: YamlMap = Field(default_factory=dict)
    wiring: Connections = Field(default_factory=Connections)


class Sensor(Connected):
    rate: Number = None
    bandwidth: Number = None
    booster: str | None = None


class Driver(Connected):
    rate: Number = None
    energy_a: Number = None
    energy_b: Number = None
    energy_c: Number = None
    pulse_width: Number = None
    duration: Number = None
    coherence: Number = None
    strength: Number = None
    bias_half: Number = None
    bias_period: Number = None
    sense: str | None = None
    source: str | None = None


class Booster(Connected):
    rate: Number = None
    gain: Number = None
    level: Number = None


class Supply(Connected):
    level: Number = None


def register_fakes() -> None:
    component_registry.register("fake/sensor", Sensor)
    component_registry.register("fake/drive/a", Driver)
    component_registry.register("fake/drive/b", Driver)
    component_registry.register("fake/booster", Booster)
    component_registry.register("fake/supply", Supply)
    for name, kind in {
        "driver": "fake/drive/*",
        "sense": "fake/sensor",
        "control": "fake/drive/*",
        "target": "fake/drive/*",
        "coupler": "coupler/*",
    }.items():
        role_registry.register(name, RoleSpec(kind))
    role_registry.configure(
        shorthand=("driver", "sense"), focus_kinds=("fake/drive/*",)
    )


def registry_state() -> tuple[dict[str, object], dict[str, object]]:
    """Snapshot mutable tables without comparing a copied registry by identity."""
    return deepcopy(
        (
            {
                key: value
                for key, value in vars(component_registry).items()
                if key != "roles"
            },
            vars(role_registry),
        )
    )


def restore_registry(state: tuple[dict[str, object], dict[str, object]]) -> None:
    components, roles = state
    vars(component_registry).clear()
    vars(component_registry).update(components)
    component_registry.roles = role_registry
    vars(role_registry).clear()
    vars(role_registry).update(roles)
