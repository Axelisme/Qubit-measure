"""Small component shapes for fixed captures, bindings and predictor RPC tests."""

from typing import Annotated

from pydantic import BaseModel, Field
from zcu_tools.format_version import YamlMap
from zcu_tools.resources.entry.registry import ComponentRegistry, RoleSpec
from zcu_tools.resources.entry.schema import ComponentSchema

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


def register_all(registry: ComponentRegistry) -> None:
    registry.register("resonator", ResonatorSchema)
    registry.register("qubit/fluxonium", QubitSchema)
    registry.register("qubit/transmon", TransmonSchema)
    registry.roles.register("qubit", RoleSpec("qubit/*"))
    registry.roles.register("resonator", RoleSpec("resonator"))
    registry.roles.configure(shorthand=("qubit", "resonator"), focus_kinds=("qubit/*",))
