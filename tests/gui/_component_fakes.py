"""Small native component models for GUI binding and predictor collaborators."""

from pydantic import BaseModel, Field
from zcu_tools.format_version import YamlMap
from zcu_tools.resources.entry.schema import ComponentSchema


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
