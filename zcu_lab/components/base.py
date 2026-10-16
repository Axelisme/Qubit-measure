"""Shared lab field types, channel wiring and component storage fields."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator
from zcu_tools.format_version import YamlMap
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
