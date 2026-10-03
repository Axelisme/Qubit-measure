"""Typed setup document at the persistence boundary."""

import keyword
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from zcu_tools.format_version import YamlMap
from zcu_tools.resources.document_store import UnitSpec

_VIEW_NAMES = frozenset(
    {
        "description",
        "edit",
        "refresh",
        "add_component",
        "general",
        "meta",
        "set",
        "move",
        "resolve",
        "components",
        "entry_id",
        "created_at",
        "kind",
        "wiring",
        "ext",
    }
)


def validate_component_name(name: str, *, source: Path | None = None) -> None:
    if (
        not name.isidentifier()
        or name.startswith("_")
        or keyword.iskeyword(name)
        or name in _VIEW_NAMES
    ):
        raise ValueError(
            f"{source}: invalid component name {name!r}; expected a public identifier that does not shadow a view"
        )


class SetupGeneral(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_id: str
    created_at: str
    description: str | None = None
    ext: YamlMap = Field(default_factory=dict)

    @field_validator("entry_id")
    @classmethod
    def validate_entry_id(cls, value: str) -> str:
        UUID(value)
        return value

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        if datetime.fromisoformat(value).utcoffset() != timedelta(0):
            raise ValueError("created_at must be a UTC timestamp")
        return value


class WiringSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ch: int | None = Field(default=None, ge=0, strict=True)
    ro_ch: int | None = Field(default=None, ge=0, strict=True)
    flux_ch: int | None = Field(default=None, ge=0, strict=True)
    time_of_flight: Annotated[float | None, UnitSpec("s", "us")] = Field(
        default=None, ge=0
    )


class ComponentSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    wiring: WiringSchema = Field(default_factory=WiringSchema)
    ext: YamlMap = Field(default_factory=dict)


class ResonatorSchema(ComponentSchema):
    freq: Annotated[float | None, UnitSpec("Hz", "MHz")] = None


class SetupDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: str
    format_version: str
    general: SetupGeneral
    components: dict[str, ComponentSchema] = Field(default_factory=dict)
    provenance: dict[str, YamlMap] = Field(default_factory=dict)

    @field_validator("components", mode="before")
    @classmethod
    def validate_components(cls, value: object) -> dict[str, ComponentSchema]:
        # Import locally because registered models derive from ComponentSchema.
        from .registry import component_registry

        components = TypeAdapter(dict[str, YamlMap]).validate_python(value)
        result: dict[str, ComponentSchema] = {}
        for name, fields in components.items():
            kind = fields.get("kind")
            model = (
                component_registry.get(kind)
                if isinstance(kind, str)
                else ComponentSchema
            )
            result[name] = model.model_validate(fields)
        return result
