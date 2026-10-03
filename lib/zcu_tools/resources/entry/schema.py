"""Typed setup document at the persistence boundary."""

from datetime import datetime, timedelta
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from zcu_tools.format_version import YamlMap


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


class ComponentSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    wiring: YamlMap = Field(default_factory=dict)
    ext: YamlMap = Field(default_factory=dict)


class SetupDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: str
    format_version: str
    general: SetupGeneral
    components: dict[str, ComponentSchema] = Field(default_factory=dict)
    provenance: dict[str, YamlMap] = Field(default_factory=dict)
