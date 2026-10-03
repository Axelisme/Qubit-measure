"""Typed setup document at the persistence boundary."""

from pydantic import BaseModel, ConfigDict, Field

from zcu_tools.format_version import YamlMap


class SetupGeneral(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_id: str
    created_at: str
    description: str | None = None
    ext: YamlMap = Field(default_factory=dict)


class SetupDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: str
    format_version: str
    general: SetupGeneral
    components: dict[str, YamlMap] = Field(default_factory=dict)
    provenance: dict[str, YamlMap] = Field(default_factory=dict)
