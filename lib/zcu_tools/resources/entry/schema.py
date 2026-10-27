"""Typed setup document at the persistence boundary."""

import keyword
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import ClassVar, Literal, Self, override
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
)

from zcu_tools.format_version import YamlMap, YamlValue
from zcu_tools.resources.parameter_metadata import (
    PARAMETER_FORMAT,
    PARAMETER_VERSION,
    validate_parameter_header,
)

__all__ = (
    "PARAMETER_FORMAT",
    "PARAMETER_VERSION",
    "ComponentSchema",
    "PointDocument",
    "PointGeneral",
    "SetupDocument",
    "SetupGeneral",
    "is_forward_minor",
    "validate_component_name",
)


def is_forward_minor(document: Mapping[str, YamlValue], *, source: Path) -> bool:
    """Validate the shared parameter-container 2.x header; report a newer minor.

    Old majors and provenance tables raise with source and versions, exactly as
    the standalone parameter entry does. No data is changed or migrated.
    """
    version = validate_parameter_header(document, source=source)
    return version.minor > PARAMETER_VERSION.minor


def validate_component_name(name: str, *, source: Path | None = None) -> None:
    """Require a public Python identifier, excluding keywords and leading underscores.

    source is the optional document path for diagnostics. Invalid names raise
    ValueError. View namespace collisions belong to view binding, not schema.
    """
    if not name.isidentifier() or name.startswith("_") or keyword.iskeyword(name):
        raise ValueError(
            f"{source}: invalid component name {name!r}; expected a public identifier"
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


class PointGeneral(BaseModel):
    """Independent metadata copied from the seed setup or source point.

    entry_id optionally retains the seed entry UUID; it is not point identity.
    created_at is renewed on creation, as an ISO-8601 UTC timestamp.
    description is optional human-readable text; None means no description.
    ext is a YAML mapping, empty by default, with no unit conversion.
    Empty keys or keys containing dots cannot be addressed by set/meta.
    Unknown metadata fields and invalid timestamps raise Pydantic ValidationError.
    """

    model_config = ConfigDict(extra="forbid")

    entry_id: str | None = None
    created_at: str
    description: str | None = None
    ext: YamlMap = Field(default_factory=dict)

    @field_validator("entry_id")
    @classmethod
    def validate_entry_id(cls, value: str | None) -> str | None:
        if value is not None:
            SetupGeneral.validate_entry_id(value)
        return value

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return SetupGeneral.validate_created_at(value)


class ComponentSchema(BaseModel):
    """Base for registered components, requiring only a string kind.

    Definitions own extra policy, validators, hooks and optional containers,
    independently of the parameter-container version. Forbid rejects unknown
    fields, ignore discards them, and allow retains them at every minor version.
    Loading uses model_validate and saving uses model_dump. Mapping keys with
    dots or an empty name cannot be addressed by set/meta dotted paths.
    """

    kind: str


class _ParameterDocument(BaseModel):
    """Shared typed component boundary for complete parameter documents."""

    model_config = ConfigDict(extra="forbid")
    _source: ClassVar[Path] = Path("setup.yaml")
    _upgrade_target: ClassVar[Path | None] = None

    format: str
    format_version: str
    schema_version: int = Field(strict=True, ge=1)
    components: dict[str, ComponentSchema] = Field(default_factory=dict)
    metadata: dict[str, YamlMap] = Field(default_factory=dict)

    @classmethod
    @override
    def model_validate(
        cls,
        obj: object,
        *,
        strict: bool | None = None,
        extra: Literal["allow", "ignore", "forbid"] | None = None,
        from_attributes: bool | None = None,
        context: object = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        # Lookup errors must retain their public entry error type rather than
        # becoming Pydantic field-validator ValueErrors.
        from .registry import component_registry

        if isinstance(obj, Mapping):
            document = TypeAdapter(YamlMap).validate_python(obj)
            is_forward_minor(document, source=cls._source)
            component_registry.require_schema_policy().require_current(
                document, source=cls._source, upgrade_target=cls._upgrade_target
            )
            components = TypeAdapter(dict[str, YamlMap]).validate_python(
                document.get("components", {})
            )
            for name, fields in components.items():
                validate_component_name(name, source=cls._source)
                kind = fields.get("kind")
                if isinstance(kind, str):
                    component_registry.get(kind, source=cls._source, component=name)
        return super().model_validate(
            obj,
            strict=strict,
            extra=extra,
            from_attributes=from_attributes,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    @field_validator("components", mode="before")
    @classmethod
    def validate_components(cls, value: object) -> dict[str, ComponentSchema]:
        # Import locally because registered models derive from ComponentSchema.
        from .registry import component_registry

        # A separate validation call keeps the container's unknown-field policy
        # from overriding the original component and nested models' extra policies.
        components = TypeAdapter(dict[str, YamlMap]).validate_python(value)
        result: dict[str, ComponentSchema] = {}
        for name, fields in components.items():
            validate_component_name(name, source=cls._source)
            kind = fields.get("kind")
            model = (
                component_registry.get(kind, source=cls._source, component=name)
                if isinstance(kind, str)
                else ComponentSchema
            )
            # Validators may mutate mapping inputs; keep the caller isolated.
            adapter = TypeAdapter[dict[str, ComponentSchema]](dict[str, model])
            result[name] = adapter.validate_python({name: deepcopy(fields)})[name]
        return result

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata(cls, value: object) -> dict[str, YamlMap]:
        return TypeAdapter(dict[str, YamlMap]).validate_python(value, strict=True)


class SetupDocument(_ParameterDocument):
    """Entry identity and complete component template in working units.

    schema_version is the caller policy's current whole-component version,
    distinct from the container format_version. Missing or noncurrent versions
    fail at model_validate with a path and upgrade command, before kind lookup.
    """

    general: SetupGeneral


class PointDocument(_ParameterDocument):
    """One complete, independent point document in working units.

    format and format_version are the parameter-container header, written as 2.0.
    schema_version is the strict positive integer from the caller's registered
    ParameterSchemaPolicy, independent of that container version. Runtime reads
    require the current version and never classify or upgrade legacy documents.
    The store validates header compatibility. Values remain in working units;
    general is point-local.
    components maps public identifiers to original registered ComponentSchema models,
    including kind; each original model owns validation. Schema loading does not
    reserve view members; binding a view checks those namespace collisions. No value or source falls back to setup.
    metadata maps logical dotted paths to opaque caller YAML mappings.
    Component and nested models choose extra policy at every minor version:
    forbid rejects unknown fields, ignore discards them, and allow retains them.
    Forward-minor preservation applies only to unknown container fields.
    """

    _source: ClassVar[Path] = Path("point.yaml")
    general: PointGeneral
