"""Typed setup document at the persistence boundary."""

import keyword
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, ClassVar, Literal, Self, override
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationInfo,
    field_validator,
)

from zcu_tools.format_version import FormatVersion, YamlMap, YamlValue, validate_header

PARAMETER_FORMAT = "zcu.parameter-container"
PARAMETER_VERSION = FormatVersion(1, 0)


def is_forward_minor(document: Mapping[str, YamlValue], *, source: Path) -> bool:
    version = validate_header(
        document,
        expected_format=PARAMETER_FORMAT,
        supported_version=PARAMETER_VERSION,
        source=source,
    )
    return version.minor > PARAMETER_VERSION.minor


_VIEW_NAMES = frozenset(
    {
        "description",
        "edit",
        "refresh",
        "add_component",
        "general",
        "meta",
        "set",
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
    flux_unit: Literal["A", "V"] | None = None
    flux_value: Annotated[float | None, Field(strict=True, allow_inf_nan=False)] = None
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
    """Metadata belonging to one point, never inherited from setup.

    created_at is an ISO-8601 timestamp with a UTC offset; Z and +00:00 are valid.
    description is optional human-readable text; None means no description.
    ext is a YAML mapping, empty by default, with no unit conversion.
    Empty keys or keys containing dots cannot be addressed by set/meta.
    Global flux_value uses flux_unit (A or V); both are optional.
    Unknown metadata fields and invalid timestamps raise Pydantic ValidationError.
    """

    model_config = ConfigDict(extra="forbid")

    created_at: str
    description: str | None = None
    flux_unit: Literal["A", "V"] | None = None
    flux_value: Annotated[float | None, Field(strict=True, allow_inf_nan=False)] = None
    ext: YamlMap = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return SetupGeneral.validate_created_at(value)


class ComponentSchema(BaseModel):
    """Base for registered components, requiring only a string kind.

    Definitions own extra policy, validators, hooks and optional containers.
    Loading uses model_validate and saving uses model_dump. Mapping keys with
    dots or an empty name cannot be addressed by set/meta dotted paths.
    """

    kind: str


class _ParameterDocument(BaseModel):
    """Shared typed component boundary for complete parameter documents."""

    model_config = ConfigDict(extra="forbid")
    _source: ClassVar[Path] = Path("setup.yaml")

    format: str
    format_version: str
    components: dict[str, ComponentSchema] = Field(default_factory=dict)
    provenance: dict[str, YamlMap] = Field(default_factory=dict)

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
    def validate_components(
        cls, value: object, info: ValidationInfo
    ) -> dict[str, ComponentSchema]:
        # Import locally because registered models derive from ComponentSchema.
        from .registry import component_registry

        # This dynamic dispatch starts a separate validation call, so propagate the
        # document's future-field policy explicitly into each registered model.
        forward_minor = is_forward_minor(
            {
                "format": info.data["format"],
                "format_version": info.data["format_version"],
            },
            source=cls._source,
        )
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
            result[name] = adapter.validate_python(
                {name: deepcopy(fields)}, extra="allow" if forward_minor else None
            )[name]
        return result


class SetupDocument(_ParameterDocument):
    """Entry identity and complete component template in working units."""

    general: SetupGeneral


class PointDocument(_ParameterDocument):
    """One complete, independent point document in working units.

    format and format_version are the parameter-container header, initially 1.0.
    The store validates header compatibility. Values remain in working units;
    general is point-local.
    components maps public names to original registered ComponentSchema models,
    including kind; each original model owns validation. No value or source falls back to setup.
    provenance maps logical component-field paths to YAML source metadata.
    Component models choose extra policy. Forward-minor unknown fields remain
    on disk; only extra=allow models expose them in the typed view.
    """

    _source: ClassVar[Path] = Path("point.yaml")
    general: PointGeneral
