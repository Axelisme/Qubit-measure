"""Typed setup document at the persistence boundary."""

import keyword
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, ClassVar, Literal, Self, override
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_validator,
)
from pydantic_core import InitErrorDetails, PydanticCustomError

from zcu_tools.format_version import FormatVersion, YamlMap, YamlValue, validate_header
from zcu_tools.resources.document_store import FieldPath

PARAMETER_FORMAT = "zcu.parameter-container"
PARAMETER_VERSION = FormatVersion(1, 0)


@dataclass(frozen=True)
class UnitSpec:
    """Schema annotation for a working unit; never validates or converts values."""

    unit: str


@dataclass(frozen=True)
class Ref:
    """Annotate a string field that names a component in the same document."""


@dataclass(frozen=True)
class ModuleSlot:
    """Annotate a mapping of slot names to module_cfg path strings."""


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


def _validate_extension_keys(value: YamlMap) -> YamlMap:
    errors: list[InitErrorDetails] = []

    def collect_errors(node: YamlValue, path: tuple[str | int, ...]) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                key_path = (*path, key)
                if not key or "." in key:
                    errors.append(
                        InitErrorDetails(
                            type=PydanticCustomError(
                                "extension_key",
                                "Extension key at {path} must be non-empty and contain no '.'",
                                {"path": repr(key_path)},
                            ),
                            loc=key_path,
                            input=key,
                        )
                    )
                collect_errors(child, key_path)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                collect_errors(child, (*path, index))

    collect_errors(value, ())
    if errors:
        raise ValidationError.from_exception_data("ext", errors)
    return value


_JSON_EXTENSIONS = TypeAdapter[YamlMap](
    Annotated[YamlMap, AfterValidator(_validate_extension_keys)],
    config=ConfigDict(strict=True, allow_inf_nan=False),
)


class SetupGeneral(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_id: str
    created_at: str
    description: str | None = None
    flux_unit: Literal["A", "V"] | None = None
    flux_value: Annotated[
        float | None, UnitSpec("A/V"), Field(strict=True, allow_inf_nan=False)
    ] = None
    ext: Annotated[YamlMap, BeforeValidator(_JSON_EXTENSIONS.validate_python)] = Field(
        default_factory=dict
    )

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
    ext is a JSON mapping, empty by default, with no unit conversion.
    Mapping keys at every depth must be non-empty strings without dots.
    Global flux_value uses flux_unit (A or V); both are optional.
    Unknown metadata fields and invalid timestamps raise Pydantic ValidationError.
    """

    model_config = ConfigDict(extra="forbid")

    created_at: str
    description: str | None = None
    flux_unit: Literal["A", "V"] | None = None
    flux_value: Annotated[
        float | None, UnitSpec("A/V"), Field(strict=True, allow_inf_nan=False)
    ] = None
    ext: Annotated[YamlMap, BeforeValidator(_JSON_EXTENSIONS.validate_python)] = Field(
        default_factory=dict
    )

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return SetupGeneral.validate_created_at(value)


class ComponentSchema(BaseModel):
    """Original notebook model for complete setup and point components.

    Required fields, defaults, factories and field validators use Pydantic
    semantics. Field conversions must be idempotent under exact value equality.
    Entry reports canonical drift as ValidationError before commits or snapshot
    publication. Ext accepts JSON values with non-empty, dot-free mapping keys
    at every depth, including mappings inside lists.
    Registration does not trial sample inputs. All model-level validators and
    custom model_post_init are rejected, including inherited and direct or
    nullable nested models. Cross-field checks are unsupported in this batch.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str
    ext: YamlMap = Field(default_factory=dict)

    @field_validator("ext", mode="before")
    @classmethod
    def validate_extension(cls, value: object) -> YamlMap:
        # Keep the container boundary when a user model redeclares the field.
        return _JSON_EXTENSIONS.validate_python(value)


def canonical_errors(
    fields: YamlMap,
    validated: BaseModel,
    path: FieldPath,
    *,
    source: Path,
) -> list[InitErrorDetails]:
    """Compare supplied known fields in working units, not the original user input."""
    canonical = TypeAdapter(YamlMap).validate_python(
        validated.model_dump(exclude_unset=True)
    )
    errors: list[InitErrorDetails] = []
    for name in type(validated).model_fields:
        if name not in fields and name not in canonical:
            continue
        before = fields.get(name, "(missing)")
        after = canonical.get(name, "(missing)")
        nested = getattr(validated, name)
        if isinstance(before, dict) and isinstance(nested, BaseModel):
            errors.extend(
                canonical_errors(
                    before,
                    nested,
                    (*path, name),
                    source=source,
                )
            )
        elif (name in fields) != (name in canonical) or before != after:
            errors.append(
                InitErrorDetails(
                    type=PydanticCustomError(
                        "canonical_value",
                        "{source}: field validation changed canonical value from {before} to {after}",
                        {
                            "source": str(source),
                            "before": repr(before),
                            "after": repr(after),
                        },
                    ),
                    loc=(*path, name),
                    input=before,
                )
            )
    return errors


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
            forward_minor = is_forward_minor(document, source=cls._source)
            components = TypeAdapter(dict[str, YamlMap]).validate_python(
                document.get("components", {})
            )
            for name, fields in components.items():
                validate_component_name(name, source=cls._source)
                kind = fields.get("kind")
                if isinstance(kind, str):
                    component_registry.get(kind, source=cls._source, component=name)
                    if not forward_minor:
                        component_registry.check_fields(kind, fields, path=name)
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
            if isinstance(kind, str) and not forward_minor:
                component_registry.check_fields(kind, fields, path=name)
            model = (
                component_registry.get(kind, source=cls._source, component=name)
                if isinstance(kind, str)
                else ComponentSchema
            )
            # Field validators may mutate mapping inputs in place. Keep the
            # pre-validation working values for the canonical comparison.
            adapter = TypeAdapter[dict[str, ComponentSchema]](dict[str, model])
            result[name] = adapter.validate_python(
                {name: deepcopy(fields)}, extra="ignore" if forward_minor else None
            )[name]
            errors = canonical_errors(fields, result[name], (name,), source=cls._source)
            if errors:
                raise ValidationError.from_exception_data(model.__name__, errors)
        return result


class SetupDocument(_ParameterDocument):
    """Entry identity and complete component template in working units."""

    general: SetupGeneral


class PointDocument(_ParameterDocument):
    """One complete, independent point document in working units.

    format and format_version are the parameter-container header, initially 1.0.
    The store validates header compatibility and converts known physical leaves
    between SI on disk and working units in snapshots. general is point-local.
    components maps public names to original registered ComponentSchema models,
    including kind; required fields, field validators and same-document
    references must be valid. No value or source falls back to setup.
    provenance maps logical component-field paths to YAML source metadata.
    Unknown fields are rejected at the current minor version; a forward minor
    keeps unknown YAML fields outside the typed view.
    """

    _source: ClassVar[Path] = Path("point.yaml")
    general: PointGeneral
