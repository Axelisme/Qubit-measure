"""Typed setup document at the persistence boundary."""

import keyword
import math
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from types import UnionType
from typing import Annotated, ClassVar, Union, get_args, get_origin
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_validator,
)
from pydantic.fields import FieldInfo
from pydantic_core import InitErrorDetails, PydanticCustomError

from zcu_tools.format_version import FormatVersion, YamlMap, YamlValue, validate_header
from zcu_tools.resources.document_store import FieldPath, UnitSpec

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
    ext is an arbitrary YAML mapping, empty by default, with no unit conversion.
    Unknown metadata fields and invalid timestamps raise Pydantic ValidationError.
    """

    model_config = ConfigDict(extra="forbid")

    created_at: str
    description: str | None = None
    ext: YamlMap = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return SetupGeneral.validate_created_at(value)


class WiringSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ch: int | None = Field(default=None, ge=0, strict=True)
    ro_ch: int | None = Field(default=None, ge=0, strict=True)
    flux_ch: int | None = Field(default=None, ge=0, strict=True)
    time_of_flight: Annotated[float | None, UnitSpec("s", "us")] = Field(
        default=None, ge=0
    )

    @field_validator("ch", "ro_ch", "flux_ch", mode="before")
    @classmethod
    def reject_null_channels(cls, value: object) -> object:
        if value is None:
            raise ValueError("a supplied wiring channel must be a non-negative integer")
        return value


class ComponentSchema(BaseModel):
    """Original notebook model for complete setup and point components.

    Required fields, defaults, factories and field validators use Pydantic
    semantics. Field conversions must be idempotent under canonical equality.
    Declared UnitSpec finite-float leaves use relative tolerance 1e-12 with no
    absolute tolerance; other values compare exactly. Entry reports canonical
    drift as ValidationError before commits or snapshot publication.
    Registration does not trial sample inputs. All model-level validators and
    custom model_post_init are rejected, including inherited and direct or
    nullable nested models. Cross-field checks are unsupported in this batch.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str
    wiring: WiringSchema = Field(default_factory=WiringSchema)
    ext: YamlMap = Field(default_factory=dict)


class ResonatorSchema(ComponentSchema):
    freq: Annotated[float | None, UnitSpec("Hz", "MHz")] = None
    kappa: Annotated[float | None, UnitSpec("Hz", "MHz")] = None
    amplifier: str | None = None


class QubitSchema(ComponentSchema):
    freq: Annotated[float | None, UnitSpec("Hz", "MHz")] = None
    EJ: Annotated[float | None, UnitSpec("Hz", "GHz")] = None
    EC: Annotated[float | None, UnitSpec("Hz", "GHz")] = None
    pi_len: Annotated[float | None, UnitSpec("s", "us")] = None
    t1: Annotated[float | None, UnitSpec("s", "us")] = None
    t2: Annotated[float | None, UnitSpec("s", "us")] = None
    pi_gain: Annotated[float | None, UnitSpec("1", "1")] = None
    readout: str | None = None
    flux_source: str | None = None


class FluxoniumSchema(QubitSchema):
    EL: Annotated[float | None, UnitSpec("Hz", "GHz")] = None
    flux_half: Annotated[float | None, UnitSpec("A", "mA")] = None
    flux_period: Annotated[float | None, UnitSpec("A", "mA")] = None


class JpaSchema(ComponentSchema):
    freq: Annotated[float | None, UnitSpec("Hz", "MHz")] = None
    gain: Annotated[float | None, UnitSpec("1", "1")] = None
    current: Annotated[float | None, UnitSpec("A", "mA")] = None


class CurrentSourceSchema(ComponentSchema):
    current: Annotated[float | None, UnitSpec("A", "mA")] = None


def field_annotations(
    field: FieldInfo,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    annotation = field.annotation
    alternatives = (
        get_args(annotation)
        if get_origin(annotation) in (Union, UnionType)
        else (annotation,)
    )
    types: list[object] = []
    metadata: list[object] = list(field.metadata)
    for alternative in alternatives:
        if get_origin(alternative) is Annotated:
            alternative, *branch_metadata = get_args(alternative)
            metadata.extend(branch_metadata)
        types.append(alternative)
    return tuple(types), tuple(metadata)


def _same_canonical_value(
    before: YamlValue, after: YamlValue, field: FieldInfo
) -> bool:
    if (
        isinstance(before, float)
        and isinstance(after, float)
        and math.isfinite(before)
        and math.isfinite(after)
        and any(
            isinstance(metadata, UnitSpec) for metadata in field_annotations(field)[1]
        )
    ):
        return math.isclose(before, after, rel_tol=1e-12, abs_tol=0.0)
    return before == after


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
    for name, field in type(validated).model_fields.items():
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
        elif (name in fields) != (name in canonical) or not _same_canonical_value(
            before, after, field
        ):
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
            kind = fields.get("kind")
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
