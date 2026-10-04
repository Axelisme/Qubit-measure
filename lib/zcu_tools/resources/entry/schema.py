"""Typed setup document at the persistence boundary."""

import keyword
import math
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, ClassVar
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

    @field_validator("ch", "ro_ch", "flux_ch", mode="before")
    @classmethod
    def reject_null_channels(cls, value: object) -> object:
        if value is None:
            raise ValueError("a supplied wiring channel must be a non-negative integer")
        return value


class ComponentSchema(BaseModel):
    """Notebook declaration with separate partial and complete validation phases.

    Partial setup validates supplied fields, including field-validator conversions.
    Missing fields do not run validators. Field conversions must be idempotent
    under canonical equality. Declared UnitSpec finite-float leaves use relative
    tolerance 1e-12 with no absolute tolerance; other values compare exactly.
    Entry revalidates before commits and snapshot publication, reporting canonical
    drift as ValidationError. Registration does not trial sample inputs.
    Model after-validators run only against
    complete layered views and may check values but must not change them.
    Registration rejects model before/wrap validators and custom model_post_init.
    Cross-field constraints belong in model after-validators: field validators
    reading info.data are unsupported in partial setup; their errors propagate.
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


def _same_canonical_value(
    before: YamlValue, after: YamlValue, field: FieldInfo
) -> bool:
    if (
        isinstance(before, float)
        and isinstance(after, float)
        and math.isfinite(before)
        and math.isfinite(after)
        and any(isinstance(metadata, UnitSpec) for metadata in field.metadata)
    ):
        return math.isclose(before, after, rel_tol=1e-12, abs_tol=0.0)
    return before == after


def _canonical_errors(
    fields: YamlMap, validated: BaseModel, path: FieldPath, *, source: Path
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
                _canonical_errors(before, nested, (*path, name), source=source)
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


class SetupDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    _source: ClassVar[Path] = Path("setup.yaml")

    format: str
    format_version: str
    general: SetupGeneral
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
                component_registry.partial_model(kind)
                if isinstance(kind, str)
                else ComponentSchema
            )
            # Field validators may mutate mapping inputs in place. Keep the
            # pre-validation working values for the canonical comparison.
            result[name] = model.model_validate(
                deepcopy(fields), extra="ignore" if forward_minor else None
            )
            errors = _canonical_errors(
                fields, result[name], (name,), source=cls._source
            )
            if errors:
                raise ValidationError.from_exception_data(model.__name__, errors)
        return result
