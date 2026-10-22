"""Shared lab field types, channel wiring and component storage fields."""

import logging
from collections.abc import Mapping
from copy import deepcopy
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from zcu_tools.format_version import YamlMap
from zcu_tools.resources.entry.schema import ComponentSchema

_LOGGER = logging.getLogger(__name__)

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


def reject_null_channel(value: object) -> object:
    """Reject explicit null channel input; omitted defaults remain unset."""
    if value is None:
        raise ValueError("a supplied wiring channel must be a non-negative integer")
    return value


class WiringSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ch: int | None = Field(
        default=None, ge=0, strict=True, description="Generator (output) channel index."
    )
    ro_ch: int | None = Field(
        default=None, ge=0, strict=True, description="Readout (ADC) channel index."
    )

    reject_null_channels = field_validator("ch", "ro_ch", mode="before")(
        reject_null_channel
    )


class BuiltinComponent(ComponentSchema):
    """Lab fields with unsupported input preserved under ext.migrated.

    Each validation moves undeclared fields, including fields in nested models,
    to their component-relative paths in ext.migrated. Incoming values overwrite
    the same migrated paths; other ext content is kept. A single warning lists
    moved paths. Declared fields retain their normal validation errors.
    """

    model_config = ConfigDict(extra="forbid")

    ext: YamlMap = Field(
        default_factory=dict, description="Extra values without a fixed schema."
    )
    module: dict[str, str] = Field(
        default_factory=dict,
        description="Module slots: slot name to a module library key.",
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_unsupported_fields(cls, value: object) -> object:
        """Preserve unknown mapping fields without mutating the caller's input.

        Non-mappings and invalid ext containers pass through for normal schema
        errors. Existing non-mapping migrated branches are replaced when a moved
        path needs a mapping there.
        """
        if not isinstance(value, Mapping):
            return value
        data: dict[str, object] = deepcopy(dict(value))
        ext = data.get("ext", {})
        if not isinstance(ext, dict):
            return value

        moved: list[tuple[tuple[str, ...], object]] = []
        _collect_unsupported_fields(cls, data, (), moved)
        if not moved:
            return value

        for path, item in moved:
            target = ext
            for segment in ("migrated", *path[:-1]):
                branch = target.get(segment)
                if not isinstance(branch, dict):
                    branch = {}
                    target[segment] = branch
                target = branch
            target[path[-1]] = item
        data["ext"] = ext
        _LOGGER.warning(
            "Moved unsupported %s fields to ext.migrated: %s",
            cls.__name__,
            ", ".join(".".join(path) for path, _ in moved),
        )
        return data


def _collect_unsupported_fields(
    model: type[BaseModel],
    data: dict[str, object],
    prefix: tuple[str, ...],
    moved: list[tuple[tuple[str, ...], object]],
) -> None:
    """Remove unknown fields and collect their paths from directly nested models.

    Builtin schemas nest model fields directly; dictionaries such as ext and
    module are open-key containers and are not traversed.
    """
    for name in list(data):
        path = (*prefix, name)
        field = model.model_fields.get(name)
        if field is None:
            moved.append((path, data.pop(name)))
            continue
        nested_model = field.annotation
        nested_data = data[name]
        if (
            isinstance(nested_model, type)
            and issubclass(nested_model, BaseModel)
            and isinstance(nested_data, Mapping)
        ):
            child: dict[str, object] = dict(nested_data)
            _collect_unsupported_fields(nested_model, child, path, moved)
            data[name] = child
