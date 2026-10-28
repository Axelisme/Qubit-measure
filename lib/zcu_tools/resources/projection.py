"""Memory-only navigation for owner-controlled document projections."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import TypeGuard, get_args, get_origin

import numpy as np
from pydantic import BaseModel, TypeAdapter
from pydantic_core import (
    PydanticUndefined,
    SchemaSerializer,
    SchemaValidator,
    core_schema,
)

from zcu_tools.format_version import YamlValue
from zcu_tools.resources.parameter_fields import (
    mapping_shape,
    model_field_schema,
    sequence_shape,
)


class ProjectionPathError(AttributeError):
    """A dotted projection cannot read segment from its requested path.

    segment is the first missing key, invalid list index or child of a scalar.
    Owners may translate this error to their public path-error representation.
    """

    def __init__(self, segment: str) -> None:
        self.segment = segment
        super().__init__(f"segment {segment!r} is not readable")


def _model_schema(schema: core_schema.CoreSchema) -> TypeGuard[core_schema.ModelSchema]:
    return schema["type"] == "model"


def _tagged_schema(
    schema: core_schema.CoreSchema,
) -> TypeGuard[core_schema.TaggedUnionSchema]:
    return schema["type"] == "tagged-union"


def _definitions_schema(
    schema: core_schema.CoreSchema,
) -> TypeGuard[core_schema.DefinitionsSchema]:
    return schema["type"] == "definitions"


def _reference_schema(
    schema: core_schema.CoreSchema,
) -> TypeGuard[core_schema.DefinitionReferenceSchema]:
    return schema["type"] == "definition-ref"


def _wrapped_schema(
    schema: core_schema.CoreSchema,
) -> TypeGuard[
    core_schema.WithDefaultSchema
    | core_schema.NullableSchema
    | core_schema.BeforeValidatorFunctionSchema
    | core_schema.AfterValidatorFunctionSchema
    | core_schema.WrapValidatorFunctionSchema
]:
    return schema["type"] in (
        "default",
        "nullable",
        "function-before",
        "function-after",
        "function-wrap",
    )


def _projection_model(annotation: object, value: object) -> type[BaseModel] | None:
    schema = TypeAdapter(annotation).core_schema
    definitions: dict[str, core_schema.CoreSchema] = {}
    seen: set[str] = set()
    while True:
        if _definitions_schema(schema):
            definitions.update(
                {item["ref"]: item for item in schema["definitions"] if "ref" in item}
            )
            schema = schema["schema"]
        elif _wrapped_schema(schema):
            schema = schema["schema"]
        elif _reference_schema(schema):
            reference = schema["schema_ref"]
            if reference in seen or reference not in definitions:
                raise ValueError(f"Unresolved projection schema: {reference}")
            seen.add(reference)
            schema = definitions[reference]
        elif _tagged_schema(schema) and isinstance(schema["discriminator"], str):
            try:
                tag = projection_child(value, schema["discriminator"])
            except ProjectionPathError as exc:
                raise ValueError(
                    f"Missing projection discriminator: {exc.segment}"
                ) from exc
            if not isinstance(tag, (str, int)) or tag not in schema["choices"]:
                raise ValueError(f"Invalid projection discriminator: {tag!r}")
            schema = schema["choices"][tag]
        elif _model_schema(schema) and issubclass(schema["cls"], BaseModel):
            return schema["cls"]
        else:
            return None


def _field_input(
    value: object, model: type[BaseModel], name: str
) -> tuple[object, core_schema.CoreSchema, bool]:
    schema = model_field_schema(model, name)
    if isinstance(value, BaseModel):
        return getattr(value, name), schema, True
    if not isinstance(value, Mapping):
        raise ValueError(f"{model.__name__}: expected a mapping")
    if name in value:
        return value[name], schema, False
    default = SchemaValidator(schema).validate_python(PydanticUndefined)
    return default, schema, True


def typed_projection(
    value: object, annotation: object, segments: tuple[str, ...]
) -> object:
    """Project one schema-owned subtree without validating unselected siblings.

    annotation describes the current model/container. Selected complete models
    retain their validators; selected fields retain field rules. Discriminated
    unions inspect only their tag before choosing the model. Invalid keys/indices
    raise ProjectionPathError; malformed used values raise schema errors.
    """
    if not segments:
        validated = TypeAdapter(annotation).validate_python(deepcopy(value))
        return TypeAdapter(annotation).dump_python(validated, serialize_as_any=True)
    segment, *suffix = segments
    model = _projection_model(annotation, value)
    if model is not None:
        if segment not in model.model_fields or model.model_fields[segment].exclude:
            raise ProjectionPathError(segment)
        child, schema, validated = _field_input(value, model, segment)
        if suffix:
            return typed_projection(
                child, model.model_fields[segment].rebuild_annotation(), tuple(suffix)
            )
        if not validated:
            child = SchemaValidator(schema).validate_python(deepcopy(child))
        return SchemaSerializer(schema).to_python(
            child, mode="python", serialize_as_any=True
        )
    child = projection_child(value, segment)
    args = get_args(annotation)
    child_type = (
        args[1]
        if get_origin(annotation) is dict and len(args) == 2
        else args[0]
        if get_origin(annotation) in (list, tuple) and args
        else YamlValue
    )
    return typed_projection(child, child_type, tuple(suffix))


def typed_summary(
    value: object, annotation: object, describe: Callable[[object], object]
) -> object:
    """Project emitted model/scalar fields while describing sequence structure.

    Sequence items are not projected. Their owner field rules and container
    constraints still apply. Unknown or malformed needed fields raise; models
    are traversed by field schema, without cross-field model validation.
    """
    model = _projection_model(annotation, value)
    if model is not None:
        result: dict[str, object] = {}
        for name, field in model.model_fields.items():
            if field.exclude:
                continue
            child, schema, validated = _field_input(value, model, name)
            if isinstance(child, (list, tuple)):
                result[name] = describe(sequence_shape(child, schema))
            elif isinstance(child, (dict, BaseModel)):
                result[name] = typed_summary(
                    child, field.rebuild_annotation(), describe
                )
            else:
                if not validated:
                    child = SchemaValidator(schema).validate_python(deepcopy(child))
                result[name] = SchemaSerializer(schema).to_python(child, mode="python")
        return result
    if isinstance(value, Mapping):
        mapping_shape(value, TypeAdapter(annotation).core_schema)
        if "__complex__" in value:
            return deepcopy(value)
        args = get_args(annotation)
        child_type = (
            args[1] if get_origin(annotation) is dict and len(args) == 2 else YamlValue
        )
        return {
            key: typed_summary(child, child_type, describe)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return describe(sequence_shape(value, TypeAdapter(annotation).core_schema))
    return TypeAdapter(annotation).validate_python(value)


def projection_child(value: object, segment: str) -> object:
    """Read one mapping/model key or canonical nonnegative sequence index.

    No schema validation, serialization or I/O occurs. The returned child remains
    owner-managed. Missing keys and invalid indices raise ProjectionPathError.
    """
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, BaseModel):
        if (
            segment in type(value).model_fields
            and not type(value).model_fields[segment].exclude
        ):
            return getattr(value, segment)
        if segment in (value.model_extra or {}):
            return (value.model_extra or {})[segment]
    elif segment and isinstance(value, Mapping) and segment in value:
        return value[segment]
    elif (
        isinstance(value, (list, tuple))
        and segment.isascii()
        and segment.isdecimal()
        and (segment == "0" or not segment.startswith("0"))
        and len(segment) <= len(str(len(value)))
        and int(segment) < len(value)
    ):
        return value[int(segment)]
    raise ProjectionPathError(segment)


def projection_value(value: object, segments: tuple[str, ...]) -> object:
    """Detach a subtree after reading only its ancestors, with no validation.

    Owners validate the selected subtree according to their projection contract.
    Empty segments select the whole value. Invalid paths raise ProjectionPathError
    for the first bad segment. Models retain all subclass fields in the result.
    """
    for segment in segments:
        value = projection_child(value, segment)
    return deepcopy(TypeAdapter(object).dump_python(value, serialize_as_any=True))
