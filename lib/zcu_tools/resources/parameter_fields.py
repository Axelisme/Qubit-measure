"""Live field containers for caller-owned parameter models.

Reads follow the owning memory snapshot. Assignments edit detached candidates;
the supplied edit context owns full-model validation and value publication.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from copy import deepcopy
from typing import TypeGuard

from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_core import SchemaValidator, core_schema

from zcu_tools.core.complex_codec import decode_complex, encode_complex
from zcu_tools.format_version import YamlValue

from .parameter_metadata import parameter_path

# Logical field containers retain typed models inside mappings. No list indexing.
type FieldNode = BaseModel | dict[str, object]


def _definitions(
    schema: core_schema.CoreSchema,
) -> TypeGuard[core_schema.DefinitionsSchema]:
    return schema["type"] == "definitions"


def _model_fields(
    schema: core_schema.CoreSchema,
) -> TypeGuard[core_schema.ModelFieldsSchema]:
    return schema["type"] == "model-fields"


def _model_wrapper(
    schema: core_schema.CoreSchema,
) -> TypeGuard[
    core_schema.ModelSchema
    | core_schema.BeforeValidatorFunctionSchema
    | core_schema.AfterValidatorFunctionSchema
    | core_schema.WrapValidatorFunctionSchema
]:
    # Pydantic's schema TypedDicts mark the discriminator optional, so direct
    # discriminator checks do not narrow their union in Pyright.
    return schema["type"] in (
        "model",
        "function-before",
        "function-after",
        "function-wrap",
    )


def model_field_schema(model: type[BaseModel], name: str) -> core_schema.CoreSchema:
    """Return one field's schema, including its defaults and field validators.

    Model-level validators are excluded. Unknown fields raise AttributeError.
    Definitions required by nested/recursive field models remain available.
    No values or unrelated fields are validated.
    """
    schema = model.__pydantic_core_schema__
    definitions: list[core_schema.CoreSchema] = []
    while True:
        if _definitions(schema):
            definitions.extend(schema["definitions"])
            schema = schema["schema"]
        elif _model_wrapper(schema):
            schema = schema["schema"]
        elif _model_fields(schema):
            if name not in schema["fields"]:
                raise AttributeError(name)
            field = schema["fields"][name]["schema"]
            return (
                core_schema.definitions_schema(field, definitions)
                if definitions
                else field
            )
        else:
            raise TypeError(f"{model.__name__}: unsupported model field schema")


def _list_schema(schema: core_schema.CoreSchema) -> TypeGuard[core_schema.ListSchema]:
    return schema["type"] == "list"


def _field_wrapper(
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


def _sequence_shape_schema(schema: core_schema.CoreSchema) -> core_schema.CoreSchema:
    if _list_schema(schema):
        return {**schema, "items_schema": core_schema.any_schema()}
    if _field_wrapper(schema):
        wrapped = deepcopy(schema)
        wrapped["schema"] = _sequence_shape_schema(schema["schema"])
        return wrapped
    return schema


def sequence_shape(value: object, schema: core_schema.CoreSchema) -> object:
    """Validate a described list's container and field rules, not item schemas.

    schema is the owning field's core schema, including defaults and field-level
    validators. List annotations keep their length/type constraints; scalar
    annotations still reject lists. Returned items are not projected or sent on
    wire. Field rules may inspect them when that is necessary to describe shape.
    Malformed used structure raises its original validation error.
    """
    return SchemaValidator(_sequence_shape_schema(schema)).validate_python(
        deepcopy(value)
    )


def _dict_schema(schema: core_schema.CoreSchema) -> TypeGuard[core_schema.DictSchema]:
    return schema["type"] == "dict"


def _mapping_shape_schema(schema: core_schema.CoreSchema) -> core_schema.CoreSchema:
    if _dict_schema(schema):
        return {**schema, "values_schema": core_schema.any_schema()}
    if _field_wrapper(schema):
        wrapped = deepcopy(schema)
        wrapped["schema"] = _mapping_shape_schema(schema["schema"])
        return wrapped
    return schema


def mapping_shape(value: object, schema: core_schema.CoreSchema) -> object:
    """Validate mapping structure and field rules without consuming child values.

    schema is the owning field schema. Mapping keys and container constraints
    remain checked; scalar annotations reject mappings. The owner then projects
    individual output children. Malformed used structure raises schema errors.
    """
    return SchemaValidator(_mapping_shape_schema(schema)).validate_python(
        deepcopy(value)
    )


def model_field_value(node: BaseModel, name: str) -> object:
    """Return a model's declared field or extra key without copying or hooks.

    node is a cached model; name is one literal key. Missing extras return None.
    Declared fields use their original attribute read; its errors propagate.
    No mutation, validation, serialization or I/O occurs.
    """
    if name in type(node).model_fields:
        return getattr(node, name)
    return (node.model_extra or {}).get(name)


def read_field(node: FieldNode, name: str, path: str) -> object:
    """Return one logical field/key from a model or mapping without copying.

    name is a literal key; path is the parent path used in error messages.
    Missing/unset model fields raise AttributeError; missing mapping keys raise
    KeyError. Explicit null is readable when set. No mutation, model validation,
    serialization or I/O occurs; returned containers remain owner-managed.
    """
    if isinstance(node, BaseModel):
        if name not in type(node).model_fields and name not in (node.model_extra or {}):
            raise AttributeError(f"{path}.{name}: field is not set")
        value = model_field_value(node, name)
        if name not in node.model_fields_set and value is None:
            raise AttributeError(f"{path}.{name}: field is not set")
        return value
    return node[name]


def field_container(value: object, path: str) -> FieldNode:
    """Return an existing model/mapping container, or raise AttributeError.

    value is a cached logical field and path identifies it in diagnostics.
    Returned nodes retain their model types and are not copied or serialized.
    Callers must use detached candidates for writes.
    """
    if isinstance(value, (BaseModel, dict)):
        return value
    raise AttributeError(f"{path}: not a field container")


def _write_field(node: FieldNode, name: str, value: object, path: str) -> None:
    if (
        isinstance(node, BaseModel)
        and name not in type(node).model_fields
        and type(node).model_config.get("extra") != "allow"
    ):
        # Preserve the forbid error contract without invoking migration validators.
        raise ValidationError.from_exception_data(
            type(node).__name__,
            [
                {
                    "type": "extra_forbidden",
                    "loc": tuple(path.split(".")),
                    "input": value,
                }
            ],
        )
    if isinstance(value, complex):
        value = encode_complex(value)
    copied = deepcopy(value)
    if isinstance(node, BaseModel):
        if name in type(node).model_fields:
            setattr(node, name, copied)
        elif type(node).model_config.get("extra") == "allow":
            extras = node.model_extra
            if extras is None:
                raise ValueError(f"{name!r}: model has no extra field storage")
            extras[name] = copied
            node.model_fields_set.add(name)
    else:
        node[name] = copied


class FieldView:
    """Read live model/mapping fields and route writes to an owning edit scope.

    Scalars and lists are detached YAML values. Nested models/mappings return
    child views following the same memory reader and edit callback. Complex
    tags decode to native complex scalars. Missing/unset fields and malformed
    tags raise rather than substituting defaults. Reads never perform I/O.
    Writes to undeclared model fields raise ValidationError with the full path
    unless the model allows extras; this rejection does not run model validators.
    """

    _model: Callable[[], FieldNode]
    _edit: Callable[[str], AbstractContextManager[FieldNode]]
    _path: str

    def __init__(
        self,
        model: Callable[[], FieldNode],
        edit: Callable[[str], AbstractContextManager[FieldNode]],
        path: str,
    ) -> None:
        """Bind a container to memory reads and atomic candidate edits.

        model returns the latest container without I/O. edit(relative_path)
        yields an independent candidate and validates the complete owning model
        on normal exit. Rejection leaves values and metadata unchanged. path is
        the full logical container path used in errors.
        """
        self._model = model
        self._edit = edit
        self._path = path

    def __getattr__(self, name: str) -> YamlValue | complex | FieldView:
        """Read a field/key with complex tags decoded, without I/O.

        Missing/unset fields raise AttributeError; malformed tags raise the
        core codec ValueError or OverflowError.
        """
        try:
            return self[name]
        except KeyError as cause:
            raise AttributeError(f"{self._path}.{name}: field is not set") from cause

    def __getitem__(self, name: str) -> YamlValue | complex | FieldView:
        """Read a literal key, decoding a complex tag to a native scalar.

        Other dict/model values return live child views. Scalars and lists are
        independent YAML values; lists retain their YAML elements. Malformed
        tagged mappings raise the core codec ValueError or OverflowError.
        Missing keys raise KeyError; unset null-default model fields raise
        AttributeError. Empty or dotted keys cannot be addressed by set/meta.
        """
        value = read_field(self._model(), name, self._path)
        if isinstance(value, Mapping) and "__complex__" in value:
            return decode_complex(value)
        if isinstance(value, (BaseModel, dict)):
            return FieldView(
                lambda: field_container(
                    read_field(self._model(), name, self._path), f"{self._path}.{name}"
                ),
                lambda field: self._edit_child(name, field),
                f"{self._path}.{name}",
            )
        serialized = TypeAdapter(object).dump_python(value, serialize_as_any=True)
        return deepcopy(TypeAdapter(YamlValue).validate_python(serialized))

    @contextmanager
    def _edit_child(self, name: str, field: str) -> Generator[FieldNode]:
        with self._edit(f"{name}.{field}") as parent:
            child = field_container(
                read_field(parent, name, self._path), f"{self._path}.{name}"
            )
            if isinstance(parent, BaseModel):
                parent.model_fields_set.add(name)
            yield child

    def __setattr__(self, name: str, value: object) -> None:
        """Encode a native complex scalar, then validate the complete owning model."""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self[name] = value

    def __setitem__(self, name: str, value: object) -> None:
        """Encode a native complex scalar and copy into an atomic candidate.

        The core codec supplies the persisted tag. The owning edit callback
        validates the complete model; failures reject values and metadata.
        Container assignments retain the existing YAML/model contract. Undeclared
        model fields are rejected with a full-path ValidationError before model
        validation unless the model allows extras.
        """
        with self._edit(name) as draft:
            _write_field(draft, name, value, ".".join(filter(None, (self._path, name))))


@contextmanager
def stage_model[T: BaseModel](model: T, publish: Callable[[T], None]) -> Generator[T]:
    """Edit an independent candidate and publish its validated complete model.

    model is the current caller-owned value model. publish replaces that model
    in a detached document draft after successful body and full-model validation.
    Original model validators, serializers and extra policy apply. Body or
    validation errors leave the original model unchanged. No metadata is read
    or changed, and no I/O occurs. DocumentStore owns the final transaction.
    """
    candidate = model.model_copy(deep=True)
    yield candidate
    accepted = type(model).model_validate(
        candidate.model_dump(exclude_unset=True, serialize_as_any=True, warnings=False)
    )
    publish(accepted)


def sequence_index(value: Sequence[object], segment: str) -> int:
    """Decode an existing canonical nonnegative ASCII decimal sequence index.

    value supplies only the length; this does not choose readable or writable
    container types. segment must be "0" or digits without leading zero, and
    identify an existing element. Invalid syntax, overly long numbers and
    out-of-range indices raise ValueError before integer conversion.
    """
    if (
        not segment.isascii()
        or not segment.isdecimal()
        or (segment != "0" and segment.startswith("0"))
        or len(segment) > len(str(len(value)))
    ):
        raise ValueError(f"Invalid sequence index: {segment!r}")
    index = int(segment)
    if index >= len(value):
        raise ValueError(f"Sequence index out of range: {segment!r}")
    return index


def set_model_value(model: BaseModel, path: str, value: object) -> None:
    """Assign one dotted field in an independent candidate, without validation.

    model must be a detached candidate managed by stage_model or a DocumentStore
    draft. path uses nonempty model/mapping segments or canonical nonnegative
    ASCII indices through existing lists. Numeric mapping keys remain keys.
    Lists allow existing elements only: no auto-create, append, insert, slice
    or negative indexing. Tuple/ndarray writes are not supported.
    Native complex scalars use the shared persisted tag. Values are copied;
    model types and field presence are retained. Missing/non-container parents
    raise KeyError/AttributeError. An undeclared field on a model that does not
    allow extras raises ValidationError naming the full path, without invoking
    model validators. The owner must validate the full model before publication.
    No metadata or files are changed.
    """
    parts = parameter_path(path)
    node: FieldNode | list[object] = model
    for index, name in enumerate(parts[:-1]):
        parent_path = ".".join(parts[:index])
        if isinstance(node, list):
            child = node[sequence_index(node, name)]
        else:
            child = read_field(node, name, parent_path)
        if isinstance(node, BaseModel):
            node.model_fields_set.add(name)
        node = (
            child
            if isinstance(child, list)
            else field_container(child, ".".join(parts[: index + 1]))
        )
    if isinstance(node, list):
        node[sequence_index(node, parts[-1])] = deepcopy(
            encode_complex(value) if isinstance(value, complex) else value
        )
    else:
        _write_field(node, parts[-1], value, path)
