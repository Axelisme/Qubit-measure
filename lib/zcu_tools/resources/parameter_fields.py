"""Live field containers for caller-owned parameter models.

Reads follow the owning memory snapshot. Assignments edit detached candidates;
the supplied edit context owns full-model validation and value publication.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from copy import deepcopy

from pydantic import BaseModel, TypeAdapter, ValidationError

from zcu_tools.core.complex_codec import decode_complex, encode_complex
from zcu_tools.format_version import YamlValue

from .parameter_metadata import parameter_path

# Logical field containers retain typed models inside mappings. No list indexing.
type FieldNode = BaseModel | dict[str, object]


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


def set_model_value(model: BaseModel, path: str, value: object) -> None:
    """Assign one dotted field in an independent candidate, without validation.

    model must be a detached candidate managed by stage_model or a DocumentStore
    draft. path uses nonempty mapping/model segments, with no list indexing.
    Native complex scalars use the shared persisted tag. Values are copied;
    model types and field presence are retained. Missing/non-container parents
    raise KeyError/AttributeError. An undeclared field on a model that does not
    allow extras raises ValidationError naming the full path, without invoking
    model validators. The owner must validate the full model before publication.
    No metadata or files are changed.
    """
    parts = parameter_path(path)
    node: FieldNode = model
    for index, name in enumerate(parts[:-1]):
        parent_path = ".".join(parts[:index])
        child = read_field(node, name, parent_path)
        if isinstance(node, BaseModel):
            node.model_fields_set.add(name)
        node = field_container(child, ".".join(parts[: index + 1]))
    _write_field(node, parts[-1], value, path)
