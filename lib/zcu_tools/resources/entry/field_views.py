"""Live field containers shared by setup, point, role and transaction views.

Reads follow the owning memory snapshot. Assignments edit detached candidates;
the supplied edit context owns full-model validation and provenance publication.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from copy import deepcopy

from pydantic import BaseModel, TypeAdapter

from zcu_tools.core.complex_codec import decode_complex, encode_complex
from zcu_tools.format_version import YamlValue

from .schema import ComponentSchema, PointGeneral, SetupGeneral

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


def _field_container(value: object, path: str) -> FieldNode:
    # Preserve models inside mappings rather than projecting them to YAML.
    if isinstance(value, (BaseModel, dict)):
        return value
    raise AttributeError(f"{path}: not a field container")


def _write_field(node: FieldNode, name: str, value: object) -> None:
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
            # The original model applies forbid/ignore, not a registry precheck.
            type(node).model_validate(
                {
                    **node.model_dump(exclude_unset=True, serialize_as_any=True),
                    name: copied,
                }
            )
    else:
        node[name] = copied


class FieldView:
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
        on normal exit. Rejection leaves values and sources unchanged. path is
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
                lambda: _field_container(
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
            child = _field_container(
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

        The core codec supplies the persisted tag. Original owning-model and
        source validation still apply; failures reject values and sources.
        Container assignments retain the existing YAML/model contract.
        """
        with self._edit(name) as draft:
            _write_field(draft, name, value)


class GeneralView(FieldView):
    _general_model: Callable[[], SetupGeneral | PointGeneral]
    _extension: FieldView

    def __init__(
        self,
        model: Callable[[], SetupGeneral | PointGeneral],
        edit: Callable[[str], AbstractContextManager[SetupGeneral | PointGeneral]],
    ) -> None:
        super().__init__(model, edit, "general")
        self._general_model = model

        @contextmanager
        def edit_extension(name: str) -> Generator[FieldNode]:
            with edit(f"ext.{name}") as draft:
                yield _field_container(draft.ext, "general.ext")
                draft.model_fields_set.add("ext")

        self._extension = FieldView(
            lambda: _field_container(model().ext, "general.ext"),
            edit_extension,
            "general.ext",
        )

    @property
    def ext(self) -> FieldView:
        return self._extension

    @property
    def description(self) -> str | None:
        return self._general_model().description

    @description.setter
    def description(self, value: str | None) -> None:
        self["description"] = value


class ComponentView:
    _model: Callable[[], ComponentSchema]
    _fields: FieldView

    def __init__(
        self,
        model: Callable[[], ComponentSchema],
        edit: Callable[[str], AbstractContextManager[ComponentSchema]],
        path: str,
    ) -> None:
        """Bind a component to memory reads and complete-model field edits.

        model supplies the latest snapshot. edit(relative_path) yields a detached
        candidate, validates it on normal exit, and accepts its value and source.
        path is the component name. Tagged mappings decode to native complex;
        other dict/model reads return live child views using the same callback.
        Scalars/lists are independent YAML values. No read performs I/O.
        Malformed tags propagate the core codec error.
        """
        self._model = model
        self._fields = FieldView(model, edit, path)

    @property
    def kind(self) -> str:
        """Return the component's declared kind."""
        return self._model().kind

    def __getattr__(self, name: str) -> YamlValue | complex | FieldView:
        """Read a field with complex tags decoded; missing/unset raises AttributeError.

        Malformed tags propagate the core codec ValueError or OverflowError.
        """
        return self._fields.__getattr__(name)

    def __setattr__(self, name: str, value: object) -> None:
        """Validate and accept a field and source, or leave both unchanged."""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._fields[name] = value
