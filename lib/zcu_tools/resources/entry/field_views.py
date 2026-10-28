"""Entry-specific field containers using the shared parameter field view."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager

from zcu_tools.core.complex_codec import decode_complex
from zcu_tools.format_version import YamlValue
from zcu_tools.resources.parameter_fields import (
    FieldNode,
    FieldView,
    field_container,
    model_field_value,
    read_field,
)

from .schema import ComponentSchema, ParameterContent, PointGeneral, SetupGeneral

__all__ = (
    "ComponentView",
    "FieldNode",
    "FieldView",
    "GeneralView",
    "model_field_value",
    "read_field",
)


class GeneralView(FieldView):
    _general_model: Callable[[], SetupGeneral | PointGeneral]
    _extension: FieldView

    def __init__(
        self,
        model: Callable[[], SetupGeneral | PointGeneral],
        edit: Callable[[str], AbstractContextManager[SetupGeneral | PointGeneral]],
        *,
        content: Callable[[], ParameterContent] | None = None,
    ) -> None:
        super().__init__(model, edit, "general")
        self._general_model = model
        self._content_fields = (
            None if content is None else ContentFields(content, edit, "general")
        )

        @contextmanager
        def edit_extension(name: str) -> Generator[FieldNode]:
            with edit(f"ext.{name}") as draft:
                yield field_container(draft.ext, "general.ext")
                draft.model_fields_set.add("ext")

        self._extension = (
            FieldView(
                lambda: field_container(model().ext, "general.ext"),
                edit_extension,
                "general.ext",
            )
            if content is None
            else ContentFields(content, edit_extension, "general.ext")
        )

    def __getitem__(self, name: str) -> YamlValue | complex | FieldView:
        if self._content_fields is not None:
            return self._content_fields[name]
        return super().__getitem__(name)

    @property
    def ext(self) -> FieldView:
        return self._extension

    @property
    def description(self) -> str | None:
        value = self["description"]
        if value is not None and not isinstance(value, str):
            raise ValueError("general.description must be a string or null")
        return value

    @description.setter
    def description(self, value: str | None) -> None:
        self["description"] = value


class ContentFields(FieldView):
    """Navigate captured resource content while validating only consumed leaves.

    content supplies cached content without I/O. edit retains the existing full
    model transaction. path is the concrete component/general field prefix.
    """

    def __init__(
        self,
        content: Callable[[], ParameterContent],
        edit: Callable[[str], AbstractContextManager[FieldNode]],
        path: str,
    ) -> None:
        super().__init__(lambda: {}, edit, path)
        self._content = content

    def __getitem__(self, name: str) -> YamlValue | complex | FieldView:
        """Read a literal field; mappings remain navigable without sibling validation."""
        path = f"{self._path}.{name}"
        try:
            node = self._content().node(path)
        except AttributeError as exc:
            raise KeyError(name) from exc
        if isinstance(node, dict) and "__complex__" not in node:
            return ContentFields(
                self._content,
                lambda field: self._edit_child(name, field),
                path,
            )
        value = self._content().value(path)
        if isinstance(value, dict) and "__complex__" in value:
            return decode_complex(value)
        return value


class ComponentView:
    _model: Callable[[], ComponentSchema]
    _fields: FieldView

    def __init__(
        self,
        model: Callable[[], ComponentSchema],
        edit: Callable[[str], AbstractContextManager[ComponentSchema]],
        path: str,
        *,
        content: Callable[[], ParameterContent] | None = None,
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
        self._fields = (
            FieldView(model, edit, path)
            if content is None
            else ContentFields(content, edit, path)
        )

    @property
    def kind(self) -> str:
        """Return the component's declared kind."""
        value = self._fields["kind"]
        if not isinstance(value, str):
            raise ValueError("Component kind must be a string")
        return value

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
