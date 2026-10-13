"""Entry-specific field containers using the shared parameter field view."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager

from zcu_tools.format_version import YamlValue
from zcu_tools.resources.parameter_fields import (
    FieldNode,
    FieldView,
    field_container,
    model_field_value,
    read_field,
)

from .schema import ComponentSchema, PointGeneral, SetupGeneral

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
    ) -> None:
        super().__init__(model, edit, "general")
        self._general_model = model

        @contextmanager
        def edit_extension(name: str) -> Generator[FieldNode]:
            with edit(f"ext.{name}") as draft:
                yield field_container(draft.ext, "general.ext")
                draft.model_fields_set.add("ext")

        self._extension = FieldView(
            lambda: field_container(model().ext, "general.ext"),
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
