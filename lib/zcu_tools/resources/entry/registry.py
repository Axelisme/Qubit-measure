"""Component model declarations, independent of experiment definitions."""

from collections.abc import Mapping, Sequence
from difflib import get_close_matches
from pathlib import Path

from zcu_tools.resources.document_store import FieldPath, UnitSpec

from .errors import UnknownFieldError, UnknownKindError
from .schema import ComponentSchema, ResonatorSchema


class ComponentRegistry:
    def __init__(self) -> None:
        self._models: dict[str, type[ComponentSchema]] = {}
        self._references: dict[str, tuple[str, ...]] = {}
        self._units: dict[str, dict[FieldPath, UnitSpec]] = {}

    def register(
        self, kind: str, model: type[ComponentSchema], *, references: Sequence[str] = ()
    ) -> None:
        if kind in self._models:
            raise ValueError(f"Kind {kind!r} is already registered")
        self._models[kind] = model
        self._references[kind] = tuple(references)
        self._units[kind] = {
            (name,): metadata
            for name, field in model.model_fields.items()
            for metadata in field.metadata
            if isinstance(metadata, UnitSpec)
        }

    def unregister(self, kind: str) -> None:
        del self._models[kind]
        del self._references[kind]
        del self._units[kind]

    def get(
        self, kind: str, *, source: Path | None = None, component: str | None = None
    ) -> type[ComponentSchema]:
        try:
            return self._models[kind]
        except KeyError as cause:
            raise UnknownKindError(
                source, component, kind, tuple(get_close_matches(kind, self._models))
            ) from cause

    def check_fields(
        self, kind: str, fields: Mapping[str, object], *, path: str
    ) -> None:
        known_fields = self.get(kind).model_fields
        for name in fields:
            if name not in known_fields:
                raise UnknownFieldError(
                    f"{path}.{name}", name, tuple(get_close_matches(name, known_fields))
                )

    def units(
        self, kind: str, *, source: Path | None = None, component: str | None = None
    ) -> Mapping[FieldPath, UnitSpec]:
        self.get(kind, source=source, component=component)
        return dict(self._units[kind])


component_registry = ComponentRegistry()
component_registry.register("resonator", ResonatorSchema)
