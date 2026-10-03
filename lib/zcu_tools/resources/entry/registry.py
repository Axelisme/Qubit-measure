"""Component model declarations, independent of experiment definitions."""

from collections.abc import Sequence

from .schema import ComponentSchema


class ComponentRegistry:
    def __init__(self) -> None:
        self._models: dict[str, type[ComponentSchema]] = {}
        self._references: dict[str, tuple[str, ...]] = {}

    def register(
        self, kind: str, model: type[ComponentSchema], *, references: Sequence[str] = ()
    ) -> None:
        if kind in self._models:
            raise ValueError(f"Kind {kind!r} is already registered")
        self._models[kind] = model
        self._references[kind] = tuple(references)

    def unregister(self, kind: str) -> None:
        del self._models[kind]
        del self._references[kind]

    def get(self, kind: str) -> type[ComponentSchema]:
        return self._models[kind]


component_registry = ComponentRegistry()
component_registry.register("resonator", ComponentSchema)
