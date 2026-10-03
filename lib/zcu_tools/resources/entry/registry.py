"""Component model declarations, independent of experiment definitions."""

from collections.abc import Sequence

from .schema import ComponentSchema


class ComponentRegistry:
    def register(
        self, kind: str, model: type[ComponentSchema], *, references: Sequence[str] = ()
    ) -> None:
        raise NotImplementedError("component model registration")

    def unregister(self, kind: str) -> None:
        raise NotImplementedError("component model lifecycle")

    def get(self, kind: str) -> type[ComponentSchema]:
        raise NotImplementedError("component model lookup")


component_registry = ComponentRegistry()
