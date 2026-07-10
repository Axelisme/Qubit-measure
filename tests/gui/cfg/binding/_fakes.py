from __future__ import annotations

from collections.abc import Sequence

from zcu_tools.gui.cfg.binding import ResolvedReference


class BindingPorts:
    def __init__(self) -> None:
        self.expressions: dict[str, int | float] = {}
        self.options: dict[str, tuple[object, ...]] = {}
        self.references: dict[tuple[str, str], ResolvedReference] = {}

    def evaluate(self, expression: str) -> int | float:
        try:
            return self.expressions[expression]
        except KeyError as exc:
            raise RuntimeError(f"unknown expression {expression!r}") from exc

    def provide(self, source_id: str) -> Sequence[object]:
        try:
            return self.options[source_id]
        except KeyError as exc:
            raise RuntimeError(f"unknown option source {source_id!r}") from exc

    def keys(self, kind: str, allowed_labels: frozenset[str]) -> Sequence[str]:
        return tuple(
            sorted(
                key
                for (entry_kind, key), resolved in self.references.items()
                if entry_kind == kind and resolved.label in allowed_labels
            )
        )

    def resolve(self, kind: str, key: str) -> ResolvedReference | None:
        return self.references.get((kind, key))
