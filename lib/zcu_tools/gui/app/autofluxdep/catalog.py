"""Ordered measurement catalog injected by the application composition root."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from zcu_tools.gui.session.types import SessionEnv

from .nodes.builder import Builder, PlacedNode


@dataclass(frozen=True, slots=True, init=False)
class ExperimentCatalog:
    """Immutable ordered measurement-experiment catalog.

    declarations is an ordered iterable of stateless Builder instances.
    Raise TypeError for non-Builders and ValueError for empty/duplicate names,
    or duplicate dependency/output declarations. Names are declared identities,
    independent of the Builder's source module or filename.
    Construction preserves the caller's declaration order and Builder identities.

    Catalog order controls only the GUI add menu. Runtime execution continues to
    use the persisted workflow's user-defined placement order.
    """

    _builders: tuple[Builder, ...]
    _by_name: Mapping[str, Builder]

    def __init__(self, declarations: Iterable[Builder]) -> None:
        ordered = tuple(declarations)
        by_name: dict[str, Builder] = {}
        for builder in ordered:
            _validate_builder(builder)
            if builder.name in by_name:
                raise ValueError(f"duplicate experiment name: {builder.name!r}")
            by_name[builder.name] = builder
        object.__setattr__(self, "_builders", ordered)
        object.__setattr__(self, "_by_name", MappingProxyType(by_name))

    def names(self) -> tuple[str, ...]:
        """Return experiment names in GUI menu order."""
        return tuple(self._by_name)

    def builders(self) -> tuple[Builder, ...]:
        """Return authoritative Builder singletons in GUI menu order."""
        return self._builders

    def create_placement(
        self, type_name: str, ctx: SessionEnv | None = None
    ) -> PlacedNode:
        """Create independent placement defaults for the named measurement Builder.

        type_name is a name returned by names(). ctx seeds fresh defaults only;
        None uses context-free defaults. Unknown names raise KeyError.
        The returned placement owns its schema; no run state is stored here.
        """
        return PlacedNode(builder=self._by_name[type_name], default_context=ctx)


def _validate_builder(builder: object) -> None:
    if not isinstance(builder, Builder):
        raise TypeError(
            "experiment catalog declarations must be Builder instances, "
            f"got {type(builder).__name__}"
        )
    if not builder.name:
        raise ValueError("experiment catalog names must be non-empty")

    _require_unique(builder, "provides", builder.provides)
    _require_unique(builder, "requires", (item.key for item in builder.requires))
    _require_unique(builder, "optional", (item.key for item in builder.optional))
    _require_unique(
        builder,
        "requires_modules",
        (item.name for item in builder.requires_modules),
    )
    _require_unique(
        builder,
        "optional_modules",
        (item.name for item in builder.optional_modules),
    )
    _require_unique(builder, "provides_modules", builder.provides_modules)
    _require_unique(
        builder,
        "feedback_slots",
        (item.key for item in builder.feedback_slots),
    )


def _require_unique(builder: Builder, declaration: str, values: Iterable[str]) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(
                f"experiment {builder.name!r} has duplicate {declaration} "
                f"declaration: {value!r}"
            )
        seen.add(value)
