"""Explicit ordered composition root for user-placeable experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, PlacedNode

from .lenrabi import EXPERIMENT as LENRABI
from .mist import EXPERIMENT as MIST
from .qubit_freq import EXPERIMENT as QUBIT_FREQ
from .ro_optimize import EXPERIMENT as RO_OPTIMIZE
from .t1 import EXPERIMENT as T1
from .t2echo import EXPERIMENT as T2ECHO
from .t2ramsey import EXPERIMENT as T2RAMSEY


@dataclass(frozen=True, slots=True, init=False)
class ExperimentCatalog:
    """Immutable ordered measurement-experiment catalog.

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

    def create_placement(self, type_name: str, ctx: Any | None = None) -> PlacedNode:
        """Create a fresh placement; unknown names preserve ``KeyError``."""
        return PlacedNode(builder=self._by_name[type_name], default_context=ctx)


def _validate_builder(builder: object) -> None:
    if not isinstance(builder, Builder):
        raise TypeError(
            "experiment catalog declarations must be Builder instances, "
            f"got {type(builder).__name__}"
        )
    if not builder.name:
        raise ValueError("experiment catalog names must be non-empty")

    module_stem = builder.__class__.__module__.rsplit(".", 1)[-1]
    if module_stem != builder.name:
        raise ValueError(
            "experiment module stem must match Builder.name: "
            f"{module_stem!r} != {builder.name!r}"
        )

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


_DECLARATIONS: tuple[Builder, ...] = (
    QUBIT_FREQ,
    LENRABI,
    RO_OPTIMIZE,
    T1,
    T2RAMSEY,
    T2ECHO,
    MIST,
)

CATALOG = ExperimentCatalog(_DECLARATIONS)


def names() -> tuple[str, ...]:
    return CATALOG.names()


def builders() -> tuple[Builder, ...]:
    return CATALOG.builders()


def create_placement(type_name: str, ctx: Any | None = None) -> PlacedNode:
    return CATALOG.create_placement(type_name, ctx)
