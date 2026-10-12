"""Typed deferred defaults for context-free measure cfg definitions."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar, cast

from zcu_tools.gui.app.measure.reference_defaults import resolve_library_reference
from zcu_tools.gui.cfg import (
    DirectValue,
    EvalValue,
    ReferenceValue,
    ScalarLeafInput,
    SweepValue,
)
from zcu_tools.gui.session.aliases import AliasRow, AliasTable
from zcu_tools.gui.session.value_lookup import (
    MissingValue,
    ScalarType,
    UnavailableValue,
    ValueRef,
    resolve_value_ref,
)
from zcu_tools.resources.entry import RoleResolutionError

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.adapter import SessionEnv

T = TypeVar("T", covariant=True)


@dataclass(frozen=True)
class Seed(Generic[T]):
    """A default resolved only when a definition is instantiated.

    _resolver reads a fixed SessionEnv and returns T; description names its
    source in error notes. Resolution deep-copies mutable results so instances
    do not share cfg values. Resolver exceptions propagate."""

    _resolver: Callable[[SessionEnv], T]
    description: str

    def resolve(self, ctx: SessionEnv) -> T:
        # Values such as SweepValue and ReferenceValue are mutable. Every
        # instantiation receives an isolated tree even for literal/custom seeds.
        """Resolve against the supplied SessionEnv and return a fresh carrier.

        Seed propagates resolver failures; SweepDefault resolves start/stop seeds
        and preserves the declared point count, propagating bound failures."""
        return deepcopy(self._resolver(ctx))


def literal(value: T) -> Seed[T]:
    """Snapshot value now and return a seed yielding isolated copies at resolution."""
    snapshot = deepcopy(value)
    return Seed(lambda _ctx: snapshot, description=repr(snapshot))


def custom(resolve: Callable[[SessionEnv], T], *, description: str) -> Seed[T]:
    """The single escape hatch for a named, low-frequency domain policy.

    The resolver must be a pure context lookup. Unlike :func:`literal`, this
    explicit escape hatch cannot snapshot arbitrary state captured by a caller's
    closure; callers own that purity contract. The resolved result is still
    deep-copied before it enters a cfg instance.
    """

    if not description.strip():
        raise ValueError("custom seed description must not be empty")
    return Seed(resolve, description=description)


def alias_name_seed(alias: str) -> Seed[DirectValue]:
    """Freeze a bare alias name as direct input when a fresh cfg is created.

    alias is an alias-table identifier without ``$`` or a field suffix. Invalid
    identifiers raise ValueError at declaration. The fixed SessionEnv parameter
    capture resolves the name once; later alias changes do not update this input.
    Missing captures or alias resolution errors produce a direct Invalid input
    naming ``$alias``, with no fallback. Other source failures propagate.
    """
    AliasTable((AliasRow(alias, ""),))
    path = f"${alias}"

    def resolve(ctx: SessionEnv) -> DirectValue:
        """Resolve against the supplied SessionEnv and return a fresh carrier.

        Seed propagates resolver failures; SweepDefault resolves start/stop seeds
        and preserves the declared point count, propagating bound failures."""
        if ctx.parameters is None:
            return DirectValue(
                None,
                raw=path,
                error=f"Cannot resolve {path}: parameter context is unavailable",
            )
        try:
            name = ctx.parameters.resolve_path(path)
        except RoleResolutionError as exc:
            return DirectValue(None, raw=path, error=f"Cannot resolve {path}: {exc}")
        return DirectValue(name)

    return Seed(resolve, description=f"alias name:{path}")


def path_seed(
    path: str, *, fallback: ScalarLeafInput, expr: str | None = None
) -> Seed[ScalarLeafInput]:
    """Store a deferred calibration expression without reading its source.

    path is a canonical $alias.path or concrete/general path; expr optionally
    supplies arithmetic. Empty paths raise ValueError. Missing aliases and
    calibrations become Invalid at resolution. fallback remains accepted but
    is not substituted for scalar defaults.
    """
    del fallback  # Scalar defaults never substitute missing calibration.
    if not path.strip():
        raise ValueError("parameter seed path must not be empty")

    def resolve(_ctx: SessionEnv) -> ScalarLeafInput:
        """Resolve against the supplied SessionEnv and return a fresh carrier.

        Seed propagates resolver failures; SweepDefault resolves start/stop seeds
        and preserves the declared point count, propagating bound failures."""
        return EvalValue(expr=expr if expr is not None else path)

    return Seed(resolve, description=f"parameter:{path}")


def module_seed(
    path: str, *, fallback: Seed[ReferenceValue] | None
) -> Seed[ReferenceValue | None]:
    """Mount a captured module slot/root, or an explicit fresh fallback.

    path is a role/component module slot or a concrete library root. Missing,
    null and dangling slots use fallback (None disables an optional module).
    Missing direct roots also use fallback. Invalid paths, unavailable captures,
    malformed modules and asset/materialization failures propagate. A mounted
    reference retains path so later lowering resolves it on the same capture.
    A declared unbound role instead returns a linked empty section without lookup
    or fallback, even when fallback is None. Cfg publishes missing-reference Invalid;
    refresh can supply the bound catalog shape, while reset reruns fallback policy.
    """
    if not path.strip():
        raise ValueError("module seed path must not be empty")

    def resolve(ctx: SessionEnv) -> ReferenceValue | None:
        """Resolve against the supplied SessionEnv and return a fresh carrier.

        Seed propagates resolver failures; SweepDefault resolves start/stop seeds
        and preserves the declared point count, propagating bound failures."""
        return resolve_library_reference(
            path,
            ctx,
            kind="module",
            fallback=None if fallback is None else fallback.resolve,
        )

    return Seed(resolve, description=f"module:{path}")


def scaled_path_seed(
    path: str, *, factor: float, fallback_value: float
) -> Seed[float | EvalValue]:
    """Store factor * canonical path without reading calibration at creation.

    Missing aliases or calibration become Invalid at resolution. Empty paths
    raise ValueError. fallback_value remains accepted but is not substituted.
    """
    del fallback_value  # Scalar defaults never substitute missing calibration.
    if not path.strip():
        raise ValueError("scaled parameter seed path must not be empty")

    def resolve(_ctx: SessionEnv) -> float | EvalValue:
        """Resolve against the supplied SessionEnv and return a fresh carrier.

        Seed propagates resolver failures; SweepDefault resolves start/stop seeds
        and preserves the declared point count, propagating bound failures."""
        return EvalValue(expr=f"{factor} * {path}")

    return Seed(resolve, description=f"{factor} * parameter:{path}")


@dataclass(frozen=True)
class SweepDefault:
    """Sweep bounds with independently deferred inputs.

    start and stop are numeric bounds or expression/seed carriers in the sweep
    axis units. expts is the point count passed unchanged to SweepValue.
    resolve evaluates seeds against one fixed SessionEnv without reading devices."""

    start: float | EvalValue | Seed[float | EvalValue]
    stop: float | EvalValue | Seed[float | EvalValue]
    expts: int

    def resolve(self, ctx: SessionEnv) -> SweepValue:
        """Resolve against the supplied SessionEnv and return a fresh carrier.

        Seed propagates resolver failures; SweepDefault resolves start/stop seeds
        and preserves the declared point count, propagating bound failures."""
        return SweepValue(
            start=_resolve_input(self.start, ctx),
            stop=_resolve_input(self.stop, ctx),
            expts=self.expts,
        )


class _NoFallback:
    pass


NO_FALLBACK = _NoFallback()


def value_source(
    key: str,
    *,
    target_type: ScalarType,
    type_name: str | None = None,
    fallback: ScalarLeafInput | _NoFallback = NO_FALLBACK,
) -> Seed[ScalarLeafInput]:
    """Resolve one registered session value once while creating a fresh cfg.

    key identifies the value registry entry, target_type is the required scalar
    Python type, and type_name optionally constrains the registered type label.
    Missing/unavailable sources use the explicit fallback; NO_FALLBACK re-raises.
    Other lookup/type failures propagate. Returns a deferred scalar Seed."""

    ref = ValueRef(key, type_name)

    def resolve(ctx: SessionEnv) -> ScalarLeafInput:
        """Resolve against the supplied SessionEnv and return a fresh carrier.

        Seed propagates resolver failures; SweepDefault resolves start/stop seeds
        and preserves the declared point count, propagating bound failures."""
        try:
            return cast(
                ScalarLeafInput,
                resolve_value_ref(ref, ctx.values, target_type=target_type),
            )
        except (MissingValue, UnavailableValue):
            if isinstance(fallback, _NoFallback):
                raise
            return fallback

    return Seed(resolve, description=f"value source:{key}")


def _resolve_input(value: T | Seed[T], ctx: SessionEnv) -> T:
    if isinstance(value, Seed):
        return value.resolve(ctx)
    return deepcopy(value)


__all__ = [
    "NO_FALLBACK",
    "Seed",
    "SweepDefault",
    "alias_name_seed",
    "custom",
    "literal",
    "module_seed",
    "path_seed",
    "scaled_path_seed",
    "value_source",
]
