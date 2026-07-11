"""Typed deferred defaults for context-free measure cfg definitions."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar, cast

from zcu_tools.gui.cfg import EvalValue, ScalarLeafInput, SweepValue
from zcu_tools.gui.session.value_lookup import (
    MissingValue,
    ScalarType,
    UnavailableValue,
    ValueRef,
    resolve_value_ref,
)

from .ctx_helpers import (
    md_has_key,
    proper_flux_range,
    proper_qub_freq_range,
    proper_res_freq_range,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import ExpContext

T = TypeVar("T", covariant=True)


@dataclass(frozen=True)
class Seed(Generic[T]):
    """A fresh-cfg default resolved only when a definition is instantiated."""

    _resolver: Callable[[ExpContext], T]
    description: str

    def resolve(self, ctx: ExpContext) -> T:
        # Values such as SweepValue and ReferenceValue are mutable. Every
        # instantiation receives an isolated tree even for literal/custom seeds.
        return deepcopy(self._resolver(ctx))


def literal(value: T) -> Seed[T]:
    snapshot = deepcopy(value)
    return Seed(lambda _ctx: snapshot, description=repr(snapshot))


def custom(resolve: Callable[[ExpContext], T], *, description: str) -> Seed[T]:
    """The single escape hatch for a named, low-frequency domain policy.

    The resolver must be a pure context lookup. Unlike :func:`literal`, this
    explicit escape hatch cannot snapshot arbitrary state captured by a caller's
    closure; callers own that purity contract. The resolved result is still
    deep-copied before it enters a cfg instance.
    """

    if not description.strip():
        raise ValueError("custom seed description must not be empty")
    return Seed(resolve, description=description)


def md(
    key: str,
    *,
    fallback: ScalarLeafInput,
    expr: str | None = None,
) -> Seed[ScalarLeafInput]:
    """Use a live md expression when ``key`` exists, else a fixed fallback."""

    if not key.strip():
        raise ValueError("md seed key must not be empty")

    def resolve(ctx: ExpContext) -> ScalarLeafInput:
        if md_has_key(ctx, key):
            return EvalValue(expr=expr if expr is not None else key)
        return fallback

    return Seed(resolve, description=f"md:{key}")


def scaled_md(
    key: str,
    *,
    factor: float,
    fallback_value: float,
) -> Seed[float | EvalValue]:
    """Use ``factor * key`` live; ``fallback_value`` is already the final value."""

    if not key.strip():
        raise ValueError("scaled_md seed key must not be empty")

    def resolve(ctx: ExpContext) -> float | EvalValue:
        if md_has_key(ctx, key):
            return EvalValue(expr=f"{factor} * {key}")
        return fallback_value

    return Seed(resolve, description=f"{factor} * md:{key}")


@dataclass(frozen=True)
class SweepDefault:
    """A sweep whose bounds may independently be deferred."""

    start: float | EvalValue | Seed[float | EvalValue]
    stop: float | EvalValue | Seed[float | EvalValue]
    expts: int

    def resolve(self, ctx: ExpContext) -> SweepValue:
        return SweepValue(
            start=_resolve_input(self.start, ctx),
            stop=_resolve_input(self.stop, ctx),
            expts=self.expts,
        )


def res_freq_range(
    *,
    expts: int,
    span_factor: float = 1.5,
) -> Seed[SweepValue]:
    """Deferred resonator frequency range using the established measure policy."""

    return custom(
        lambda ctx: proper_res_freq_range(ctx, expts, span_factor=span_factor),
        description=f"resonator frequency range ({expts} points)",
    )


def qub_freq_range(*, expts: int, span_factor: float = 1.5) -> Seed[SweepValue]:
    return custom(
        lambda ctx: proper_qub_freq_range(ctx, expts, span_factor=span_factor),
        description=f"qubit frequency range ({expts} points)",
    )


def flux_range(*, expts: int) -> Seed[SweepValue]:
    return custom(
        lambda ctx: proper_flux_range(ctx, expts),
        description=f"flux range ({expts} points)",
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
    """Resolve one registered session value once while creating a fresh cfg."""

    ref = ValueRef(key, type_name)

    def resolve(ctx: ExpContext) -> ScalarLeafInput:
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


def _resolve_input(value: T | Seed[T], ctx: ExpContext) -> T:
    if isinstance(value, Seed):
        return value.resolve(ctx)
    return deepcopy(value)


__all__ = [
    "NO_FALLBACK",
    "Seed",
    "SweepDefault",
    "custom",
    "flux_range",
    "literal",
    "md",
    "qub_freq_range",
    "res_freq_range",
    "scaled_md",
    "value_source",
]
