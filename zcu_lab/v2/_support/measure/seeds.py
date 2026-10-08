"""Typed deferred defaults for context-free measure cfg definitions."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar, cast

from zcu_tools.gui.app.measure.cfg_schemas import module_cfg_to_value
from zcu_tools.gui.cfg import EvalValue, ReferenceValue, ScalarLeafInput, SweepValue
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.value_lookup import (
    MissingValue,
    ScalarType,
    UnavailableValue,
    ValueRef,
    resolve_value_ref,
)

from zcu_lab.v2._support.measure.ctx_helpers import (
    md_has_key,
    proper_flux_range,
    proper_qub_freq_range,
    proper_res_freq_range,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.adapter import SessionEnv
    from zcu_tools.gui.session.parameters import ParameterContext

T = TypeVar("T", covariant=True)


@dataclass(frozen=True)
class Seed(Generic[T]):
    """A fresh-cfg default resolved only when a definition is instantiated."""

    _resolver: Callable[[SessionEnv], T]
    description: str

    def resolve(self, ctx: SessionEnv) -> T:
        # Values such as SweepValue and ReferenceValue are mutable. Every
        # instantiation receives an isolated tree even for literal/custom seeds.
        return deepcopy(self._resolver(ctx))


def literal(value: T) -> Seed[T]:
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


def md(
    key: str,
    *,
    fallback: ScalarLeafInput,
    expr: str | None = None,
) -> Seed[ScalarLeafInput]:
    """Use a live md expression when ``key`` exists, else a fixed fallback."""

    if not key.strip():
        raise ValueError("md seed key must not be empty")

    def resolve(ctx: SessionEnv) -> ScalarLeafInput:
        if md_has_key(ctx, key):
            return EvalValue(expr=expr if expr is not None else key)
        return fallback

    return Seed(resolve, description=f"md:{key}")


def path_seed(
    path: str, *, fallback: ScalarLeafInput, expr: str | None = None
) -> Seed[ScalarLeafInput]:
    """Use a role/component parameter path when present, else explicit fallback.

    Resolve only at fresh defaults. Empty paths raise ValueError; source faults
    propagate rather than becoming fallback. expr overrides the live expression.
    """
    if not path.strip():
        raise ValueError("parameter seed path must not be empty")

    def resolve(ctx: SessionEnv) -> ScalarLeafInput:
        if ctx.parameters is not None and ctx.parameters.lookup(path).present:
            return EvalValue(expr=expr if expr is not None else path)
        return fallback

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
    """
    if not path.strip():
        raise ValueError("module seed path must not be empty")

    def resolve(ctx: SessionEnv) -> ReferenceValue | None:
        parameters = ctx.parameters
        if parameters is not None:
            try:
                key = parameters.library_key(path)
            except AttributeError:
                # The path owner defines unset and dangling module slots.
                key = None
            if key is not None:
                try:
                    parameters.library.module_spec(key)
                except KeyError:
                    # A valid direct root can name a not-yet-created entry.
                    pass
                else:
                    _, value = module_cfg_to_value(parameters.library.get_module(key))
                    return ReferenceValue(chosen_key=path, value=value)
        return fallback.resolve(ctx) if fallback is not None else None

    return Seed(resolve, description=f"module:{path}")


def scaled_path_seed(
    path: str, *, factor: float, fallback_value: float
) -> Seed[float | EvalValue]:
    """Seed factor * path, or the already scaled fallback when absent.

    Empty paths raise ValueError. Capture faults propagate; no focus rebind.
    """
    if not path.strip():
        raise ValueError("scaled parameter seed path must not be empty")

    def resolve(ctx: SessionEnv) -> float | EvalValue:
        if ctx.parameters is not None and ctx.parameters.lookup(path).present:
            return EvalValue(expr=f"{factor} * {path}")
        return fallback_value

    return Seed(resolve, description=f"{factor} * parameter:{path}")


def scaled_md(
    key: str,
    *,
    factor: float,
    fallback_value: float,
) -> Seed[float | EvalValue]:
    """Use ``factor * key`` live; ``fallback_value`` is already the final value."""

    if not key.strip():
        raise ValueError("scaled_md seed key must not be empty")

    def resolve(ctx: SessionEnv) -> float | EvalValue:
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

    def resolve(self, ctx: SessionEnv) -> SweepValue:
        return SweepValue(
            start=_resolve_input(self.start, ctx),
            stop=_resolve_input(self.stop, ctx),
            expts=self.expts,
        )


def _parameters(ctx: SessionEnv) -> ParameterContext:
    if ctx.parameters is None:
        raise FailedPreconditionError(
            "No parameter context is available", reason_code="no_context"
        )
    return ctx.parameters


def res_freq_range(
    *,
    expts: int,
    span_factor: float = 1.5,
) -> Seed[SweepValue]:
    """Deferred resonator frequency range using the established measure policy."""

    return custom(
        lambda ctx: proper_res_freq_range(
            _parameters(ctx), expts, span_factor=span_factor
        ),
        description=f"resonator frequency range ({expts} points)",
    )


def qub_freq_range(*, expts: int, span_factor: float = 1.5) -> Seed[SweepValue]:
    return custom(
        lambda ctx: proper_qub_freq_range(
            _parameters(ctx), expts, span_factor=span_factor
        ),
        description=f"qubit frequency range ({expts} points)",
    )


def flux_range(*, expts: int) -> Seed[SweepValue]:
    return custom(
        lambda ctx: proper_flux_range(_parameters(ctx), expts),
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

    def resolve(ctx: SessionEnv) -> ScalarLeafInput:
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
    "custom",
    "flux_range",
    "literal",
    "md",
    "module_seed",
    "path_seed",
    "qub_freq_range",
    "scaled_path_seed",
    "res_freq_range",
    "scaled_md",
    "value_source",
]
