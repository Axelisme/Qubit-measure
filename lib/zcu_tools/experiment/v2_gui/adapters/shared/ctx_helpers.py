"""Type-guarded helpers for reading ExpContext.md values."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zcu_tools.gui.adapter import (
    DirectValue,
    EvalValue,
    MetaDictWriteback,
    ScalarValue,
)

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import ExpContext


def md_get_float(ctx: ExpContext, key: str, default: float) -> float:
    value = ctx.md.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def md_get_int(ctx: ExpContext, key: str, default: int) -> int:
    value = ctx.md.get(key)
    if isinstance(value, int):
        return value
    return default


def md_has_key(ctx: ExpContext, key: str) -> bool:
    sentinel = object()
    try:
        value = ctx.md.get(key, sentinel)  # type: ignore[call-arg]
    except TypeError:
        value = ctx.md.get(key)
    return value is not sentinel and value is not None


def md_eval_float(ctx: ExpContext, key: str, default: float) -> EvalValue:
    resolved = md_get_float(ctx, key, default)
    return EvalValue(expr=key, resolved=resolved, error=None)


def md_eval_int(ctx: ExpContext, key: str, default: int) -> EvalValue:
    resolved = md_get_int(ctx, key, default)
    return EvalValue(expr=key, resolved=resolved, error=None)


def md_scalar_float(ctx: ExpContext, key: str, default: float) -> ScalarValue:
    if md_has_key(ctx, key):
        return md_eval_float(ctx, key, default)
    return DirectValue(default)


def md_scalar_int(ctx: ExpContext, key: str, default: int) -> ScalarValue:
    if md_has_key(ctx, key):
        return md_eval_int(ctx, key, default)
    return DirectValue(default)


def proper_relax(
    ctx: ExpContext, factor: float = 5.0, fallback: float = 100.0
) -> ScalarValue:
    """Default relax_delay value: ``factor * t1`` when md has t1, else fallback.

    Mirrors the notebook idiom ``relax_delay = 5 * md.t1``. When ``t1`` is present
    the result is an EvalValue (``f"{factor} * t1"``) so the GUI keeps the live
    expression; otherwise a plain DirectValue fallback.
    """
    if md_has_key(ctx, "t1"):
        t1 = md_get_float(ctx, "t1", fallback / factor)
        return EvalValue(expr=f"{factor} * t1", resolved=factor * t1, error=None)
    return DirectValue(fallback)


def md_writeback(
    ctx: ExpContext,
    key: str,
    description: str,
    value: float,
    ndigits: int = 4,
) -> MetaDictWriteback:
    """Build a MetaDictWriteback for a single md scalar.

    Collapses the recurring shape where ``key`` and ``md_key`` are the same and
    ``current_value`` is just ``ctx.md.get(key)``; the proposed value is rounded.
    """
    return MetaDictWriteback(
        key=key,
        description=description,
        current_value=ctx.md.get(key),
        md_key=key,
        proposed_value=round(value, ndigits),
    )
