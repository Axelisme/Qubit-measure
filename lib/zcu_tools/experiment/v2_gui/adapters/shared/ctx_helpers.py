"""Type-guarded helpers for reading ExpContext.md values."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zcu_tools.gui.adapter import DirectValue, EvalValue, ScalarValue

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
