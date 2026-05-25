"""Type-guarded helpers for reading ExpContext.md values."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zcu_tools.gui.adapter import EvalValue

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


def md_eval_float(ctx: ExpContext, key: str, default: float) -> EvalValue:
    resolved = md_get_float(ctx, key, default)
    return EvalValue(expr=key, resolved=resolved, error=None)
