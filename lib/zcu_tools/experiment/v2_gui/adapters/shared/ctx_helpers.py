"""Type-guarded helpers for reading ExpContext.md values."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from zcu_tools.gui.adapter import (
    DirectValue,
    EvalValue,
    MetaDictWriteback,
    ScalarValue,
    SweepValue,
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


def _freq_range(
    ctx: ExpContext,
    center_key: str,
    width_key: str,
    expts: int,
    span_factor: float,
    center_default: float,
    width_default: float,
) -> SweepValue:
    """A symmetric ``center ± span_factor*width`` SweepValue.

    Each edge is an EvalValue (``f"{center} - {span_factor} * {width}"``) when both
    md keys exist — so the GUI keeps the live expression — otherwise a plain scalar
    fallback. Width <= 0 falls back to a fixed ±30 MHz span.
    """
    center = md_get_float(ctx, center_key, center_default)
    width = md_get_float(ctx, width_key, width_default)
    half_span = span_factor * width if width > 0 else 30.0
    have_md = md_has_key(ctx, center_key) and md_has_key(ctx, width_key)

    # ``1 * width`` reads as just ``width``; keep the coefficient otherwise.
    width_term = width_key if span_factor == 1.0 else f"{span_factor} * {width_key}"

    def _edge(sign: int) -> Union[float, EvalValue]:
        op = "-" if sign < 0 else "+"
        resolved = center + sign * half_span
        if have_md:
            return EvalValue(
                expr=f"{center_key} {op} {width_term}",
                resolved=resolved,
                error=None,
            )
        return resolved

    return SweepValue(start=_edge(-1), stop=_edge(+1), expts=expts)


def proper_res_freq_range(
    ctx: ExpContext, expts: int, *, span_factor: float = 1.5
) -> SweepValue:
    """Resonator frequency sweep range: ``r_f ± span_factor*rf_w``."""
    return _freq_range(ctx, "r_f", "rf_w", expts, span_factor, 6000.0, 20.0)


def proper_qub_freq_range(
    ctx: ExpContext, expts: int, *, span_factor: float = 1.5
) -> SweepValue:
    """Qubit frequency sweep range: ``q_f ± span_factor*qf_w``."""
    return _freq_range(ctx, "q_f", "qf_w", expts, span_factor, 4000.0, 20.0)


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
