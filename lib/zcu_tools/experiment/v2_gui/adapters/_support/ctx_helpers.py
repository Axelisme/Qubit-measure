"""Type-guarded helpers for reading ExpContext.md values."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zcu_tools.gui.cfg import (
    EvalValue,
    SweepValue,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import ExpContext


def md_get_float(ctx: ExpContext, key: str, default: float) -> float:
    value = ctx.md.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def md_has_key(ctx: ExpContext, key: str) -> bool:
    sentinel = object()
    value = ctx.md.get(key, sentinel)
    return value is not sentinel and value is not None


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

    def _edge(sign: int) -> float | EvalValue:
        op = "-" if sign < 0 else "+"
        if have_md:
            return EvalValue(expr=f"{center_key} {op} {width_term}")
        return center + sign * half_span

    return SweepValue(start=_edge(-1), stop=_edge(+1), expts=expts)


def proper_res_freq_range(
    ctx: ExpContext, expts: int, *, span_factor: float = 1.5
) -> SweepValue:
    """Resonator frequency sweep range: ``r_f ± span_factor*rf_w``."""
    return _freq_range(ctx, "r_f", "rf_w", expts, span_factor, 6500.0, 500.0)


def proper_qub_freq_range(
    ctx: ExpContext, expts: int, *, span_factor: float = 1.5
) -> SweepValue:
    """Qubit frequency sweep range: ``q_f ± span_factor*qf_w``."""
    return _freq_range(ctx, "q_f", "qf_w", expts, span_factor, 5000.0, 1000.0)


def proper_flux_range(ctx: ExpContext, expts: int) -> SweepValue:
    """Flux sweep range spanning one period around the calibrated flux points.

    Extrapolates 10% past the two fitted positions (flx_half / flx_int):
    ``start = 1.1*flx_int - 0.1*flx_half``, ``stop = 1.1*flx_half - 0.1*flx_int``.
    When the md keys are absent, falls back to a fixed ``[-4e-3, 4e-3]`` scan.
    """
    if md_has_key(ctx, "flx_half") and md_has_key(ctx, "flx_int"):
        start: float | EvalValue = EvalValue(expr="1.1 * flx_int - 0.1 * flx_half")
        stop: float | EvalValue = EvalValue(expr="1.1 * flx_half - 0.1 * flx_int")
    else:
        start = -4e-3
        stop = 4e-3
    return SweepValue(start=start, stop=stop, expts=expts)
