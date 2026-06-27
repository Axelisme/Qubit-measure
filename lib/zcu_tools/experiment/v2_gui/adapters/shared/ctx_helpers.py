"""Type-guarded helpers for reading ExpContext.md values."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zcu_tools.gui.app.main.adapter import (
    DirectValue,
    EvalValue,
    ScalarValue,
    SweepValue,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import ExpContext


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
    value = ctx.md.get(key, sentinel)
    return value is not sentinel and value is not None


def md_eval_float(key: str) -> EvalValue:
    return EvalValue(expr=key)


def md_eval_int(key: str) -> EvalValue:
    return EvalValue(expr=key)


def md_scalar_float(ctx: ExpContext, key: str, default: float) -> ScalarValue:
    if md_has_key(ctx, key):
        return md_eval_float(key)
    return DirectValue(default)


def md_scalar_int(ctx: ExpContext, key: str, default: int) -> ScalarValue:
    if md_has_key(ctx, key):
        return md_eval_int(key)
    return DirectValue(default)


def md_eval_scaled(
    ctx: ExpContext, key: str, factor: float, fallback: float
) -> float | EvalValue:
    """A ``factor * <key>`` sweep edge that stays md-linked.

    When md has ``key`` the edge is an ``EvalValue(f"{factor} * {key}")`` so the
    GUI keeps the live expression (re-derives if md changes); otherwise a plain
    ``factor * fallback`` float. Use for ``SweepValue.start``/``stop`` edges
    derived from an md quantity (e.g. ``stop = 5 * md.t1``). Returns the edge
    value, not a whole SweepValue.
    """
    if md_has_key(ctx, key):
        return EvalValue(expr=f"{factor} * {key}")
    return factor * fallback


def proper_relax(
    ctx: ExpContext, factor: float = 5.0, fallback: float = 100.0
) -> ScalarValue:
    """Default relax_delay value: ``factor * t1`` when md has t1, else fallback.

    Mirrors the notebook idiom ``relax_delay = 5 * md.t1``. When ``t1`` is present
    the result is an EvalValue (``f"{factor} * t1"``) so the GUI keeps the live
    expression; otherwise a plain DirectValue fallback.
    """
    if md_has_key(ctx, "t1"):
        return EvalValue(expr=f"{factor} * t1")
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


def proper_reset_freq_range(
    ctx: ExpContext, expts: int, *, half_span_default: float = 50.0
) -> SweepValue:
    """Sideband-reset frequency sweep range: ``reset_f ± span``.

    ``reset_f`` (= r_f - q_f) is the sweep centre. When md also has ``resetf_w``
    the half-span is ``1.5*resetf_w`` and each edge stays an EvalValue (the GUI
    re-derives if md changes); otherwise a fixed ``reset_f ± half_span_default``
    (notebook single-tone reset: ``reset_f - 50`` .. ``reset_f + 50``). When
    ``reset_f`` is absent the scan centres on 0.0.
    """
    center = md_get_float(ctx, "reset_f", 3000.0)
    have_center = md_has_key(ctx, "reset_f")
    if have_center and md_has_key(ctx, "resetf_w"):
        start: float | EvalValue = EvalValue(expr="reset_f - 1.5 * resetf_w")
        stop: float | EvalValue = EvalValue(expr="reset_f + 1.5 * resetf_w")
    elif have_center:
        start = EvalValue(expr=f"reset_f - {half_span_default}")
        stop = EvalValue(expr=f"reset_f + {half_span_default}")
    else:
        start = center - half_span_default
        stop = center + half_span_default
    return SweepValue(start=start, stop=stop, expts=expts)


def proper_reset_freq_axis(
    ctx: ExpContext,
    center_key: str,
    expts: int,
    *,
    half_span_default: float = 50.0,
) -> SweepValue:
    """One axis of a dual-tone reset frequency map: ``<center_key> ± span``.

    Mirrors ``proper_reset_freq_range`` but parameterised on the centre md key so
    each of the two sweep axes (``reset_f1`` / ``reset_f2``) seeds from its own
    centre. When the matching width key (``<center_key>_w``, e.g. ``reset_f1_w``)
    is present the half-span is ``1.5*width`` and each edge stays an EvalValue (the
    GUI re-derives if md changes); otherwise a fixed ``center ± half_span_default``.
    When the centre key is absent the scan centres on 0.0.
    """
    width_key = f"{center_key}_w"
    center = md_get_float(ctx, center_key, 3000.0)
    have_center = md_has_key(ctx, center_key)
    if have_center and md_has_key(ctx, width_key):
        start: float | EvalValue = EvalValue(expr=f"{center_key} - 1.5 * {width_key}")
        stop: float | EvalValue = EvalValue(expr=f"{center_key} + 1.5 * {width_key}")
    elif have_center:
        start = EvalValue(expr=f"{center_key} - {half_span_default}")
        stop = EvalValue(expr=f"{center_key} + {half_span_default}")
    else:
        start = center - half_span_default
        stop = center + half_span_default
    return SweepValue(start=start, stop=stop, expts=expts)


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
