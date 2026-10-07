"""Fixed parameter range policies and remaining legacy seed reads."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

from zcu_tools.gui.cfg import EvalValue, SweepValue

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.adapter import SessionEnv
    from zcu_tools.gui.session.parameters import ParameterContext


def md_get_float(ctx: SessionEnv, key: str, default: float) -> float:
    value = ctx.md.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def md_has_key(ctx: SessionEnv, key: str) -> bool:
    sentinel = object()
    value = ctx.md.get(key, sentinel)
    return value is not sentinel and value is not None


def _optional_scalar(parameters: ParameterContext, path: str) -> float | None:
    result = parameters.lookup(path)
    if not result.present or result.value is None:
        return None
    value = result.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Parameter must be a real scalar: {path!r}")
    if not isfinite(value):
        raise ValueError(f"Parameter must be finite: {path!r}")
    return float(value)


def _freq_range(
    parameters: ParameterContext,
    keys: tuple[str, str],
    defaults: tuple[float, float],
    expts: int,
    span_factor: float,
) -> SweepValue:
    """Read the fixed scalar paths and build the established live range."""
    center_key, width_key = keys
    center_value = _optional_scalar(parameters, center_key)
    width_value = _optional_scalar(parameters, width_key)
    center = defaults[0] if center_value is None else center_value
    width = defaults[1] if width_value is None else width_value
    half_span = span_factor * width if width > 0 else 30.0
    center_term = repr(center) if center_value is None else center_key
    width_term = width_key if span_factor == 1.0 else f"{span_factor} * {width_key}"
    if width_value is None:
        width_term = repr(half_span)

    if center_value is not None or width_value is not None:
        start: float | EvalValue = EvalValue(expr=f"{center_term} - {width_term}")
        stop: float | EvalValue = EvalValue(expr=f"{center_term} + {width_term}")
    else:
        start = center - half_span
        stop = center + half_span
    return SweepValue(start=start, stop=stop, expts=expts)


def proper_res_freq_range(
    parameters: ParameterContext, expts: int, *, span_factor: float = 1.5
) -> SweepValue:
    """Build resonator.freq ± span_factor*resonator.kappa in working MHz.

    Existing numeric fields remain live EvalValue terms. Absent/null fields
    use the original 6500 MHz center and 500 MHz width fallbacks.
    Present non-real/nonfinite scalars fail.
    Role, source and library errors propagate without fallback.
    """
    return _freq_range(
        parameters,
        ("resonator.freq", "resonator.kappa"),
        (6500.0, 500.0),
        expts,
        span_factor,
    )


def proper_qub_freq_range(
    parameters: ParameterContext, expts: int, *, span_factor: float = 1.5
) -> SweepValue:
    """Build qubit.freq ± span_factor*qubit.kappa in working MHz.

    Existing numeric fields remain live EvalValue terms. Absent/null fields
    use the original 5000 MHz center and 1000 MHz width fallbacks.
    Present non-real/nonfinite scalars fail.
    Role, source and library errors propagate without fallback.
    """
    return _freq_range(
        parameters,
        ("qubit.freq", "qubit.kappa"),
        (5000.0, 1000.0),
        expts,
        span_factor,
    )


def proper_flux_range(parameters: ParameterContext, expts: int) -> SweepValue:
    """Build a scan 10% beyond qubit.flux_int and qubit.flux_half.

    Both present real finite calibrations produce live EvalValue edges.
    Missing/null calibration falls back to [-4e-3, 4e-3] in the working flux
    unit. Present invalid calibrations and role/source errors propagate.
    """
    flux_half = _optional_scalar(parameters, "qubit.flux_half")
    flux_int = _optional_scalar(parameters, "qubit.flux_int")
    if flux_half is not None and flux_int is not None:
        start: float | EvalValue = EvalValue(
            expr="1.1 * qubit.flux_int - 0.1 * qubit.flux_half"
        )
        stop: float | EvalValue = EvalValue(
            expr="1.1 * qubit.flux_half - 0.1 * qubit.flux_int"
        )
    else:
        start = -4e-3
        stop = 4e-3
    return SweepValue(start=start, stop=stop, expts=expts)
