"""Fixed parameter range policies and remaining legacy seed reads."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zcu_tools.gui.cfg import SweepValue

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


def _freq_range(
    parameters: ParameterContext,
    center_key: str,
    width_key: str,
    expts: int,
    span_factor: float,
    center_default: float,
    width_default: float,
) -> SweepValue:
    """Read the fixed scalar paths and build the established live range."""
    raise NotImplementedError("Parameter frequency range is not implemented")


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
        "resonator.freq",
        "resonator.kappa",
        expts,
        span_factor,
        6500.0,
        500.0,
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
        parameters, "qubit.freq", "qubit.kappa", expts, span_factor, 5000.0, 1000.0
    )


def proper_flux_range(parameters: ParameterContext, expts: int) -> SweepValue:
    """Build a scan 10% beyond qubit.flux_int and qubit.flux_half.

    Both present real finite calibrations produce live EvalValue edges.
    Missing/null calibration falls back to [-4e-3, 4e-3] in the working flux
    unit. Present invalid calibrations and role/source errors propagate.
    """
    raise NotImplementedError("Parameter flux range is not implemented")
