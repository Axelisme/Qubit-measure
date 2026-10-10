"""Fixed parameter scalar reads and deferred frequency/flux range policies."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

from zcu_tools.gui.cfg import EvalValue, SweepValue

if TYPE_CHECKING:
    from zcu_tools.gui.session.parameters import ParameterContext


def optional_parameter_float(parameters: ParameterContext, path: str) -> float | None:
    """Read a finite real scalar from one fixed parameter capture.

    path is a direct component/general path or a captured role path accepted by
    ParameterContext.lookup. Return None only for an absent leaf or explicit
    null; otherwise return a float in the parameter's working units.
    Bool/non-real values raise TypeError; nonfinite numbers raise ValueError.
    Invalid paths, missing roles, unavailable sources and library errors
    propagate from the capture without fallback, refresh or file I/O.
    """
    result = parameters.lookup(path)
    if not result.present or result.value is None:
        return None
    value = result.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Parameter must be a real scalar: {path!r}")
    if not isfinite(value):
        raise ValueError(f"Parameter must be finite: {path!r}")
    return float(value)


def _freq_range(keys: tuple[str, str], expts: int, span_factor: float) -> SweepValue:
    center, width = keys
    return SweepValue(
        start=EvalValue(f"{center} - {span_factor} * {width}"),
        stop=EvalValue(f"{center} + {span_factor} * {width}"),
        expts=expts,
    )


def proper_res_freq_range(
    parameters: ParameterContext | None, expts: int, *, span_factor: float = 1.5
) -> SweepValue:
    """Store $resonator.freq ± span_factor*$resonator.kappa in working MHz.

    No calibration is read here, even with parameters=None. Missing aliases or
    calibration publish Invalid when the cfg resolves; no static fallback.
    """
    del parameters  # Keep the established caller signature without reading calibration.
    return _freq_range(("$resonator.freq", "$resonator.kappa"), expts, span_factor)


def proper_qub_freq_range(
    parameters: ParameterContext | None, expts: int, *, span_factor: float = 1.5
) -> SweepValue:
    """Store $qubit.freq ± span_factor*$qubit.kappa in working MHz.

    parameters may be None; no source is read. Missing calibration becomes
    Invalid at cfg resolution rather than a static default.
    """
    del parameters  # Keep the established caller signature without reading calibration.
    return _freq_range(("$qubit.freq", "$qubit.kappa"), expts, span_factor)


def proper_flux_range(parameters: ParameterContext | None, expts: int) -> SweepValue:
    """Store a scan 10% beyond $qubit.flux_int and $qubit.flux_half.

    parameters may be None; no calibration is read. Missing calibration becomes
    Invalid at cfg resolution rather than a static default.
    """
    del parameters  # Keep the established caller signature without reading calibration.
    return SweepValue(
        start=EvalValue("1.1 * $qubit.flux_int - 0.1 * $qubit.flux_half"),
        stop=EvalValue("1.1 * $qubit.flux_half - 0.1 * $qubit.flux_int"),
        expts=expts,
    )
