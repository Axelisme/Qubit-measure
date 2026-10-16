"""Fixed parameter scalar reads and shared calibration starting-value policies."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

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


def t1_initial_time(
    parameters: ParameterContext | None,
    *,
    multiplier: float,
    with_tone: bool = False,
) -> float | None:
    """Choose an editable T1-derived time in microseconds from a fixed capture.

    parameters is the Open/Reset capture, or None when no capture exists.
    multiplier must be finite and positive and scales the selected T1.
    Normally select $qubit.t1. With with_tone=True, first select
    $qubit.ext.t1_with_tone and consult ordinary T1 only if tone T1 is absent
    or null. If all selected leaves are absent/null (or there is no capture),
    use a 30 us fair time scale without writing calibration data.

    A selected bool, non-real, nonfinite or nonpositive stored value returns
    None for a required blank; it never falls through to another source.
    Invalid multiplier raises ValueError. Alias, path, namespace and source
    errors propagate from lookup unchanged. No refresh or file I/O occurs.
    """
    if not isfinite(multiplier) or multiplier <= 0:
        raise ValueError("T1 multiplier must be finite and positive")
    paths = ("$qubit.ext.t1_with_tone", "$qubit.t1") if with_tone else ("$qubit.t1",)
    if parameters is not None:
        for path in paths:
            result = parameters.lookup(path)
            if not result.present or result.value is None:
                continue
            value = result.value
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value <= 0
            ):
                return None
            return multiplier * value
    return multiplier * 30.0
