"""Fixed parameter scalar reads."""

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
