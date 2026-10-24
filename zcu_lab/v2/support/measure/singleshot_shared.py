"""Single-shot defaults and analysis calibration from fixed parameter captures.

Sweep runs classify shots using resolved cfg values. Analysis-only consumers
read their Run's ParameterContext; missing sources never use current globals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from zcu_tools.format_version import YamlValue

from zcu_lab.v2.support.measure.ctx_helpers import optional_parameter_float

if TYPE_CHECKING:
    from zcu_tools.gui.session.parameters import ParameterContext


def _analysis_value(parameters: ParameterContext | None, path: str) -> YamlValue:
    if parameters is None:
        raise RuntimeError(f"Analysis requires a captured parameter source: {path}")
    result = parameters.lookup(path)
    return result.value if result.present else None


def read_analysis_float(parameters: ParameterContext | None, path: str) -> float | None:
    """Read an optional finite scalar at a canonical path from the Run capture.

    Missing/null leaves return None for the core's optional analysis input.
    Missing capture raises RuntimeError; invalid sources, aliases and values
    propagate without falling back to current globals or static calibration.
    """
    if parameters is None:
        raise RuntimeError(f"Analysis requires a captured parameter source: {path}")
    return optional_parameter_float(parameters, path)


def read_confusion_matrix(
    parameters: ParameterContext | None,
) -> NDArray[np.float64] | None:
    """Decode optional $qubit.ext.readout_cal.confusion_matrix as a float array.

    Missing/null calibration leaves return None. Missing capture raises
    RuntimeError; invalid aliases, sources or non-array data fail explicitly.
    The experiment core owns matrix shape and classification validation.
    """
    value = _analysis_value(parameters, "$qubit.ext.readout_cal.confusion_matrix")
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)


def read_chi_kappa(parameters: ParameterContext | None) -> tuple[float, float]:
    """Read required $qubit.ext.chi and $resonator.kappa in MHz from the Run.

    Missing capture or calibration raises RuntimeError naming the canonical path.
    Invalid scalar values, aliases and unavailable sources propagate.
    """
    return (
        _require_float(parameters, "$qubit.ext.chi"),
        _require_float(parameters, "$resonator.kappa"),
    )


def _require_float(parameters: ParameterContext | None, path: str) -> float:
    value = read_analysis_float(parameters, path)
    if value is None:
        raise RuntimeError(f"Missing parameter calibration: {path}")
    return value
