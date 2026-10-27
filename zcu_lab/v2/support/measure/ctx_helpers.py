"""Fixed parameter scalar reads and shared calibration starting-value policies."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import TYPE_CHECKING

from zcu_tools.analysis.flux import DeviceFluxPoint
from zcu_tools.cfg_model import validate_device_name
from zcu_tools.datafile import JsonObject

from zcu_lab.components.qubit import QubitBaseSchema
from zcu_lab.v2.support.flux import (
    device_flux_points,
    qubit_flux_frame,
    snapshot_numeric_values,
)

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


def flux_pick_initial_points(
    parameters: ParameterContext | None,
    device_name: str,
    snapshot: Mapping[str, JsonObject] | None,
) -> DeviceFluxPoint | None:
    """Return scalar picker's half/integer lines from the fixed Run capture.

    device_name is the acquired cfg's scanned device, not flux_source.
    snapshot is the Run's native device settings; only other contributors'
    numeric values are used. No capture or no selected calibration returns
    None, leaving the picker owner's defaults. Invalid names, non-qubit target,
    inconsistent calibration or missing/invalid contributor values raise
    ValueError naming the device or target. Alias/source errors propagate.
    No live state or files are read and no inputs are changed.
    """
    validate_device_name(device_name)
    if parameters is None:
        return None
    target = parameters.resolve_path("$qubit")
    component = parameters.document().components.get(target)
    if not isinstance(component, QubitBaseSchema):
        raise ValueError(f"{target!r}: flux picking requires a qubit component")
    qubit_flux_frame(component.flux)
    if device_name not in component.flux:
        return None
    values = snapshot_numeric_values(snapshot, component.flux)
    return device_flux_points(component.flux, device_name, values)


def t1_initial_time(
    parameters: ParameterContext | None,
    *,
    multiplier: float,
    with_tone: bool = False,
    qubit: str = "$qubit",
) -> float | None:
    """Choose an editable T1-derived time in microseconds from a fixed capture.

    parameters is the Open/Reset capture, or None when no capture exists.
    qubit is a direct component name or captured alias (default $qubit).
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
    paths = (
        (f"{qubit}.ext.t1_with_tone", f"{qubit}.t1") if with_tone else (f"{qubit}.t1",)
    )
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
