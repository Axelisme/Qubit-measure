"""Read qubit calibration blocks from fixed measurement parameter captures."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zcu_lab.components.qubit import QubitBaseSchema, QubitFluxCalibration
from zcu_lab.v2.support.flux import qubit_flux_frame

if TYPE_CHECKING:
    from zcu_tools.gui.session.parameters import ParameterContext


def captured_qubit_flux(
    parameters: ParameterContext | None, qubit: str = "$qubit"
) -> dict[str, QubitFluxCalibration]:
    """Copy one qubit's validated flux block from a fixed Run/Start capture.

    qubit is a concrete component name or captured alias, defaulting to $qubit.
    Return detached device-keyed calibrations in native units, including signed
    periods. An empty block means this captured qubit has no calibration.
    Missing capture or a non-qubit target raises ValueError; alias/source errors
    propagate. Invalid frame spacing or device names raise ValueError naming
    the device. No live point, file or device is read, and no calibration changes.
    """
    if parameters is None:
        raise ValueError("Record flux frame requires a parameter capture")
    target = parameters.resolve_path(qubit)
    component = parameters.document().components.get(target)
    if not isinstance(component, QubitBaseSchema):
        raise ValueError(f"{target!r}: record flux frame requires a qubit component")
    qubit_flux_frame(component.flux)
    return {
        name: calibration.model_copy(deep=True)
        for name, calibration in component.flux.items()
    }
