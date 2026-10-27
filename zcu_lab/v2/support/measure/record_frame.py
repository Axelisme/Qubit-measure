"""Read qubit calibration blocks from fixed measurement parameter captures."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zcu_lab.components.qubit import QubitBaseSchema, QubitFluxSchema

if TYPE_CHECKING:
    from zcu_tools.gui.session.parameters import ParameterContext


def captured_qubit_flux(
    parameters: ParameterContext | None, qubit: str = "$qubit"
) -> QubitFluxSchema:
    """Copy one qubit's validated flux block from a fixed Run/Start capture.

    qubit is a concrete component name or captured alias, defaulting to $qubit.
    Return a detached offset/period block, including signed native periods.
    A null offset means this captured qubit has no calibrated zero-bias flux.
    Missing capture or a non-qubit target raises ValueError; alias/source errors
    propagate. Component validation owns finite periods and device names.
    No live point, file or device is read, and no calibration changes.
    """
    if parameters is None:
        raise ValueError("Record flux frame requires a parameter capture")
    target = parameters.resolve_path(qubit)
    component = parameters.document().components.get(target)
    if not isinstance(component, QubitBaseSchema):
        raise ValueError(f"{target!r}: record flux frame requires a qubit component")
    return component.flux.model_copy(deep=True)
