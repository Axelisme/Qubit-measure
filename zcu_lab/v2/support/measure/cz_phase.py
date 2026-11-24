"""Bind ordered CZ phase calibration to a fixed two-qubit Form capture."""

from zcu_tools.cfg_model import Expr
from zcu_tools.gui.form_adapter import FormContext

from .paired_qubits import CapturedQubits


def captured_phase_pair(ctx: FormContext, qubits: CapturedQubits) -> dict[str, Expr]:
    """Seed editable angle references after verifying the ordered source names.

    qubits must come from the same Open/Reset capture with a coupler bound.
    Raise ValueError naming the position when its payload name is missing or
    does not match the bound component. Missing angles stay Invalid in Form.
    """
    parameters = ctx.parameters
    if parameters is None:
        raise ValueError("CZ phase pair requires a fixed parameter capture")
    root = "$coupler.ext.cz_phase"
    for position, name in (("qubit1", qubits.qubit1), ("qubit2", qubits.qubit2)):
        # Persistent names: $coupler.ext.cz_phase.qubit1 and $coupler.ext.cz_phase.qubit2.
        path = f"{root}.{position}"
        source = parameters.lookup(path)
        if not source.present:
            raise ValueError(f"{path} is missing; expected ${position}={name!r}")
        if source.value != name:
            raise ValueError(
                f"{path} names {source.value!r}; expected ${position}={name!r}"
            )
    # Persistent names: $coupler.ext.cz_phase.correction_qubit1_deg
    # and $coupler.ext.cz_phase.correction_qubit2_deg.
    return {
        "correction_qubit1_deg": Expr(f"{root}.correction_qubit1_deg"),
        "correction_qubit2_deg": Expr(f"{root}.correction_qubit2_deg"),
    }
