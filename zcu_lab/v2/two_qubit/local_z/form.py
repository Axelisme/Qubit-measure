"""Fixed-capture inputs for a two-target local-Z experiment."""

from zcu_tools.cfg_model import Hidden, Slot
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.mux_forms import (
    mux_readout_initial,
)
from zcu_lab.v2.support.measure.paired_qubits import TwoQubitForm, captured_qubits

from .core import LocalZCfg


def make_form(ctx: FormContext) -> Form[LocalZCfg]:
    """Hide captured tone indices and seed the calibrated pulses.

    Missing CZ slots stay required. Both role tones need circle + radius from
    GE Post before Run. No other experiment Form or simulator truth supplies inputs.
    reset starts as None and can be authored explicitly.
    """
    qubits = captured_qubits(ctx, require_coupler=True)
    readout = mux_readout_initial(ctx)
    return TwoQubitForm(
        LocalZCfg,
        readout_of=lambda cfg: cfg.readout,
        tree={
            "reps": 1000,
            "rounds": 10,
            "relax_delay": 1.0,
            "qubit1_tone": Hidden(qubits.qubit1_tone),
            "qubit2_tone": Hidden(qubits.qubit2_tone),
            "reset": None,
            "x90_qubit1": Slot("$qubit1.module.x90"),
            "x90_qubit2": Slot("$qubit2.module.x90"),
            "readout": readout,
            "drive": {
                "qubit1": Slot("$coupler.module.cz_qubit1"),
                "qubit2": Slot("$coupler.module.cz_qubit2"),
            },
        },
    )
