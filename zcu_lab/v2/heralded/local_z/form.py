"""Fixed-capture inputs for heralded Local Z."""

from zcu_tools.cfg_model import Hidden, Slot
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.mux_forms import mux_readout_initial
from zcu_lab.v2.support.measure.paired_qubits import TwoQubitForm, captured_qubits

from .core import LocalZHeraldedCfg


def make_form(ctx: FormContext) -> Form[LocalZHeraldedCfg]:
    """Use captured tone positions and the paired qubit and coupler slots.

    Keep the full mux readout. Its two role tones need GE circles and GE Post
    radii. Reset starts empty but may be set to a preparation pulse reset.
    """
    qubits = captured_qubits(ctx, require_coupler=True)
    return TwoQubitForm(
        LocalZHeraldedCfg,
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
            "readout": mux_readout_initial(ctx),
            "drive": {
                "qubit1": Slot("$coupler.module.cz_qubit1"),
                "qubit2": Slot("$coupler.module.cz_qubit2"),
            },
        },
    )
