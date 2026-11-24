"""Fixed-capture tone positions and calibration inputs for joint readout."""

from zcu_tools.cfg_model import Hidden, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source
from zcu_lab.v2.support.measure.mux_forms import mux_readout_seed_metadata
from zcu_lab.v2.support.measure.paired_qubits import TwoQubitForm, captured_qubits

from .core import JointReadoutCfg


def make_form(ctx: FormContext) -> Form[JointReadoutCfg]:
    """Seed joint readout from the GUI's $qubit1, $qubit2 and $mux aliases.

    The fixed Open/Reset capture supplies both hidden tone indices. These
    GUI aliases are separate from Notebook's built-in bindings of the same
    names. Missing or inconsistent bindings or a captured tone-count mismatch
    raise ValueError. The capture is not changed. Copy the Xπ slots into
    editable inputs and reference the mux readout slot. Missing pulses remain
    unfilled; without a readout slot, start with a blank readout template.
    """
    qubits = captured_qubits(ctx)
    readout = source(
        ctx,
        "$mux.module.readout",
        Template("readout/mux:blank"),
    )
    return TwoQubitForm(
        JointReadoutCfg,
        readout_of=lambda cfg: cfg.readout,
        tree={
            "reps": 1000,
            "rounds": 10,
            "relax_delay": 1.0,
            "qubit1_tone": Hidden(qubits.qubit1_tone),
            "qubit2_tone": Hidden(qubits.qubit2_tone),
            "reset": None,
            "x180_qubit1": Slot("$qubit1.module.x180"),
            "x180_qubit2": Slot("$qubit2.module.x180"),
            "readout": readout,
        },
    ).with_seed_metadata(mux_readout_seed_metadata(readout))
