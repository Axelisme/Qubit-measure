"""Fixed-capture calibration and editable inputs for CZ optimization."""

from zcu_tools.cfg_model import Hidden, Slot
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.cz_phase import captured_phase_pair
from zcu_lab.v2.support.measure.mux_forms import mux_readout_seed_metadata
from zcu_lab.v2.support.measure.paired_qubits import TwoQubitForm, captured_qubits

from .core import CZOptimizeCfg


def make_form(ctx: FormContext) -> Form[CZOptimizeCfg]:
    """Capture calibration slots and ordered CZ corrections at Open or Reset.

    Missing sources remain invalid, and the optional reset starts unset.
    Operators fill the fixed depth, sequence seeds and CMA settings before Run.
    """
    qubits = captured_qubits(ctx, require_coupler=True)
    readout = Slot("$mux.module.readout")
    return TwoQubitForm(
        CZOptimizeCfg,
        readout_of=lambda cfg: cfg.readout,
        tree={
            "reps": 1000,
            "relax_delay": 1.0,
            "qubit1_tone": Hidden(qubits.qubit1_tone),
            "qubit2_tone": Hidden(qubits.qubit2_tone),
            "reset": None,
            "x90_qubit1": Slot("$qubit1.module.x90"),
            "x90_qubit2": Slot("$qubit2.module.x90"),
            "x180_qubit1": Slot("$qubit1.module.x180"),
            "x180_qubit2": Slot("$qubit2.module.x180"),
            "drive": {
                "qubit1": Slot("$coupler.module.cz_qubit1"),
                "qubit2": Slot("$coupler.module.cz_qubit2"),
            },
            "readout": readout,
            "phase_pair": captured_phase_pair(ctx, qubits),
        },
    ).with_seed_metadata(mux_readout_seed_metadata(readout))
