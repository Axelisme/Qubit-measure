"""Fixed-capture conditional Ramsey Form, with editable calibrated pulses."""

from zcu_tools.cfg_model import Hidden, Range, Slot
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.mux_forms import (
    mux_readout_initial,
)
from zcu_lab.v2.support.measure.paired_qubits import TwoQubitForm, captured_qubits

from .core import IdleZZCfg


def make_form(ctx: FormContext) -> Form[IdleZZCfg]:
    """Seed conditional Ramsey from GUI aliases $qubit1, $qubit2 and $mux.

    The fixed capture seeds hidden tone indices; no $coupler binding is needed.
    Notebook bindings of the same names are separate. Missing or inconsistent
    qubit/mux bindings raise ValueError. Missing Xπ or Xπ/2 stays unfilled.
    Readout may be manually calibrated if no slot exists.
    Optional DC source starts disabled; enabling it requires a direct connected device
    name, working unit and authorized source-coordinate sweep. Open/Reset is the
    only time sources are read; Run and historical analysis never consult it.
    """
    qubits = captured_qubits(ctx)
    readout = mux_readout_initial(ctx)
    return TwoQubitForm(
        IdleZZCfg,
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
            "x90_qubit1": Slot("$qubit1.module.x90"),
            "x90_qubit2": Slot("$qubit2.module.x90"),
            "readout": readout,
            "detune": None,
            "delay": Range(0, 10, expts=101),
            "dc_device": None,
            "dc": Range(0, 0, expts=1),
        },
    )
