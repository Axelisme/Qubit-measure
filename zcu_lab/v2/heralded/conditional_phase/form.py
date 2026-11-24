"""Fixed-capture inputs for conditional phase calibration."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source
from zcu_lab.v2.support.measure.mux_forms import mux_readout_seed_metadata
from zcu_lab.v2.support.measure.paired_qubits import TwoQubitForm, captured_qubits

from .core import ConditionalPhaseHeraldedCfg


def make_form(ctx: FormContext) -> Form[ConditionalPhaseHeraldedCfg]:
    """Seed calibrated pulses and hidden tone positions.

    Missing CZ slots stay required. Both qubit tones need circle + radius from
    GE Post before Run. No other experiment Form or simulator truth supplies inputs.
    reset starts as None and can be authored explicitly.
    """
    qubits = captured_qubits(ctx, require_coupler=True)
    readout = source(
        ctx,
        "$mux.module.readout",
        Template("readout/mux:blank"),
    )
    return TwoQubitForm(
        ConditionalPhaseHeraldedCfg,
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
            "drive": {
                # Keep the refs; hide the frequency Run overrides in inline.
                "qubit1": Slot("$coupler.module.cz_qubit1", set={"freq": Hidden(0.0)}),
                "qubit2": Slot("$coupler.module.cz_qubit2", set={"freq": Hidden(0.0)}),
            },
            "frequency": Range(None, None, expts=101),
        },
    ).with_seed_metadata(mux_readout_seed_metadata(readout))
