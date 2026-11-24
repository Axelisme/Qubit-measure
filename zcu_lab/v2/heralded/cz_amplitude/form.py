"""Fixed-capture inputs for the four measured CZ return branches."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source
from zcu_lab.v2.support.measure.mux_forms import mux_readout_initial
from zcu_lab.v2.support.measure.paired_qubits import TwoQubitForm, captured_qubits

from .core import HeraldedCZAmplitudeCfg


def make_form(ctx: FormContext) -> Form[HeraldedCZAmplitudeCfg]:
    """Seed editable pulses and readout from fixed qubit and coupler bindings.

    Qubit1/Qubit2 require circle classifiers with radius from GE Post. Reset starts None.

    Existing coupler CZ slots supply library references. Otherwise
    explicit cosine inputs require operator frequency/gain/duration/Nyquist zone;
    only generator wiring comes from capture. No simulator truth or drive preset
    supplies missing calibrations. A common scale range is required explicitly.
    """
    qubits = captured_qubits(ctx, require_coupler=True)
    drive = {}
    for role in ("qubit1", "qubit2"):
        # CZ slots: $coupler.module.cz_qubit1 and $coupler.module.cz_qubit2.
        slot = f"$coupler.module.cz_{role}"
        drive[role] = source(
            ctx,
            slot,
            Template(
                "pulse:blank",
                set={
                    "ch": Expr(f"${role}.wiring.ch"),
                    "nqz": None,
                    "freq": None,
                    "gain": None,
                    "waveform.style": "cosine",
                    "waveform.length": None,
                },
            ),
        )
    readout = mux_readout_initial(ctx)
    return TwoQubitForm(
        HeraldedCZAmplitudeCfg,
        readout_of=lambda cfg: cfg.readout,
        tree={
            "reps": 1000,
            "rounds": 10,
            "relax_delay": 1.0,
            "qubit1_tone": Hidden(qubits.qubit1_tone),
            "qubit2_tone": Hidden(qubits.qubit2_tone),
            "reset": None,
            "drive": drive,
            "x180_qubit1": Slot("$qubit1.module.x180"),
            "x180_qubit2": Slot("$qubit2.module.x180"),
            "readout": readout,
            "scale": Range(None, None, expts=101),
        },
    )
