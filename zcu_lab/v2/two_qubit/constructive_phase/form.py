"""Fixed-capture duration scan with explicit two-line drive and GE calibration inputs."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source
from zcu_lab.v2.support.measure.mux_forms import mux_readout_seed_metadata

from .._measure import TwoQubitForm, captured_qubits
from .core import ConstructivePhaseCfg


def make_form(ctx: FormContext) -> Form[ConstructivePhaseCfg]:
    """Seed pulses, hidden tone positions and a full turn of relative phase.

    Relative phase stays editable. The operator supplies both duration endpoints.

    Existing CZ slots supply library references, not certified gate settings.
    Missing slots offer explicit cosine templates on captured qubit wiring.
    Missing or linear qubit tone classifiers, or missing radii, stay Invalid. Run GE Post first. Open/Reset never performs acquisition.
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
    readout = source(
        ctx,
        "$mux.module.readout",
        Template("readout/mux:blank"),
    )
    return TwoQubitForm(
        ConstructivePhaseCfg,
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
            "drive": drive,
            "relative_phase": Range(-180.0, 180.0, expts=51),
            "duration": Range(None, None, expts=101),
        },
    ).with_seed_metadata(mux_readout_seed_metadata(readout))
