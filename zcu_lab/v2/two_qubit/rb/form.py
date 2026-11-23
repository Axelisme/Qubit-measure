"""Fixed-capture inputs for paired two-qubit RB."""

from typing import TypeVar

from zcu_tools.cfg_model import Hidden, Slot
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.mux_forms import mux_readout_seed_metadata

from .._cz_phase_form import captured_phase_pair
from .._measure import TwoQubitForm, captured_qubits
from .core import RBCfg
from .core_heralded import RBHeraldedCfg

T_Cfg = TypeVar("T_Cfg", bound=RBCfg | RBHeraldedCfg)


def rb_form(ctx: FormContext, cfg_type: type[T_Cfg]) -> Form[T_Cfg]:
    """Capture role identities, X90/X180/CZ/readout slots and named phase pair.

    Missing pulse/readout slots or correction fields stay Invalid, never zero
    or simulator defaults. Circle centers and radii come from the saved role-tone
    readout classifiers. Reset choices follow the passed cfg's accepted module types.
    All pulse and correction initial values remain operator-overridable; Run
    uses the accepted fixed capture, not current aliases or library.
    """
    qubits = captured_qubits(ctx, require_coupler=True)
    readout = Slot("$mux.module.readout")
    return TwoQubitForm(
        cfg_type,
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
            "x180_qubit1": Slot("$qubit1.module.x180"),
            "x180_qubit2": Slot("$qubit2.module.x180"),
            "readout": readout,
            "drive": {
                "qubit1": Slot("$coupler.module.cz_qubit1"),
                "qubit2": Slot("$coupler.module.cz_qubit2"),
            },
            "phase_pair": captured_phase_pair(ctx, qubits),
        },
    ).with_seed_metadata(mux_readout_seed_metadata(readout))


def make_form(ctx: FormContext) -> Form[RBCfg]:
    """Capture paired RB inputs; the disabled Reset can select a mux herald."""
    return rb_form(ctx, RBCfg)
