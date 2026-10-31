"""Fixed-capture inputs for paired two-qubit RB."""

from typing import TypeVar

from zcu_tools.cfg_model import Expr, Slot
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.mux_forms import mux_readout_seed_metadata

from .._measure import captured_roles
from .core import RBCfg

T_Cfg = TypeVar("T_Cfg", bound=RBCfg)


def rb_form(ctx: FormContext, cfg_type: type[T_Cfg]) -> Form[T_Cfg]:
    """Capture role identities, X90/CZ/readout slots and required named phase pair.

    Missing pulse/readout slots or correction fields stay Invalid, never zero
    or simulator defaults. Circle centers and radii come from the saved role-tone readout classifiers.
    All pulse and correction initial values remain operator-overridable; Run
    uses the accepted fixed capture, not current aliases or library.
    """
    names: dict[str, str] = {}
    if ctx.parameters is not None:
        for role in ("f1", "f2"):
            path = f"$coupler.ext.cz_phase.{role}"
            captured = ctx.parameters.lookup(path)
            if captured.present:
                if not isinstance(captured.value, str):
                    raise ValueError(f"{path} must contain a component name")
                names[role] = captured.value
    readout = Slot("$mux.module.readout")
    return Form(
        cfg_type,
        field_order=(
            "roles",
            "reset",
            "x90_f1",
            "x90_f2",
            "readout",
            "drive",
            "phase_pair",
            "depths",
            "seeds",
            "reps",
            "rounds",
            "relax_delay",
            "initial_delay",
        ),
        tree={
            "reps": 1000,
            "rounds": 10,
            "roles": captured_roles(ctx),
            "reset": None,
            "x90_f1": Slot("$f1.module.x90", copy=True),
            "x90_f2": Slot("$f2.module.x90", copy=True),
            "readout": readout,
            "drive": {
                "f1": Slot("$coupler.module.cz_f1", copy=True),
                "f2": Slot("$coupler.module.cz_f2", copy=True),
            },
            "phase_pair": {
                **names,
                "correction_f1_deg": Expr("$coupler.ext.cz_phase.correction_f1_deg"),
                "correction_f2_deg": Expr("$coupler.ext.cz_phase.correction_f2_deg"),
            },
        },
    ).with_seed_metadata(mux_readout_seed_metadata(readout))


def make_form(ctx: FormContext) -> Form[RBCfg]:
    """Capture paired RB inputs with optional reset initially disabled."""
    return rb_form(ctx, RBCfg)
