"""Fixed-capture inputs for paired two-qubit RB."""

from zcu_tools.cfg_model import Expr, Slot
from zcu_tools.gui.form_adapter import Form, FormContext

from .._measure import captured_roles
from .core import RBCfg


def make_form(ctx: FormContext) -> Form[RBCfg]:
    """Capture role identities, X90/CZ/readout slots and required named phase pair.

    Missing pulse/readout slots or correction fields stay Invalid, never zero
    or simulator defaults. GE centers are optional saved analysis inputs.
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
    return Form(
        RBCfg,
        tree={
            "roles": captured_roles(ctx),
            "modules": {
                "x90_f1": Slot("$f1.module.x90", copy=True),
                "x90_f2": Slot("$f2.module.x90", copy=True),
                "readout": Slot("$mux.module.readout", copy=True),
            },
            "drive": {
                "f1": Slot("$coupler.module.cz_f1", copy=True),
                "f2": Slot("$coupler.module.cz_f2", copy=True),
            },
            "phase_pair": {
                **names,
                "correction_f1_deg": Expr("$coupler.ext.cz_phase.correction_f1_deg"),
                "correction_f2_deg": Expr("$coupler.ext.cz_phase.correction_f2_deg"),
            },
            **{
                f"{role}_centers": {
                    "g": Expr(f"${role}.ext.readout_cal.g_center"),
                    "e": Expr(f"${role}.ext.readout_cal.e_center"),
                }
                if ctx.parameters is not None
                and ctx.parameters.lookup(f"${role}.ext.readout_cal.g_center").present
                and ctx.parameters.lookup(f"${role}.ext.readout_cal.e_center").present
                else None
                for role in ("f1", "f2")
            },
        },
    )
