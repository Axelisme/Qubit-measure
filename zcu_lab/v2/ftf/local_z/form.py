"""Fixed-capture inputs for a two-target local-Z experiment."""

from zcu_tools.cfg_model import Expr, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from .._measure import captured_roles
from .core import LocalZCfg


def make_form(ctx: FormContext) -> Form[LocalZCfg]:
    """Capture identities and copy calibrated pulses; absent CZ slots stay required.

    Missing readout may be explicitly calibrated in the complete mux template.
    Missing GE centers permit raw acquisition but analysis explains missing source
    calibration. No other experiment Form or simulator truth supplies inputs.
    """
    roles = captured_roles(ctx)
    return Form(
        LocalZCfg,
        tree={
            "roles": roles,
            "modules": {
                "x90_f1": Slot("$f1.module.x90", copy=True),
                "x90_f2": Slot("$f2.module.x90", copy=True),
                "readout": Slot("$mux.module.readout")
                if ctx.has("$mux.module.readout")
                else Template("readout/mux:blank"),
            },
            "drive": {
                "f1": Slot("$coupler.module.cz_f1", copy=True),
                "f2": Slot("$coupler.module.cz_f2", copy=True),
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
