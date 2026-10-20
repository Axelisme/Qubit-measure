"""Starting values for singleshot/reset_check; sources are captured at Open/Reset."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.singleshot.reset_check.core import ResetCheckCfg


def make_form(ctx: FormContext) -> Form[ResetCheckCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        ResetCheckCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "rabi_pulse": Slot("$qubit.module.x180")
                if ctx.has("$qubit.module.x180")
                else Template("pi_pulse", bindings={"qubit": "$qubit"}),
                "tested_reset": Slot("$qubit.module.reset")
                if ctx.has("$qubit.module.reset")
                else Template("reset", bindings={"qubit": "$qubit"}),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout", bindings={"resonator": "$resonator"}),
            },
            "sweep": {
                "gain": Range.between(0.0, "4.0 * $qubit.module.x180.gain", expts=51)
            },
            "g_center": Expr("$qubit.ext.readout_cal.g_center"),
            "e_center": Expr("$qubit.ext.readout_cal.e_center"),
            "radius": Expr("$qubit.ext.readout_cal.radius"),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 5000,
            "rounds": 1,
            "initial_delay": Hidden(),
        },
    )
