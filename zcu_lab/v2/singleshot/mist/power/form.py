"""Starting values for singleshot/mist/power; sources are captured at Open/Reset."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.singleshot.mist.power.core import PowerCfg
from zcu_lab.v2.support.measure.ctx_helpers import t1_initial_time


def make_form(ctx: FormContext) -> Form[PowerCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        PowerCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "init_pulse": Slot("$qubit.module.x180"),
            "probe_pulse": Template(
                "res_probe",
                bindings={"resonator": "$resonator"},
                # Run supplies these probe scan coordinates.
                set={"gain": Hidden(0.0)},
            ),
            "readout": Slot("$qubit.module.readout")
            if ctx.has("$qubit.module.readout")
            else Template("readout", bindings={"resonator": "$resonator"}),
            "gain": Range(0.0, 1.0, expts=151),
            "g_center": Expr("$qubit.ext.readout_cal.g_center"),
            "e_center": Expr("$qubit.ext.readout_cal.e_center"),
            "radius": Expr("$qubit.ext.readout_cal.radius"),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    )
