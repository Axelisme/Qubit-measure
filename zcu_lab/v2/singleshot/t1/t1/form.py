"""Starting values for singleshot/t1/t1; sources are captured at Open/Reset."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.singleshot.t1.t1.core import T1Cfg


def make_form(ctx: FormContext) -> Form[T1Cfg]:
    """Build starting values and shape from the captured source availability."""
    relax_delay = t1_initial_time(ctx.parameters, multiplier=5.0)
    return Form(
        T1Cfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "pi_pulse": Slot("$qubit.module.x180")
                if ctx.has("$qubit.module.x180")
                else Template("pi_pulse", bindings={"qubit": "$qubit"}),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout", bindings={"resonator": "$resonator"}),
            },
            "sweep": {"length": Range.between(0.0, relax_delay, expts=101)},
            "g_center": Expr("$qubit.ext.readout_cal.g_center"),
            "e_center": Expr("$qubit.ext.readout_cal.e_center"),
            "radius": Expr("$qubit.ext.readout_cal.radius"),
            "relax_delay": relax_delay,
            "uniform": False,
            "reps": 1000,
            "rounds": 10,
            "initial_delay": Hidden(),
        },
    )
