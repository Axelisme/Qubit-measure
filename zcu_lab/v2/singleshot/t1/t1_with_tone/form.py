"""Starting values for singleshot/t1/t1_with_tone; sources are captured at Open/Reset."""

from zcu_tools.cfg_model import Expr, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.singleshot.t1.t1_with_tone.core import T1WithToneCfg
from zcu_lab.v2.support.measure.ctx_helpers import t1_initial_time


def make_form(ctx: FormContext) -> Form[T1WithToneCfg]:
    """Build starting values and shape from the captured source availability."""
    relax_delay = t1_initial_time(ctx.parameters, multiplier=5.0)
    sweep_stop = t1_initial_time(ctx.parameters, multiplier=5.0, with_tone=True)
    return Form(
        T1WithToneCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "init_pulse": Slot("$qubit.module.x180"),
            "pi_pulse": Slot("$qubit.module.x180")
            if ctx.has("$qubit.module.x180")
            else Template("pi_pulse", bindings={"qubit": "$qubit"}),
            "probe_pulse": Template("res_probe", bindings={"resonator": "$resonator"}),
            "readout": Slot("$qubit.module.readout")
            if ctx.has("$qubit.module.readout")
            else Template("readout", bindings={"resonator": "$resonator"}),
            "length": Range(0.0, sweep_stop, expts=101),
            "g_center": Expr("$qubit.ext.readout_cal.g_center"),
            "e_center": Expr("$qubit.ext.readout_cal.e_center"),
            "radius": Expr("$qubit.ext.readout_cal.radius"),
            "relax_delay": relax_delay,
            "uniform": False,
            "reps": 1000,
            "rounds": 10,
        },
    )
