"""Starting values for singleshot/mist/pre_freq, captured at Open/Reset."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.singleshot.mist.pre_freq.core import PreFreqCfg


def make_form(ctx: FormContext) -> Form[PreFreqCfg]:
    """Prepare a qubit-frequency sweep before the fixed resonator MIST probe."""
    return Form(
        PreFreqCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "init_pulse": Template("qub_probe", bindings={"qubit": "$qubit"}),
                "pi_pulse": Slot("$qubit.module.x180"),
                "probe_pulse": Template(
                    "res_probe", bindings={"resonator": "$resonator"}
                ),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout", bindings={"resonator": "$resonator"}),
            },
            "sweep": {
                "freq": Range.around("$qubit.freq", "1.5 * $qubit.kappa", expts=51)
            },
            "g_center": Expr("$qubit.ext.readout_cal.g_center"),
            "e_center": Expr("$qubit.ext.readout_cal.e_center"),
            "radius": Expr("$qubit.ext.readout_cal.radius"),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 10000,
            "rounds": 1,
            "initial_delay": Hidden(),
        },
    )
