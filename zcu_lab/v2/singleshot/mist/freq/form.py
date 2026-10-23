"""Starting values for singleshot/mist/freq; sources are captured at Open/Reset."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.singleshot.mist.freq.core import FreqCfg


def make_form(ctx: FormContext) -> Form[FreqCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        FreqCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "init_pulse": Slot("$qubit.module.x180"),
                "probe_pulse": Template(
                    "res_probe",
                    bindings={"resonator": "$resonator"},
                    # Run supplies these probe scan coordinates.
                    set={"freq": Hidden(0.0)},
                ),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout", bindings={"resonator": "$resonator"}),
            },
            "sweep": {
                "freq": Range.around(
                    "$resonator.freq", "1.5 * $resonator.kappa", expts=51
                )
            },
            "g_center": Expr("$qubit.ext.readout_cal.g_center"),
            "e_center": Expr("$qubit.ext.readout_cal.e_center"),
            "radius": Expr("$qubit.ext.readout_cal.radius"),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
            "initial_delay": Hidden(),
        },
    )
