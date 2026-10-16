"""Core-derived starting values for ro_optimize/auto_optimize."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.twotone.ro_optimize.auto_optimize.core import AutoOptCfg


def make_form(ctx: FormContext) -> Form[AutoOptCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        AutoOptCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "qub_pulse": Slot("$qubit.module.x180")
                if ctx.has("$qubit.module.x180")
                else Template("pi_pulse"),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout"),
            },
            "sweep": {
                "freq": Range.around(
                    "$resonator.freq", "0.2 * $resonator.kappa", expts=51
                ),
                "gain": Range.between(0.1, 0.25, expts=51),
                "length": Range.between(5.0, 10.0, expts=51),
            },
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
            "num_points": 1001,
            "initial_delay": Hidden(),
        },
    )
