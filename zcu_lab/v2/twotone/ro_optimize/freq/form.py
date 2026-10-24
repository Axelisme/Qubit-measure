"""Core-derived starting values for ro_optimize/freq."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.twotone.ro_optimize.freq.core import FreqCfg


def make_form(ctx: FormContext) -> Form[FreqCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        FreqCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "qub_pulse": Slot("$qubit.module.x180")
            if ctx.has("$qubit.module.x180")
            else Template("pi_pulse", bindings={"qubit": "$qubit"}),
            "readout": Slot("$qubit.module.readout", set={"freq": Hidden(0.0)})
            if ctx.has("$qubit.module.readout")
            else Template(
                "readout",
                bindings={"resonator": "$resonator"},
                set={"freq": Hidden(0.0)},
            ),
            "freq": Range.around("$resonator.freq", "$resonator.kappa", expts=301),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    )
