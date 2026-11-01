"""Core-derived starting values for ro_optimize/freq."""

from dataclasses import replace

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time
from zcu_lab.v2.twotone.ro_optimize.freq.core import FreqCfg


def make_form(ctx: FormContext) -> Form[FreqCfg]:
    """Build starting values and shape from the captured source availability."""
    readout = source(
        ctx,
        "$resonator.module.readout",
        Template(
            "readout",
            bindings={"resonator": "$resonator"},
        ),
    )
    readout = replace(readout, set={**readout.set, "freq": Hidden(0.0)})
    return Form(
        FreqCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "qub_pulse": source(
                ctx,
                "$qubit.module.x180",
                Template("pi_pulse", bindings={"qubit": "$qubit"}),
            ),
            "readout": readout,
            "freq": Range.around("$resonator.freq", "$resonator.kappa", expts=301),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    )
