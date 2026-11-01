"""Core-derived starting values for ro_optimize/power."""

from dataclasses import replace

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time
from zcu_lab.v2.twotone.ro_optimize.power.core import PowerCfg


def make_form(ctx: FormContext) -> Form[PowerCfg]:
    """Build from captured sources; wait starts at 0.01*T1, relax_delay at 5*T1."""
    readout = source(
        ctx,
        "$resonator.module.readout",
        Template(
            "readout",
            bindings={"resonator": "$resonator"},
        ),
    )
    readout = replace(readout, set={**readout.set, "gain": Hidden(0.0)})
    return Form(
        PowerCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "qub_pulse": source(
                ctx,
                "$qubit.module.x180",
                Template("pi_pulse", bindings={"qubit": "$qubit"}),
            ),
            "readout": readout,
            "gain": Range(0.001, 0.2, expts=101),
            "wait": t1_initial_time(ctx.parameters, multiplier=0.01),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    )
