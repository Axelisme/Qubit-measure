"""Starting values for singleshot/t1/t1; sources are captured at Open/Reset."""

from zcu_tools.cfg_model import Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.singleshot.t1.t1.core import T1Cfg
from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time


def make_form(ctx: FormContext) -> Form[T1Cfg]:
    """Build starting values and shape from the captured source availability."""
    relax_delay = t1_initial_time(ctx.parameters, multiplier=5.0)
    return Form(
        T1Cfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "pi_pulse": source(
                ctx,
                "$qubit.module.x180",
                Template("pi_pulse", bindings={"qubit": "$qubit"}),
            ),
            "readout": source(
                ctx,
                "$resonator.module.readout",
                Template("readout", bindings={"resonator": "$resonator"}),
            ),
            "delay": Range(0.0, relax_delay, expts=101),
            "relax_delay": relax_delay,
            "sampling": "decay",
            "reps": 1000,
            "rounds": 10,
        },
    )
