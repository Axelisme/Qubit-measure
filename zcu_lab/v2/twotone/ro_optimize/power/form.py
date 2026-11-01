"""Core-derived starting values for ro_optimize/power."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.twotone.ro_optimize.power.core import PowerCfg


def make_form(ctx: FormContext) -> Form[PowerCfg]:
    """Build from captured sources; wait starts at 0.01*T1, relax_delay at 5*T1."""
    return Form(
        PowerCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "qub_pulse": Slot(
                "$qubit.module.x180",
                fallback=Template("pi_pulse", bindings={"qubit": "$qubit"}),
            ),
            "readout": Slot(
                "$resonator.module.readout",
                set={"gain": Hidden(0.0)},
                fallback=Template(
                    "readout",
                    bindings={"resonator": "$resonator"},
                    set={"gain": Hidden(0.0)},
                ),
            ),
            "gain": Range(0.001, 0.2, expts=101),
            "wait": t1_initial_time(ctx.parameters, multiplier=0.01),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    )
