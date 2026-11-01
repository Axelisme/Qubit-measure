"""Core-derived starting values for reset/rabi_check."""

from dataclasses import replace

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time
from zcu_lab.v2.twotone.reset.rabi_check.core import RabiCheckCfg


def make_form(ctx: FormContext) -> Form[RabiCheckCfg]:
    """Build starting values and shape from the captured source availability."""
    rabi_pulse = source(
        ctx,
        "$qubit.module.x180",
        Template(
            "pi_pulse", bindings={"qubit": "$qubit"}, set={"gain": Hidden(0.0)}
        ),
    )
    rabi_pulse = replace(rabi_pulse, set={**rabi_pulse.set, "gain": Hidden(0.0)})
    return Form(
        RabiCheckCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "rabi_pulse": rabi_pulse,
            "tested_reset": source(
                ctx,
                "$qubit.module.reset",
                Template("reset", bindings={"qubit": "$qubit"}),
            ),
            "readout": source(
                ctx,
                "$resonator.module.readout",
                Template("readout", bindings={"resonator": "$resonator"}),
            ),
            "gain": Range(0.0, 1.0, expts=51),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    )
