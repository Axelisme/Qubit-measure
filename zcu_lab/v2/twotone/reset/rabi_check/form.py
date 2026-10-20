"""Core-derived starting values for reset/rabi_check."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.twotone.reset.rabi_check.core import RabiCheckCfg


def make_form(ctx: FormContext) -> Form[RabiCheckCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        RabiCheckCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                # Run supplies the diagnostic Rabi gain scan.
                "rabi_pulse": Slot("$qubit.module.x180", set={"gain": Hidden(0.0)})
                if ctx.has("$qubit.module.x180")
                else Template(
                    "pi_pulse", bindings={"qubit": "$qubit"}, set={"gain": Hidden(0.0)}
                ),
                "tested_reset": Slot("$qubit.module.reset")
                if ctx.has("$qubit.module.reset")
                else Template("reset", bindings={"qubit": "$qubit"}),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout", bindings={"resonator": "$resonator"}),
            },
            "sweep": {"gain": Range.between(0.0, 1.0, expts=51)},
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
            "initial_delay": Hidden(),
        },
    )
