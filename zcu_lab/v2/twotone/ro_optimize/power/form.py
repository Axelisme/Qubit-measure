"""Core-derived starting values for ro_optimize/power."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.twotone.ro_optimize.power.core import PowerCfg


def make_form(ctx: FormContext) -> Form[PowerCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        PowerCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "qub_pulse": Slot("$qubit.module.x180")
                if ctx.has("$qubit.module.x180")
                else Template("pi_pulse", bindings={"qubit": "$qubit"}),
                # Run supplies gain.
                "readout": Slot("$qubit.module.readout", set={"gain": Hidden(0.0)})
                if ctx.has("$qubit.module.readout")
                else Template(
                    "readout",
                    bindings={"resonator": "$resonator"},
                    set={"gain": Hidden(0.0)},
                ),
            },
            "sweep": {"gain": Range.between(0.001, 0.2, expts=101)},
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 100,
            "initial_delay": Hidden(),
        },
    )
