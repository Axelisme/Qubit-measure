"""Core-derived starting values for ro_optimize/length."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.twotone.ro_optimize.length.core import LengthCfg


def make_form(ctx: FormContext) -> Form[LengthCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        LengthCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "qub_pulse": Slot("$qubit.module.x180")
                if ctx.has("$qubit.module.x180")
                else Template("pi_pulse", bindings={"qubit": "$qubit"}),
                # Run supplies ADC integration window; waveform length remains a shape baseline.
                "readout": Slot("$qubit.module.readout", set={"ro_length": Hidden(0.0)})
                if ctx.has("$qubit.module.readout")
                else Template(
                    "readout",
                    bindings={"resonator": "$resonator"},
                    set={"ro_length": Hidden(0.0)},
                ),
            },
            "sweep": {"length": Range(0.01, 3.5, expts=51)},
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
            "initial_delay": Hidden(),
        },
    )
