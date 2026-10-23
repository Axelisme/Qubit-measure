"""Core-derived starting values for reset/single_tone/length."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.twotone.reset.single_tone.length.core import LengthCfg


def make_form(ctx: FormContext) -> Form[LengthCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        LengthCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "init_pulse": Slot("$qubit.module.x180"),
                "tested_reset": Template(
                    "reset",
                    bindings={"qubit": "$qubit"},
                    set={"pulse_cfg.freq": Expr("$qubit.ext.reset_f")},
                ),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout", bindings={"resonator": "$resonator"}),
            },
            "sweep": {"length": Range.between(0.1, 20.0, expts=50)},
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
            "initial_delay": Hidden(),
        },
    )
