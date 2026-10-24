"""Core-derived starting values for reset/bath/length."""

from zcu_tools.cfg_model import Expr, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.twotone.reset.bath.length.core import LengthCfg


def make_form(ctx: FormContext) -> Form[LengthCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        LengthCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "init_pulse": Slot("$qubit.module.x180"),
            "tested_reset": Template(
                "bath_reset",
                bindings={"qubit": "$qubit", "resonator": "$resonator"},
                set={
                    "cavity_tone_cfg.freq": Expr("$qubit.ext.reset_bath_freq"),
                    "cavity_tone_cfg.gain": Expr("$qubit.ext.reset_bath_gain"),
                },
            ),
            "readout": Slot("$qubit.module.readout")
            if ctx.has("$qubit.module.readout")
            else Template("readout", bindings={"resonator": "$resonator"}),
            "length": Range.between(0.05, 15.0, expts=201),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    )
