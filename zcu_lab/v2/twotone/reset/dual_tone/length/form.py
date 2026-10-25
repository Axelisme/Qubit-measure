"""Core-derived starting values for reset/dual_tone/length."""

from zcu_tools.cfg_model import Expr, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.twotone.reset.dual_tone.length.core import LengthCfg


def make_form(ctx: FormContext) -> Form[LengthCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        LengthCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "init_pulse": Slot("$qubit.module.x180"),
            "tested_reset": Template(
                "two_pulse_reset",
                bindings={"qubit": "$qubit"},
                set={
                    "pulse1_cfg.freq": Expr("$qubit.ext.reset_f1"),
                    "pulse2_cfg.freq": Expr("$qubit.ext.reset_f2"),
                    "pulse1_cfg.gain": Expr("$qubit.ext.reset_gain1"),
                    "pulse2_cfg.gain": Expr("$qubit.ext.reset_gain2"),
                },
            ),
            "readout": Slot("$resonator.module.readout")
            if ctx.has("$resonator.module.readout")
            else Template("readout", bindings={"resonator": "$resonator"}),
            "length": Range(0.05, 40.0, expts=51),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    )
