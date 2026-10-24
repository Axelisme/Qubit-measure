"""Core-derived starting values for reset/dual_tone/power."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.twotone.reset.dual_tone.power.core import PowerCfg


def make_form(ctx: FormContext) -> Form[PowerCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        PowerCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "init_pulse": Slot("$qubit.module.x180"),
            "tested_reset": Template(
                "two_pulse_reset",
                bindings={"qubit": "$qubit"},
                set={
                    # Run replaces these scan coordinates; other reset inputs remain.
                    "pulse1_cfg.gain": Hidden(0.0),
                    "pulse2_cfg.gain": Hidden(0.0),
                    "pulse1_cfg.freq": Expr("$qubit.ext.reset_f1"),
                    "pulse2_cfg.freq": Expr("$qubit.ext.reset_f2"),
                },
            ),
            "readout": Slot("$qubit.module.readout")
            if ctx.has("$qubit.module.readout")
            else Template("readout", bindings={"resonator": "$resonator"}),
            "gain1": Range(0.0, 1.0, expts=51),
            "gain2": Range(0.0, 1.0, expts=51),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    )
