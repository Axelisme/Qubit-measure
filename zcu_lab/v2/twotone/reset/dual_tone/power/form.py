"""Core-derived starting values for reset/dual_tone/power."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.twotone.reset.dual_tone.power.core import PowerCfg


def make_form(ctx: FormContext) -> Form[PowerCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        PowerCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "init_pulse": Slot("$qubit.module.x180"),
                "tested_reset": Template(
                    "two_pulse_reset",
                    set={
                        "pulse1_cfg.freq": Expr("$qubit.ext.reset_f1"),
                        "pulse2_cfg.freq": Expr("$qubit.ext.reset_f2"),
                    },
                ),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout"),
            },
            "sweep": {
                "gain1": Range.between(0.0, 1.0, expts=51),
                "gain2": Range.between(0.0, 1.0, expts=51),
            },
            "relax_delay": 0.5,
            "reps": 100,
            "rounds": 100,
            "initial_delay": Hidden(),
        },
    )
