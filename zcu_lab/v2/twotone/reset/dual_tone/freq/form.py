"""Core-derived starting values for reset/dual_tone/freq."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.twotone.reset.dual_tone.freq.core import FreqCfg


def make_form(ctx: FormContext) -> Form[FreqCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        FreqCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "init_pulse": Slot("$qubit.module.x180"),
                "tested_reset": Template(
                    "two_pulse_reset",
                    set={
                        "pulse1_cfg.gain": Expr("$qubit.ext.reset_gain1"),
                        "pulse2_cfg.gain": Expr("$qubit.ext.reset_gain2"),
                    },
                ),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout"),
            },
            "sweep": {
                "freq1": Range.around("$qubit.ext.reset_f1", 50.0, expts=201),
                "freq2": Range.around("$qubit.ext.reset_f2", 50.0, expts=201),
            },
            "relax_delay": 0.5,
            "reps": 100,
            "rounds": 1000,
            "initial_delay": Hidden(),
        },
    )
