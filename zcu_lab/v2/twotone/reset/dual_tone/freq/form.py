"""Core-derived starting values for reset/dual_tone/freq."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.twotone.reset.dual_tone.freq.core import FreqCfg


def make_form(ctx: FormContext) -> Form[FreqCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        FreqCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "init_pulse": Slot("$qubit.module.x180"),
            "tested_reset": Template(
                "two_pulse_reset",
                bindings={"qubit": "$qubit"},
                set={
                    # Run replaces these scan coordinates; other reset inputs remain.
                    "pulse1_cfg.freq": Hidden(0.0),
                    "pulse2_cfg.freq": Hidden(0.0),
                    "pulse1_cfg.gain": Expr("$qubit.ext.reset_gain1"),
                    "pulse2_cfg.gain": Expr("$qubit.ext.reset_gain2"),
                },
            ),
            "readout": Slot("$qubit.module.readout")
            if ctx.has("$qubit.module.readout")
            else Template("readout", bindings={"resonator": "$resonator"}),
            "freq1": Range.around("$qubit.ext.reset_f1", 50.0, expts=201),
            "freq2": Range.around("$qubit.ext.reset_f2", 50.0, expts=201),
            "method": Hidden("hard"),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    )
