"""Core-derived starting values for reset/single_tone/freq."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.twotone.reset.single_tone.freq.core import FreqCfg


def make_form(ctx: FormContext) -> Form[FreqCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        FreqCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "init_pulse": Slot("$qubit.module.x180"),
                "tested_reset": Template(
                    "reset",
                    bindings={"qubit": "$qubit"},
                    set={
                        # Run replaces these scan coordinates; other reset inputs remain.
                        "pulse_cfg.freq": Hidden(0.0),
                        "pulse_cfg.gain": 0.3,
                        "pulse_cfg.waveform.length": 5.0,
                        "pulse_cfg.post_delay": Expr(
                            "5.0 / (2 * 3.141592653589793 * $resonator.kappa)"
                        ),
                    },
                ),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout", bindings={"resonator": "$resonator"}),
            },
            "sweep": {
                "freq": Range.around(
                    "$qubit.ext.reset_f", "1.5 * $qubit.ext.reset_fwhm", expts=201
                )
            },
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
            "initial_delay": Hidden(),
        },
    )
