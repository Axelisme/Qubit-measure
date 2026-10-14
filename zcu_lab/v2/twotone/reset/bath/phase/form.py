"""Core-derived starting values for reset/bath/phase."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.twotone.reset.bath.phase.core import PhaseCfg


def make_form(ctx: FormContext) -> Form[PhaseCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        PhaseCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "init_pulse": Slot("$qubit.module.x180"),
                "tested_reset": Template(
                    "bath_reset",
                    set={
                        "cavity_tone_cfg.freq": Expr("$qubit.ext.reset_bath_freq"),
                        "cavity_tone_cfg.gain": Expr("$qubit.ext.reset_bath_gain"),
                    },
                ),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout"),
            },
            "sweep": {"phase": Range.between(-360.0, 360.0, expts=201)},
            "relax_delay": 10.5,
            "reps": 100,
            "rounds": 1000,
            "initial_delay": Hidden(),
        },
    )
