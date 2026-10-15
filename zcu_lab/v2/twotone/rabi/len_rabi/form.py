"""Experiment starting values from fixed captures and injected templates."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from .core import LenRabiCfg


def make_form(ctx: FormContext) -> Form[LenRabiCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        LenRabiCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "qub_pulse": Slot("$qubit.module.x180")
                if ctx.has("$qubit.module.x180")
                else Template("pi_pulse"),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout"),
                "init_pulse": Hidden(),
            },
            "sweep": {
                "length": Range.between(
                    0.03, "4 * $qubit.module.x180.waveform.length", expts=101
                )
            },
            "relax_delay": Expr("5 * $qubit.t1"),
            "initial_delay": Hidden(),
            "reps": 1000,
            "rounds": 100,
        },
    )
