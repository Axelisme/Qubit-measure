"""Experiment starting values from fixed captures and injected templates."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from .core import RBCfg


def make_form(ctx: FormContext) -> Form[RBCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        RBCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "X90_pulse": Slot("$qubit.module.x90")
                if ctx.has("$qubit.module.x90")
                else Template("pi2_pulse"),
                "X180_pulse": Slot("$qubit.module.x180")
                if ctx.has("$qubit.module.x180")
                else Template("pi_pulse"),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout"),
                "I_pulse": Hidden(),
            },
            "sweep": {"depth": Range.between(0, 100, expts=21)},
            "relax_delay": Expr("5 * $qubit.t1"),
            "initial_delay": Hidden(),
            "seed": 42,
            "n_seeds": 20,
            "reps": 1000,
            "rounds": 1,
        },
    )
