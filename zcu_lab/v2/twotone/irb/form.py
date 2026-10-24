"""Experiment starting values from fixed captures and injected templates."""

from zcu_tools.cfg_model import Hidden, LogPoints, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import t1_initial_time

from .core import IRBCfg


def make_form(ctx: FormContext) -> Form[IRBCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        IRBCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "X90_pulse": Slot("$qubit.module.x90")
            if ctx.has("$qubit.module.x90")
            else Template("pi2_pulse", bindings={"qubit": "$qubit"}),
            "X180_pulse": Slot("$qubit.module.x180")
            if ctx.has("$qubit.module.x180")
            else Template("pi_pulse", bindings={"qubit": "$qubit"}),
            "readout": Slot("$qubit.module.readout")
            if ctx.has("$qubit.module.readout")
            else Template("readout", bindings={"resonator": "$resonator"}),
            "I_pulse": Hidden(),
            "depths": LogPoints(max=128, points=9),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "seed": 42,
            "n_seeds": 20,
            "target_gate": "X90",
            "reps": 1000,
            "rounds": 10,
        },
    )
