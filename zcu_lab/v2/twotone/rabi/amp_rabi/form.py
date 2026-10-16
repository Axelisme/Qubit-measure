"""Experiment starting values from fixed captures and injected templates."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time

from .core import AmpRabiCfg


def make_form(ctx: FormContext) -> Form[AmpRabiCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        AmpRabiCfg,
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
            "sweep": {"gain": Range.between(-0.3, 1.0, expts=51)},
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "initial_delay": Hidden(),
            "reps": 1000,
            "rounds": 100,
        },
    )
