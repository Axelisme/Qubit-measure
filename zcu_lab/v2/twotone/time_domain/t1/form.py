"""Experiment starting values from fixed captures and injected templates."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time

from .core import T1Cfg


def make_form(ctx: FormContext) -> Form[T1Cfg]:
    """Build starting values and shape from the captured source availability."""
    relax_delay = t1_initial_time(ctx.parameters, multiplier=5.0)
    return Form(
        T1Cfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "pi_pulse": Slot("$qubit.module.x180")
                if ctx.has("$qubit.module.x180")
                else Template("pi_pulse"),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout"),
            },
            "sweep": {"length": Range.between(0.0, relax_delay, expts=101)},
            "relax_delay": relax_delay,
            "initial_delay": Hidden(),
            "uniform": True,
            "reps": 1000,
            "rounds": 100,
        },
    )
