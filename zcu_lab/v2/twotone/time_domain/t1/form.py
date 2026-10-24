"""Experiment starting values from fixed captures and injected templates."""

from zcu_tools.cfg_model import Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.support.measure.pulse_readout_metadata import (
    pulse_readout_seed_metadata,
)

from .core import T1Cfg


def make_form(ctx: FormContext) -> Form[T1Cfg]:
    """Build starting values and shape from the captured source availability."""
    relax_delay = t1_initial_time(ctx.parameters, multiplier=5.0)
    return Form(
        T1Cfg,
        field_order=(
            "reset",
            "pi_pulse",
            "readout",
            "length",
            "uniform",
            "reps",
            "rounds",
            "relax_delay",
        ),
        tree={
            "reset": Slot("$qubit.module.reset"),
            "pi_pulse": Slot("$qubit.module.x180")
            if ctx.has("$qubit.module.x180")
            else Template("pi_pulse", bindings={"qubit": "$qubit"}),
            "readout": Slot("$qubit.module.readout")
            if ctx.has("$qubit.module.readout")
            else Template("readout", bindings={"resonator": "$resonator"}),
            "length": Range.between(0.0, relax_delay, expts=101),
            "relax_delay": relax_delay,
            "uniform": True,
            "reps": 1000,
            "rounds": 10,
        },
    ).with_seed_metadata(
        (
            {}
            if ctx.has("$qubit.module.readout")
            else pulse_readout_seed_metadata(gain=0.1)
        )
    )
