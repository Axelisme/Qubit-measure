"""Experiment starting values from fixed captures and injected templates."""

from zcu_tools.cfg_model import Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.support.measure.seed_metadata import (
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
            "acquire_retry",
            "earlystop_snr",
            "relax_delay",
        ),
        tree={
            "reset": Slot("$qubit.module.reset"),
            "pi_pulse": Slot(
                "$qubit.module.x180",
                fallback=Template("pi_pulse", bindings={"qubit": "$qubit"}),
            ),
            "readout": Slot(
                "$resonator.module.readout",
                fallback=Template(
                    "readout",
                    bindings={"resonator": "$resonator"},
                    seed_metadata=pulse_readout_seed_metadata(gain=0.1),
                ),
            ),
            "length": Range(0.0, relax_delay, expts=101),
            "relax_delay": relax_delay,
            "uniform": True,
            "reps": 1000,
            "rounds": 10,
        },
    )
