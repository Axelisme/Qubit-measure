"""Experiment starting values from fixed captures and injected templates."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time
from zcu_lab.v2.support.measure.rabi_initial_values import len_rabi_length_stop
from zcu_lab.v2.support.measure.seed_metadata import (
    pulse_readout_seed_metadata,
    qub_probe_gain_seed_metadata,
)

from .core import LenRabiCfg


def make_form(ctx: FormContext) -> Form[LenRabiCfg]:
    """Build starting values and shape from the captured source availability."""
    length_stop = len_rabi_length_stop(ctx.parameters)
    qub_pulse = source(
        ctx,
        "$qubit.module.x180",
        Template(
            "qub_probe",
            bindings={"qubit": "$qubit"},
            set={"gain": 1.0},
            seed_metadata=qub_probe_gain_seed_metadata(
                ("qub_pulse", "module", "gain")
            ),
        ),
    )
    return Form(
        LenRabiCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "qub_pulse": qub_pulse,
            "readout": source(
                ctx,
                "$resonator.module.readout",
                Template(
                    "readout",
                    bindings={"resonator": "$resonator"},
                    seed_metadata=pulse_readout_seed_metadata(gain=0.1),
                ),
            ),
            "init_pulse": Hidden(),
            "length": Range(0.03, length_stop, expts=101),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    )
