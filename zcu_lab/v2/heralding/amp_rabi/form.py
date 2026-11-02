"""Experiment starting values from fixed captures and injected templates."""

from dataclasses import replace

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time
from zcu_lab.v2.support.measure.rabi_initial_values import amp_rabi_initial_pulse
from zcu_lab.v2.support.measure.seed_metadata import (
    pulse_readout_seed_metadata,
)

from .core import HeraldingAmpRabiCfg


def make_form(ctx: FormContext) -> Form[HeraldingAmpRabiCfg]:
    """Build starting values and shape from the captured source availability."""
    pulse = amp_rabi_initial_pulse(ctx.parameters)
    # Keep shape/calibration admission; Run supplies gain itself.
    pulse = replace(pulse, set={**pulse.set, "gain": Hidden(0.0)})
    return Form(
        HeraldingAmpRabiCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "qub_pulse": pulse,
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
            "gain": Range(-0.3, "min(1.0, 1.2 * $qubit.module.x180.gain)", expts=51),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    )
