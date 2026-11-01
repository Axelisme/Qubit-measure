"""Starting values captured from the selected qubit and resonator."""

from zcu_tools.cfg_model import Range, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time
from zcu_lab.v2.support.measure.seed_metadata import pulse_readout_seed_metadata

from .core import HeraldingT1Cfg


def make_form(ctx: FormContext) -> Form[HeraldingT1Cfg]:
    """Use the calibrated pi/readout slots; leave missing classifier uncalibrated."""
    relax_delay = t1_initial_time(ctx.parameters, multiplier=5.0)
    return Form(
        HeraldingT1Cfg,
        tree={
            "pi_pulse": source(
                ctx,
                "$qubit.module.x180",
                Template("pi_pulse", bindings={"qubit": "$qubit"}),
            ),
            "readout": source(
                ctx,
                "$resonator.module.readout",
                Template(
                    "readout",
                    bindings={"resonator": "$resonator"},
                    seed_metadata=pulse_readout_seed_metadata(gain=0.1),
                ),
            ),
            "delay": Range(0.0, relax_delay, expts=101),
            "relax_delay": relax_delay,
            "sampling": "linear",
            "reps": 1000,
            "rounds": 10,
        },
    )
