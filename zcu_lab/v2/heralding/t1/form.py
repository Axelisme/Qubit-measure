"""Starting values captured from the selected qubit and resonator."""

from zcu_tools.cfg_model import Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.support.measure.seed_metadata import pulse_readout_seed_metadata

from .core import HeraldingT1Cfg


def make_form(ctx: FormContext) -> Form[HeraldingT1Cfg]:
    """Use the calibrated pi/readout slots; leave missing classifier uncalibrated."""
    relax_delay = t1_initial_time(ctx.parameters, multiplier=5.0)
    return Form(
        HeraldingT1Cfg,
        field_order=(
            "pi_pulse",
            "readout",
            "length",
            "uniform",
            "reps",
            "rounds",
            "relax_delay",
        ),
        tree={
            "pi_pulse": Slot("$qubit.module.x180")
            if ctx.has("$qubit.module.x180")
            else Template("pi_pulse", bindings={"qubit": "$qubit"}),
            "readout": Slot("$resonator.module.readout")
            if ctx.has("$resonator.module.readout")
            else Template("readout", bindings={"resonator": "$resonator"}),
            "length": Range(0.0, relax_delay, expts=101),
            "relax_delay": relax_delay,
            "uniform": True,
            "reps": 1000,
            "rounds": 10,
        },
    ).with_seed_metadata(
        {}
        if ctx.has("$resonator.module.readout")
        else pulse_readout_seed_metadata(gain=0.1)
    )
