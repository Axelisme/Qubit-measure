"""Starting values for chain/amp_rabi; Qt-free data only."""

from dataclasses import replace

from zcu_tools.cfg_model import Hidden, Range, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time
from zcu_lab.v2.support.measure.rabi_initial_values import amp_rabi_initial_pulse
from zcu_lab.v2.support.measure.seed_metadata import pulse_readout_seed_metadata

from .core import ChainAmpRabiCfg


def make_form(ctx: FormContext) -> Form[ChainAmpRabiCfg]:
    """Use the twotone/rabi/amp_rabi pulse, readout and gain range; K is 5 T1.

    Reps and rounds use the cfg defaults.
    """
    pulse = amp_rabi_initial_pulse(ctx.parameters)
    # Keep shape/calibration admission; Run supplies gain itself.
    pulse = replace(pulse, set={**pulse.set, "gain": Hidden(0.0)})
    return Form(
        ChainAmpRabiCfg,
        tree={
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
            "gain": Range(-0.3, "min(1.0, 1.2 * $qubit.module.x180.gain)", expts=51),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
        },
    )
