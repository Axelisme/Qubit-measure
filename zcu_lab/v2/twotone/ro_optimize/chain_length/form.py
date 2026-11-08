"""Starting values for ro_optimize/chain_length; Qt-free data only."""

from dataclasses import replace

from zcu_tools.cfg_model import Hidden, Range, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source
from zcu_lab.v2.support.measure.seed_metadata import pulse_readout_seed_metadata
from zcu_lab.v2.twotone.ro_optimize.chain_length.core import ChainLengthCfg


def make_form(ctx: FormContext) -> Form[ChainLengthCfg]:
    """Use the captured readout slot or a fresh readout template; reps and rounds use cfg defaults."""
    readout = source(
        ctx,
        "$resonator.module.readout",
        Template(
            "readout",
            bindings={"resonator": "$resonator"},
            seed_metadata=pulse_readout_seed_metadata(gain=0.1),
        ),
    )
    # Run sets the ADC window from the sweep and the pulse length to that
    # window plus 0.11 us.
    readout = replace(
        readout,
        set={
            **readout.set,
            "ro_length": Hidden(0.0),
            "waveform.length": Hidden(0.0),
        },
    )
    return Form(
        ChainLengthCfg,
        tree={
            "readout": readout,
            "length": Range(0.2, 2.2, expts=11),
        },
    )
