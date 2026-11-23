"""Starting values for ro_optimize/chain_freq; Qt-free data only."""

from dataclasses import replace

from zcu_tools.cfg_model import Hidden, Range, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time
from zcu_lab.v2.support.measure.seed_metadata import pulse_readout_seed_metadata
from zcu_lab.v2.twotone.ro_optimize.chain_freq.core import ChainFreqCfg


def make_form(ctx: FormContext) -> Form[ChainFreqCfg]:
    """Capture readout sources and seed thermal relaxation K from 5*T1."""
    readout = source(
        ctx,
        "$resonator.module.readout",
        Template(
            "readout",
            bindings={"resonator": "$resonator"},
            seed_metadata=pulse_readout_seed_metadata(gain=0.1),
        ),
    )
    # Run replaces the readout frequency with the sweep.
    readout = replace(readout, set={**readout.set, "freq": Hidden(0.0)})
    return Form(
        ChainFreqCfg,
        tree={
            "reps": 2000,
            "rounds": 320,
            "readout": readout,
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "freq": Range.around("$resonator.freq", "$resonator.kappa", expts=31),
        },
    )
