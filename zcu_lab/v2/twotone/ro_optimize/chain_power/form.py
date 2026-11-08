"""Starting values for ro_optimize/chain_power; Qt-free data only."""

from dataclasses import replace

from zcu_tools.cfg_model import Hidden, Range, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source
from zcu_lab.v2.twotone.ro_optimize.chain_power.core import ChainPowerCfg


def make_form(ctx: FormContext) -> Form[ChainPowerCfg]:
    """Use the captured readout slot or a fresh readout template; reps and rounds use cfg defaults."""
    readout = source(
        ctx,
        "$resonator.module.readout",
        Template("readout", bindings={"resonator": "$resonator"}),
    )
    # Run replaces the readout gain with the sweep.
    readout = replace(readout, set={**readout.set, "gain": Hidden(0.0)})
    return Form(
        ChainPowerCfg,
        tree={
            "readout": readout,
            "gain": Range(0.02, 0.2, expts=10),
        },
    )
