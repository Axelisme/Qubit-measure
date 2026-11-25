"""Starting values for chain/t1; Qt-free data only."""

from zcu_tools.cfg_model import Range, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time
from zcu_lab.v2.support.measure.seed_metadata import pulse_readout_seed_metadata

from .core import ChainT1Cfg


def make_form(ctx: FormContext) -> Form[ChainT1Cfg]:
    """Use the twotone/time_domain/t1 π pulse, readout and delay range; K is 5 T1.

    Reps and rounds start at 100.
    """
    relax_delay = t1_initial_time(ctx.parameters, multiplier=5.0)
    return Form(
        ChainT1Cfg,
        tree={
            "reps": 100,
            "rounds": 100,
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
        },
    )
