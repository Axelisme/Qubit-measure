"""Starting values for chain/t2ramsey; Qt-free data only."""

from zcu_tools.cfg_model import Expr, Range, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time
from zcu_lab.v2.support.measure.seed_metadata import pulse_readout_seed_metadata

from .core import ChainT2RamseyCfg


def make_form(ctx: FormContext) -> Form[ChainT2RamseyCfg]:
    """Use the twotone/time_domain/t2ramsey π/2 pulse, readout and delay range; K is 5 T1.

    Detune starts at five fringes over the delay range. Reps and rounds
    start at 100.
    """
    return Form(
        ChainT2RamseyCfg,
        tree={
            "reps": 100,
            "rounds": 100,
            "pi2_pulse": source(
                ctx,
                "$qubit.module.x90",
                Template("pi2_pulse", bindings={"qubit": "$qubit"}),
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
            "delay": Range(0.0, "1.5 * $qubit.t2r", expts=101),
            "detune": Expr("5 / (1.5 * $qubit.t2r)"),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
        },
    )
