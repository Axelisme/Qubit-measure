"""Starting values for the fixed-length flux reset gain scan."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time
from zcu_lab.v2.twotone.reset.flux.gain.core import GainCfg


def make_form(ctx: FormContext) -> Form[GainCfg]:
    """Start at zero gain; the operator supplies the allowed flux scan range."""
    return Form(
        GainCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "pi_pulse": source(
                ctx,
                "$qubit.module.x180",
                Template("pi_pulse", bindings={"qubit": "$qubit"}),
            ),
            "tested_reset": Template(
                "flux_reset",
                bindings={"qubit": "$qubit"},
                set={"pulse_cfg.gain": Hidden(0.0)},
            ),
            "readout": source(
                ctx,
                "$resonator.module.readout",
                Template("readout", bindings={"resonator": "$resonator"}),
            ),
            "gain": Range(0.0, 0.0, expts=1),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    )
