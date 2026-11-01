"""Core-derived starting values for the two-branch resonator spectrum."""

from dataclasses import replace

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time
from zcu_lab.v2.twotone.dispersive.core import DispersiveCfg


def make_form(ctx: FormContext) -> Form[DispersiveCfg]:
    """Use captured x180/readout slots, or editable templates when absent."""
    readout = source(
        ctx,
        "$resonator.module.readout",
        Template(
            "readout",
            bindings={"resonator": "$resonator"},
            set={"freq": Hidden(0.0)},
        ),
    )
    readout = replace(readout, set={**readout.set, "freq": Hidden(0.0)})
    return Form(
        DispersiveCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "init_pulse": Hidden(),
            "qub_pulse": source(
                ctx,
                "$qubit.module.x180",
                Template("pi_pulse", bindings={"qubit": "$qubit"}),
            ),
            "readout": readout,
            "freq": Range.around(
                "$resonator.freq", "1.5 * $resonator.kappa", expts=301
            ),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    )
