"""Core-derived starting values for ro_optimize/freq_gain."""

from dataclasses import replace

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time
from zcu_lab.v2.twotone.ro_optimize.freq_gain.core import FreqGainCfg


def make_form(ctx: FormContext) -> Form[FreqGainCfg]:
    """Build starting values and shape from the captured source availability."""
    readout = source(
        ctx,
        "$resonator.module.readout",
        Template(
            "readout",
            bindings={"resonator": "$resonator"},
            set={"freq": Hidden(0.0), "gain": Hidden(0.0)},
        ),
    )
    readout = replace(readout, set={**readout.set, "freq": Hidden(0.0), "gain": Hidden(0.0)})
    return Form(
        FreqGainCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "qub_pulse": source(
                ctx,
                "$qubit.module.x180",
                Template("pi_pulse", bindings={"qubit": "$qubit"}),
            ),
            "readout": readout,
            "freq": Range.around("$resonator.freq", "0.5 * $resonator.kappa", expts=31),
            "gain": Range(0.0, 0.2, expts=31),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    )
