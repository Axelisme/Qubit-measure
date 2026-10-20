"""Core-derived starting values for ro_optimize/freq_gain."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.twotone.ro_optimize.freq_gain.core import FreqGainCfg


def make_form(ctx: FormContext) -> Form[FreqGainCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        FreqGainCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "qub_pulse": Slot("$qubit.module.x180")
                if ctx.has("$qubit.module.x180")
                else Template("pi_pulse", bindings={"qubit": "$qubit"}),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout", bindings={"resonator": "$resonator"}),
            },
            "sweep": {
                "freq": Range.around(
                    "$resonator.freq", "0.5 * $resonator.kappa", expts=31
                ),
                "gain": Range.between(0.0, 0.2, expts=31),
            },
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 100,
            "rounds": 1000,
            "initial_delay": Hidden(),
        },
    )
