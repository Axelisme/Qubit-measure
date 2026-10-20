"""Starting values for jpa/freq; Qt-free declarations."""

from zcu_tools.cfg_model import AliasName, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.jpa.freq.core import FreqCfg


def make_form(ctx: FormContext) -> Form[FreqCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        FreqCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "pi_pulse": Slot("$qubit.module.x180")
                if ctx.has("$qubit.module.x180")
                else Template("pi_pulse", bindings={"qubit": "$qubit"}),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout", bindings={"resonator": "$resonator"}),
            },
            "jpa_rf_dev": AliasName("jpa_rf_dev"),
            "sweep": {
                "jpa_freq": Range.between(
                    "0.98 * $jpa.pump_freq", "1.02 * $jpa.pump_freq", expts=101
                )
            },
            "skew_penalty": 0.0,
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=1),
            "reps": 10000,
            "rounds": 1,
            "initial_delay": Hidden(),
        },
    )
