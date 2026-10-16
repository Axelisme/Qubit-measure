"""Starting values for jpa/power; Qt-free declarations."""

from zcu_tools.cfg_model import AliasName, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.jpa.power.core import PowerCfg


def make_form(ctx: FormContext) -> Form[PowerCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        PowerCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "pi_pulse": Slot("$qubit.module.x180")
                if ctx.has("$qubit.module.x180")
                else Template("pi_pulse"),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout"),
            },
            "jpa_rf_dev": AliasName("jpa_rf_dev"),
            "sweep": {"jpa_power": Range.around("$jpa.pump_power", 5.0, expts=101)},
            "skew_penalty": 0.0,
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=1),
            "reps": 10000,
            "rounds": 1,
            "initial_delay": Hidden(),
        },
    )
