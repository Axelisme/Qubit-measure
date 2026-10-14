"""Starting values for jpa/power; Qt-free declarations."""

from zcu_tools.cfg_model import AliasName, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form

from zcu_lab.v2.jpa.power.core import PowerCfg

FORM = Form(
    PowerCfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "pi_pulse": Slot("$qubit.module.x180", fallback=Template("pi_pulse")),
            "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
        },
        "jpa_rf_dev": AliasName("jpa_rf_dev"),
        "sweep": {
            "jpa_power": Range.around("$jpa.pump_power", 5.0, expts=101),
        },
        "skew_penalty": 0.0,
        "relax_delay": 0.5,
        "reps": 10000,
        "rounds": 1,
        "initial_delay": Hidden(),
    },
)
