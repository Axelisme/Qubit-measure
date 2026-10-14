"""Starting values for jpa/freq; Qt-free declarations."""

from zcu_tools.cfg_model import AliasName, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form

from zcu_lab.v2.jpa.freq.core import FreqCfg

FORM = Form(
    FreqCfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "pi_pulse": Slot("$qubit.module.x180", fallback=Template("pi_pulse")),
            "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
        },
        "jpa_rf_dev": AliasName("jpa_rf_dev"),
        "sweep": {
            "jpa_freq": Range.between(
                "0.98 * $jpa.pump_freq", "1.02 * $jpa.pump_freq", expts=101
            ),
        },
        "skew_penalty": 0.0,
        "relax_delay": 0.5,
        "reps": 10000,
        "rounds": 1,
        "initial_delay": Hidden(),
    },
)
