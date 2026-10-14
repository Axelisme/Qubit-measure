"""Starting values for jpa/check; Qt-free declarations."""

from zcu_tools.cfg_model import AliasName, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form

from zcu_lab.v2.jpa.check.core import CheckCfg

FORM = Form(
    CheckCfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            # The Run frequency sweep overwrites both readout frequencies.
            "readout": Template("readout"),
        },
        "jpa_rf_dev": AliasName("jpa_rf_dev"),
        "sweep": {
            "freq": Range.around(
                "$resonator.freq", "1.5 * $resonator.kappa", expts=101
            ),
        },
        "relax_delay": 0.5,
        "reps": 1000,
        "rounds": 5,
        "initial_delay": Hidden(),
    },
)
