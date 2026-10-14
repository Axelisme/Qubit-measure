"""Starting values for jpa/auto_optimize; Qt-free declarations."""

from zcu_tools.cfg_model import AliasName, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form

from zcu_lab.v2.jpa.auto_optimize.core import JPAOptCfg

FORM = Form(
    JPAOptCfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "pi_pulse": Slot("$qubit.module.x180", fallback=Template("pi_pulse")),
            "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
        },
        "jpa_rf_dev": AliasName("jpa_rf_dev"),
        "jpa_flux_dev": AliasName("flux_dev"),
        "flux_unit": "A",
        "sweep": {
            "jpa_flux": Range.around("$jpa.flux", 5.0e-3, expts=101),
            "jpa_freq": Range.between(
                "0.98 * $jpa.pump_freq", "1.02 * $jpa.pump_freq", expts=101
            ),
            "jpa_power": Range.around("$jpa.pump_power", 5.0, expts=101),
        },
        "num_points": 1001,
        "skew_penalty": 0.0,
        "relax_delay": 30.5,
        "reps": 1000,
        "rounds": 1,
        "initial_delay": Hidden(),
    },
)
