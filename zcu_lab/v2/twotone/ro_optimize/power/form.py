"""Core-derived starting values for ro_optimize/power."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form

from zcu_lab.v2.twotone.ro_optimize.power.core import PowerCfg

FORM = Form(
    PowerCfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "qub_pulse": Slot("$qubit.module.x180", fallback=Template("pi_pulse")),
            "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
        },
        "sweep": {
            "gain": Range.between(0.001, 0.2, expts=101),
        },
        "relax_delay": Expr("5.0 * $qubit.t1"),
        "reps": 1000,
        "rounds": 100,
        "initial_delay": Hidden(),
    },
)
