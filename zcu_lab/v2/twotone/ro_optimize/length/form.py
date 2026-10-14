"""Core-derived starting values for ro_optimize/length."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form

from zcu_lab.v2.twotone.ro_optimize.length.core import LengthCfg

FORM = Form(
    LengthCfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "qub_pulse": Slot("$qubit.module.x180", fallback=Template("pi_pulse")),
            "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
        },
        "sweep": {
            "length": Range.between(0.01, 3.5, expts=51),
        },
        "relax_delay": Expr("5.0 * $qubit.t1"),
        "reps": 10000,
        "rounds": 1,
        "initial_delay": Hidden(),
    },
)
