"""Core-derived starting values for ro_optimize/auto_optimize."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form

from zcu_lab.v2.twotone.ro_optimize.auto_optimize.core import AutoOptCfg

FORM = Form(
    AutoOptCfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "qub_pulse": Slot("$qubit.module.x180", fallback=Template("pi_pulse")),
            "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
        },
        "sweep": {
            "freq": Range.around("$resonator.freq", "0.2 * $resonator.kappa", expts=51),
            "gain": Range.between(0.1, 0.25, expts=51),
            "length": Range.between(5.0, 10.0, expts=51),
        },
        "relax_delay": Expr("3.0 * $qubit.t1"),
        "reps": 1000,
        "rounds": 10,
        "num_points": 1001,
        "initial_delay": Hidden(),
    },
)
