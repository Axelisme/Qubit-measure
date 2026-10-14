"""Core-derived starting values for ro_optimize/freq."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form

from zcu_lab.v2.twotone.ro_optimize.freq.core import FreqCfg

FORM = Form(
    FreqCfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "qub_pulse": Slot("$qubit.module.x180", fallback=Template("pi_pulse")),
            "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
        },
        "sweep": {
            "freq": Range.around("$resonator.freq", "$resonator.kappa", expts=301),
        },
        "relax_delay": Expr("5.0 * $qubit.t1"),
        "reps": 1000,
        "rounds": 100,
        "initial_delay": Hidden(),
    },
)
