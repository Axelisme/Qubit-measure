"""Core-derived starting values for ro_optimize/freq_gain."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form

from zcu_lab.v2.twotone.ro_optimize.freq_gain.core import FreqGainCfg

FORM = Form(
    FreqGainCfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "qub_pulse": Slot("$qubit.module.x180", fallback=Template("pi_pulse")),
            "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
        },
        "sweep": {
            "freq": Range.around("$resonator.freq", "0.5 * $resonator.kappa", expts=31),
            "gain": Range.between(0.0, 0.2, expts=31),
        },
        "relax_delay": Expr("5.0 * $qubit.t1"),
        "reps": 100,
        "rounds": 1000,
        "initial_delay": Hidden(),
    },
)
