"""Core-derived starting values for reset/rabi_check."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form

from zcu_lab.v2.twotone.reset.rabi_check.core import RabiCheckCfg

FORM = Form(
    RabiCheckCfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "rabi_pulse": Slot("$qubit.module.x180", fallback=Template("pi_pulse")),
            "tested_reset": Slot("$qubit.module.reset", fallback=Template("reset")),
            "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
        },
        "sweep": {
            "gain": Range.between(0.0, 1.0, expts=51),
        },
        "relax_delay": 1.0,
        "reps": 1000,
        "rounds": 10,
        "initial_delay": Hidden(),
    },
)
