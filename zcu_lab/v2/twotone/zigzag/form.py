"""Experiment starting values from fixed captures and injected templates."""

from zcu_tools.cfg_model import Expr, Hidden, Slot, Template
from zcu_tools.gui.form_adapter import Form

from .core import ZigZagCfg

FORM = Form(
    ZigZagCfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "X90_pulse": Slot("$qubit.module.x90", fallback=Template("pi2_pulse")),
            "X180_pulse": Slot("$qubit.module.x180", fallback=Template("pi_pulse")),
            "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
        },
        "relax_delay": Expr("5 * $qubit.t1"),
        "initial_delay": Hidden(),
        "n_times": 10,
        "repeat_on": "X180_pulse",
        "reset_phase_cycle": False,
        "reps": 1000,
        "rounds": 100,
    },
)
