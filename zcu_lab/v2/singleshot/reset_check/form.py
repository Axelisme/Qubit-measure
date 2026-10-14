"""Starting values for singleshot/reset_check; sources are captured at Open/Reset."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form

from zcu_lab.v2.singleshot.reset_check.core import ResetCheckCfg

FORM = Form(
    ResetCheckCfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "rabi_pulse": Slot("$qubit.module.x180", fallback=Template("pi_pulse")),
            "tested_reset": Slot("$qubit.module.reset", fallback=Template("reset")),
            "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
        },
        "sweep": {
            "gain": Range.between(0.0, "4.0 * $qubit.module.x180.gain", expts=51),
        },
        "g_center": Expr("$qubit.ext.readout_cal.g_center"),
        "e_center": Expr("$qubit.ext.readout_cal.e_center"),
        "radius": Expr("$qubit.ext.readout_cal.radius"),
        "relax_delay": Expr("5.0 * $qubit.t1"),
        "reps": 5000,
        "rounds": 1,
        "initial_delay": Hidden(),
    },
)
