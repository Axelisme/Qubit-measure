"""Starting values for singleshot/t1/t1; sources are captured at Open/Reset."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form

from zcu_lab.v2.singleshot.t1.t1.core import T1Cfg

FORM = Form(
    T1Cfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "pi_pulse": Slot("$qubit.module.x180", fallback=Template("pi_pulse")),
            "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
        },
        "sweep": {
            "length": Range.between(0.0, "5.0 * $qubit.t1", expts=101),
        },
        "g_center": Expr("$qubit.ext.readout_cal.g_center"),
        "e_center": Expr("$qubit.ext.readout_cal.e_center"),
        "radius": Expr("$qubit.ext.readout_cal.radius"),
        "relax_delay": Expr("5.0 * $qubit.t1"),
        "uniform": False,
        "reps": 1000,
        "rounds": 10,
        "initial_delay": Hidden(),
    },
)
