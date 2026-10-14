"""Starting values for singleshot/t1/t1_with_tone; sources are captured at Open/Reset."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form

from zcu_lab.v2.singleshot.t1.t1_with_tone.core import T1WithToneCfg

FORM = Form(
    T1WithToneCfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "init_pulse": Slot("$qubit.module.x180"),
            "pi_pulse": Slot("$qubit.module.x180", fallback=Template("pi_pulse")),
            "probe_pulse": Template("res_probe"),
            "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
        },
        "sweep": {
            "length": Range.between(0.0, "5.0 * $qubit.ext.t1_with_tone", expts=101),
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
