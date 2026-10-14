"""Starting values for singleshot/check; sources are captured at Open/Reset."""

from zcu_tools.cfg_model import Expr, Hidden, Locked, Slot, Template
from zcu_tools.gui.form_adapter import Form

from zcu_lab.v2.singleshot.check.core import CheckCfg

FORM = Form(
    CheckCfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "init_pulse": Slot("$qubit.module.x180"),
            "probe_pulse": Template("qub_probe"),
            "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
        },
        "relax_delay": Expr("5.0 * $qubit.t1"),
        "shots": 5000,
        "reps": Locked(1),
        "rounds": Locked(1),
        "initial_delay": Hidden(),
    },
)
