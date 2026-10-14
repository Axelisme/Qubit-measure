"""Starting values for singleshot/amp_rabi; sources are captured at Open/Reset."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form

from zcu_lab.v2.singleshot.amp_rabi.core import AmpRabiCfg

FORM = Form(
    AmpRabiCfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "qub_pulse": Template("qub_probe", set={"gain": 1.0}),
            "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
        },
        "sweep": {
            "gain": Range.between(-0.3, "4.0 * $qubit.module.x180.gain", expts=51),
        },
        "g_center": Expr("$qubit.ext.readout_cal.g_center"),
        "e_center": Expr("$qubit.ext.readout_cal.e_center"),
        "radius": Expr("$qubit.ext.readout_cal.radius"),
        "relax_delay": 50.5,
        "reps": 1000,
        "rounds": 1,
        "initial_delay": Hidden(),
    },
)
