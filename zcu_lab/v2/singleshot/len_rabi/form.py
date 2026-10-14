"""Starting values for singleshot/len_rabi; sources are captured at Open/Reset."""

from zcu_tools.cfg_model import Expr, Hidden, Locked, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form

from zcu_lab.v2.singleshot.len_rabi.core import LenRabiCfg

FORM = Form(
    LenRabiCfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "qub_pulse": Template("qub_probe", set={"gain": 1.0}),
            "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
        },
        "sweep": {
            "length": Range.between(
                0.03, "4.0 * $qubit.module.x180.waveform.length", expts=51
            ),
        },
        "g_center": Expr("$qubit.ext.readout_cal.g_center"),
        "e_center": Expr("$qubit.ext.readout_cal.e_center"),
        "radius": Expr("$qubit.ext.readout_cal.radius"),
        "relax_delay": 50.5,
        "shots": 1000,
        "reps": Locked(1),
        "rounds": Locked(1),
        "initial_delay": Hidden(),
    },
)
