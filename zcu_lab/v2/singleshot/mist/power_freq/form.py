"""Starting values for singleshot/mist/power_freq; sources are captured at Open/Reset."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form

from zcu_lab.v2.singleshot.mist.power_freq.core import FreqPowerCfg

FORM = Form(
    FreqPowerCfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "init_pulse": Slot("$qubit.module.x180"),
            "probe_pulse": Template("res_probe"),
            "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
        },
        "sweep": {
            "freq": Range.around("$resonator.freq", "1.5 * $resonator.kappa", expts=51),
            "gain": Range.between(0.0, 1.0, expts=51),
        },
        "g_center": Expr("$qubit.ext.readout_cal.g_center"),
        "e_center": Expr("$qubit.ext.readout_cal.e_center"),
        "radius": Expr("$qubit.ext.readout_cal.radius"),
        "relax_delay": Expr("5.0 * $qubit.t1"),
        "reps": 1000,
        "rounds": 100,
        "initial_delay": Hidden(),
    },
)
