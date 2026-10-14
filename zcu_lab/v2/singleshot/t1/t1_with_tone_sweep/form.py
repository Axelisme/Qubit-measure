"""Each form exposes length and exactly one outer sweep; the other stays None."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form

from zcu_lab.v2.singleshot.t1.t1_with_tone_sweep.core import T1WithToneSweepCfg

FORM_GAIN = Form(
    T1WithToneSweepCfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "pi_pulse": Slot("$qubit.module.x180", fallback=Template("pi_pulse")),
            "probe_pulse": Template("res_probe"),
            "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
        },
        "sweep": {
            "length": Range.between(0.0, "5.0 * $qubit.t1", expts=51),
            "gain": Range.between(0.0, 1.0, expts=21),
            "freq": Hidden(),
        },
        "g_center": Expr("$qubit.ext.readout_cal.g_center"),
        "e_center": Expr("$qubit.ext.readout_cal.e_center"),
        "radius": Expr("$qubit.ext.readout_cal.radius"),
        "relax_delay": Expr("5.0 * $qubit.t1"),
        "uniform": True,
        "reps": 1000,
        "rounds": 10,
        "initial_delay": Hidden(),
    },
)

FORM_FREQ = Form(
    T1WithToneSweepCfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "pi_pulse": Slot("$qubit.module.x180", fallback=Template("pi_pulse")),
            "probe_pulse": Template("res_probe"),
            "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
        },
        "sweep": {
            "length": Range.between(0.0, "5.0 * $qubit.t1", expts=51),
            "freq": Range.around("$resonator.freq", "1.5 * $resonator.kappa", expts=21),
            "gain": Hidden(),
        },
        "g_center": Expr("$qubit.ext.readout_cal.g_center"),
        "e_center": Expr("$qubit.ext.readout_cal.e_center"),
        "radius": Expr("$qubit.ext.readout_cal.radius"),
        "relax_delay": Expr("5.0 * $qubit.t1"),
        "uniform": True,
        "reps": 1000,
        "rounds": 10,
        "initial_delay": Hidden(),
    },
)
