"""Experiment starting values from fixed captures and injected templates."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form

from .core import ZigZagScanCfg

GAIN_FORM = Form(
    ZigZagScanCfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "X90_pulse": Slot("$qubit.module.x90", fallback=Template("pi2_pulse")),
            "X180_pulse": Slot("$qubit.module.x180", fallback=Template("pi_pulse")),
            "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
        },
        "relax_delay": Expr("5 * $qubit.t1"),
        "initial_delay": Hidden(),
        "n_times": 6,
        "repeat_on": "X180_pulse",
        "reps": 100,
        "rounds": 100,
        "sweep": {
            "gain": Range.between(
                "0.8 * $qubit.module.x180.gain",
                "1.2 * $qubit.module.x180.gain",
                expts=101,
            ),
            "freq": Hidden(),
        },
    },
)

FREQ_FORM = Form(
    ZigZagScanCfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "X90_pulse": Slot("$qubit.module.x90", fallback=Template("pi2_pulse")),
            "X180_pulse": Slot("$qubit.module.x180", fallback=Template("pi_pulse")),
            "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
        },
        "relax_delay": Expr("5 * $qubit.t1"),
        "initial_delay": Hidden(),
        "n_times": 6,
        "repeat_on": "X180_pulse",
        "reps": 100,
        "rounds": 100,
        "sweep": {
            "freq": Range.around("$qubit.freq", "2.0", expts=101),
            "gain": Hidden(),
        },
    },
)
