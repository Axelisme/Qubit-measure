"""Experiment starting values from fixed captures and injected templates."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form

from .core import AmpRabiCfg

FORM = Form(
    AmpRabiCfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "qub_pulse": Slot("$qubit.module.x180", fallback=Template("pi_pulse")),
            "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
            "init_pulse": Hidden(),
        },
        "sweep": {
            "gain": Range.between(-0.3, "1.2 * $qubit.module.x180.gain", expts=51)
        },
        "relax_delay": Expr("5 * $qubit.t1"),
        "initial_delay": Hidden(),
        "reps": 1000,
        "rounds": 100,
    },
)
