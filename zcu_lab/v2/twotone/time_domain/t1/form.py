"""Experiment starting values from fixed captures and injected templates."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form

from .core import T1Cfg

FORM = Form(
    T1Cfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "pi_pulse": Slot("$qubit.module.x180", fallback=Template("pi_pulse")),
            "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
        },
        "sweep": {"length": Range.between(0.0, "5 * $qubit.t1", expts=101)},
        "relax_delay": Expr("5 * $qubit.t1"),
        "initial_delay": Hidden(),
        "uniform": True,
        "reps": 1000,
        "rounds": 100,
    },
)
