"""Experiment starting values from fixed captures and injected templates."""

from typing import Annotated

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form import Display, FormModel
from zcu_tools.gui.form_adapter import Form

from zcu_lab.v2._support.measure.detune import resolve_detune

from .core import T2RamseyCfg


class DetuneInput(FormModel):
    """GUI phase-ramp input; ratio counts fringe periods per delay step."""

    detune_ratio: Annotated[
        float, Display(label="Detune ratio (fringes/step)", decimals=3)
    ] = 0.05


def lower_detune(extras: DetuneInput, cfg: T2RamseyCfg) -> T2RamseyCfg:
    """Return native MHz detune; reject nonzero ratio on a zero-step sweep."""
    cfg.detune = resolve_detune(extras.detune_ratio, cfg.sweep.length.step)
    return cfg


FORM = Form(
    T2RamseyCfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "pi2_pulse": Slot("$qubit.module.x90", fallback=Template("pi2_pulse")),
            "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
        },
        "sweep": {"length": Range.between(0.0, "1.5 * $qubit.t2r", expts=101)},
        "relax_delay": Expr("5 * $qubit.t1"),
        "initial_delay": Hidden(),
        "reps": 1000,
        "rounds": 100,
        "detune": Hidden(),
    },
    extras=DetuneInput,
    lower=lower_detune,
)
