"""Experiment starting values from fixed captures and injected templates."""

from typing import Annotated

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form import Display, FormModel
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2._support.measure.pulse_readout_metadata import (
    pulse_readout_seed_metadata,
)

from .core import T2RamseyCfg


class DetuneInput(FormModel):
    """GUI phase-ramp input; ratio counts fringe periods per delay step."""

    detune_ratio: Annotated[
        float, Display(label="Detune ratio (fringes/step)", decimals=3)
    ] = 0.05


def lower_detune(extras: DetuneInput, cfg: T2RamseyCfg) -> T2RamseyCfg:
    """Set detune (MHz) to fringes/step divided by the delay step (us).

    A zero ratio disables detune even on a zero-step sweep. A nonzero ratio
    on a zero-step sweep raises ValueError.
    """
    if extras.detune_ratio == 0.0:
        cfg.detune = 0.0
    elif cfg.sweep.length.step == 0.0:
        raise ValueError(
            "cannot apply a nonzero detune_ratio on a degenerate "
            "length sweep (expts < 2, step == 0)"
        )
    else:
        cfg.detune = extras.detune_ratio / cfg.sweep.length.step
    return cfg


def make_form(ctx: FormContext) -> Form[T2RamseyCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        T2RamseyCfg,
        field_order=(
            "modules",
            "sweep",
            "reps",
            "rounds",
            "relax_delay",
            "initial_delay",
        ),
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "pi2_pulse": Slot("$qubit.module.x90")
                if ctx.has("$qubit.module.x90")
                else Template("pi2_pulse", bindings={"qubit": "$qubit"}),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout", bindings={"resonator": "$resonator"}),
            },
            "sweep": {"length": Range(0.0, "1.5 * $qubit.t2r", expts=101)},
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "initial_delay": Hidden(),
            "reps": 1000,
            "rounds": 10,
            "detune": Hidden(),
        },
        extras=DetuneInput,
        lower=lower_detune,
    ).with_seed_metadata(
        (
            {}
            if ctx.has("$qubit.module.readout")
            else pulse_readout_seed_metadata(gain=0.1)
        )
    )
