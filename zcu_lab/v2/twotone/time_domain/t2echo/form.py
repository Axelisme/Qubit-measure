"""Experiment starting values from fixed captures and injected templates."""

from typing import Annotated

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form import Display, FormModel
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2._support.measure.pulse_readout_metadata import (
    pulse_readout_seed_metadata,
)

from .core import T2EchoCfg


class DetuneInput(FormModel):
    """GUI phase-ramp input; ratio counts fringe periods per delay step."""

    detune_ratio: Annotated[
        float, Display(label="Detune ratio (fringes/step)", decimals=3)
    ] = 0.1


def lower_detune(extras: DetuneInput, cfg: T2EchoCfg) -> T2EchoCfg:
    """Set detune (MHz) to fringes/step divided by the delay step (us).

    A zero ratio disables detune even on a zero-step sweep. A nonzero ratio
    on a zero-step sweep raises ValueError.
    """
    if extras.detune_ratio == 0.0:
        cfg.detune = 0.0
    elif cfg.length.step == 0.0:
        raise ValueError(
            "cannot apply a nonzero detune_ratio on a degenerate "
            "length sweep (expts < 2, step == 0)"
        )
    else:
        cfg.detune = extras.detune_ratio / cfg.length.step
    return cfg


def make_form(ctx: FormContext) -> Form[T2EchoCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        T2EchoCfg,
        field_order=(
            "reset",
            "pi2_pulse",
            "pi_pulse",
            "readout",
            "length",
            "reps",
            "rounds",
            "relax_delay",
        ),
        tree={
            "reset": Slot("$qubit.module.reset"),
            "pi_pulse": Slot("$qubit.module.x180")
            if ctx.has("$qubit.module.x180")
            else Template("pi_pulse", bindings={"qubit": "$qubit"}),
            "pi2_pulse": Slot("$qubit.module.x90")
            if ctx.has("$qubit.module.x90")
            else Template("pi2_pulse", bindings={"qubit": "$qubit"}),
            "readout": Slot("$qubit.module.readout")
            if ctx.has("$qubit.module.readout")
            else Template("readout", bindings={"resonator": "$resonator"}),
            "length": Range.between(0.0, "1.5 * $qubit.t2e", expts=101),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
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
