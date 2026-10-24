"""Experiment starting values from fixed captures and injected templates."""

from typing import Annotated

from zcu_tools.cfg_model import ConfigBase, FieldMeta, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.program.v2 import Pulse, Readout, Reset, SweepCfg

from zcu_lab.v2.support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.support.measure.pulse_readout_metadata import (
    pulse_readout_seed_metadata,
)


class T2RamseyInput(ConfigBase):
    """Captured pulses, delay sweep and phase advance in fringes per step."""

    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    pi2_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi/2 Pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    length: Annotated[SweepCfg, FieldMeta(label="Delay", unit="us")]
    detune_ratio: Annotated[
        float, FieldMeta(label="Detune ratio (fringes/step)", decimals=3)
    ] = 0.05


def make_form(ctx: FormContext) -> Form[T2RamseyInput]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        T2RamseyInput,
        field_order=(
            "reset",
            "pi2_pulse",
            "readout",
            "length",
            "reps",
            "rounds",
            "relax_delay",
        ),
        tree={
            "reset": Slot("$qubit.module.reset"),
            "pi2_pulse": Slot("$qubit.module.x90")
            if ctx.has("$qubit.module.x90")
            else Template("pi2_pulse", bindings={"qubit": "$qubit"}),
            "readout": Slot("$qubit.module.readout")
            if ctx.has("$qubit.module.readout")
            else Template("readout", bindings={"resonator": "$resonator"}),
            "length": Range.between(0.0, "1.5 * $qubit.t2r", expts=101),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    ).with_seed_metadata(
        (
            {}
            if ctx.has("$qubit.module.readout")
            else pulse_readout_seed_metadata(gain=0.1)
        )
    )
