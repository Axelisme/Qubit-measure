"""Shared time-window inputs and captured pulse selections for CPMG."""

from typing import Annotated

from pydantic import Field, model_validator
from zcu_tools.cfg_model import ConfigBase, FieldMeta, LogPoints, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.program.v2 import Pulse, Readout, Reset

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2._support.measure.pulse_readout_metadata import (
    pulse_readout_seed_metadata,
)


class CPMGInput(ConfigBase):
    """Captured pulses, pi-pulse counts and a shared total-time window in us."""

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
    reset: Reset.Cfg | None = None
    pi2_pulse: Pulse.Cfg
    pi_pulse: Pulse.Cfg
    readout: Readout.Cfg
    times: Annotated[
        list[Annotated[int, Field(strict=True, ge=1)]],
        Field(min_length=1),
        FieldMeta(label="Pi pulse counts"),
    ]
    detune_ratio: Annotated[
        float, FieldMeta(label="Detune ratio (fringes/step)", decimals=3)
    ] = 0.0
    earlystop_snr: Annotated[float | None, FieldMeta(label="Early-stop SNR")] = None
    min_length_us: Annotated[
        float,
        Field(gt=0, allow_inf_nan=False),
        FieldMeta(label="Minimum time", unit="us"),
    ] = 1.0
    max_length_us: Annotated[
        float,
        Field(gt=0, allow_inf_nan=False),
        FieldMeta(label="Maximum time", unit="us"),
    ] = 100.0
    length_points: Annotated[int, Field(ge=2), FieldMeta(label="Time points")] = 101

    @model_validator(mode="after")
    def validate_window(self) -> "CPMGInput":
        """Reject reversed or empty windows before converting to a run cfg."""
        if self.max_length_us <= self.min_length_us:
            raise ValueError("Maximum time must exceed minimum time")
        return self


def make_form(ctx: FormContext) -> Form[CPMGInput]:
    """Build CPMG counts, shared time window and fixed-capture module defaults."""
    return Form(
        CPMGInput,
        field_order=(
            "reset",
            "pi2_pulse",
            "pi_pulse",
            "readout",
            "times",
            "detune_ratio",
            "earlystop_snr",
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
            "times": LogPoints(max=16, points=5, include_zero=False),
            "detune_ratio": 0.1,
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
