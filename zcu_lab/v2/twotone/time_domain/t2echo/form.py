"""Experiment starting values from fixed captures and injected templates."""

from typing import Annotated

from pydantic import Field
from zcu_tools.cfg_model import ConfigBase, FieldMeta, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.program.v2 import Pulse, Readout, Reset, SweepCfg

from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time
from zcu_lab.v2.support.measure.seed_metadata import (
    pulse_readout_seed_metadata,
)


class T2EchoInput(ConfigBase):
    """Captured pulses, delay sweep and phase advance in fringes per step."""

    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Choose a reset before the first π/2 pulse, or leave empty to skip it.",
        ),
    ] = None
    pi2_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Pi/2 Pulse",
            description="Enter the Rabi-calibrated π/2 pulse for preparation and final projection.",
        ),
    ]
    pi_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Pi Pulse",
            description="Enter the Rabi-calibrated π pulse that refocuses the echo at mid-delay.",
        ),
    ]
    readout: Annotated[
        Readout.Cfg,
        FieldMeta(
            label="Readout",
            description="Enter the calibrated single-tone readout after the final π/2 pulse.",
        ),
    ]
    delay: Annotated[
        SweepCfg,
        FieldMeta(
            label="Total echo delay",
            unit="µs",
            description="Enter the sum of both free-evolution intervals (µs).",
        ),
    ]
    detune_ratio: Annotated[
        float,
        FieldMeta(
            label="Detune ratio (fringes/step)",
            description="Enter the fringe advance per delay step for the final π/2 pulse, or 0 to disable its phase ramp.",
            decimals=3,
        ),
    ] = 0.1
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter the shots per delay in each round (count).",
        ),
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter the maximum number of complete sweeps to average (count).",
        ),
    ] = 1
    earlystop_snr: Annotated[
        float | None,
        FieldMeta(
            label="Early-stop SNR",
            description="Enter the trace SNR to stop averaging after a complete sweep, or leave empty to run all rounds.",
        ),
    ] = Field(default=None, gt=0)
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Enter the relaxation wait after each shot (µs).",
        ),
    ] = 1.0


def make_form(ctx: FormContext) -> Form[T2EchoInput]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        T2EchoInput,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "pi_pulse": source(
                ctx,
                "$qubit.module.x180",
                Template("pi_pulse", bindings={"qubit": "$qubit"}),
            ),
            "pi2_pulse": source(
                ctx,
                "$qubit.module.x90",
                Template("pi2_pulse", bindings={"qubit": "$qubit"}),
            ),
            "readout": source(
                ctx,
                "$resonator.module.readout",
                Template(
                    "readout",
                    bindings={"resonator": "$resonator"},
                    seed_metadata=pulse_readout_seed_metadata(gain=0.1),
                ),
            ),
            "delay": Range(0.0, "1.5 * $qubit.t2e", expts=101),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    )
