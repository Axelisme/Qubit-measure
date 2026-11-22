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
            description="Optional preparation before the first π/2 pulse.",
        ),
    ] = None
    pi2_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="π/2 pulse",
            description="Select the calibrated pulse used to start and end the echo.",
        ),
    ]
    pi_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Refocusing π pulse",
            description="Select the calibrated rotation at the midpoint of the total delay.",
        ),
    ]
    readout: Annotated[
        Readout.Cfg,
        FieldMeta(
            label="Readout",
            description="Select the pulse and integration window that measure IQ after the final rotation.",
        ),
    ]
    delay: Annotated[
        SweepCfg,
        FieldMeta(
            label="Total free-evolution delay",
            unit="µs",
            description="Sweep the sum of the two waits on either side of the π pulse.",
        ),
    ]
    detune_ratio: Annotated[
        float,
        FieldMeta(
            label="Detune ratio (fringes/step)",
            description="Enter the number of fringes per delay step for the final-pulse phase ramp.",
            decimals=3,
        ),
    ] = 0.1
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps", unit="count", description="Number of repetitions per round."
        ),
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Number of software acquisition rounds.",
        ),
    ] = 1
    earlystop_snr: Annotated[
        float | None,
        FieldMeta(
            label="Early-stop SNR",
            description="Stop averaging after a completed round reaches this SNR; None disables.",
        ),
    ] = Field(default=None, gt=0)
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Relaxation delay after each repetition.",
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
