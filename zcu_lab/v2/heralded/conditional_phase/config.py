"""Native inputs for heralded conditional phase."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.program.v2 import Pulse, Reset, SweepCfg
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support.paired_counts import POINT_ALIAS_DESCRIPTION
from zcu_lab.v2.support.paired_drive import SweptQubitDriveCfg
from zcu_lab.v2.support.paired_phase_inputs import validate_conditional_phase_inputs


class ConditionalPhaseHeraldedCfg(ConfigBase):
    """Optional reset and mux herald before independent classified X/Y projections.

    Frequencies are MHz, phases degrees and delays us. reps is shots per round;
    rounds accumulate counts at acquisition. Qubit1/Qubit2 require circle classifiers
    with GE Post radii. reset=None skips reset; otherwise reset starts each shot.
    target is lowercase qubit1/qubit2; gate_count contains unique positive train lengths.
    Both reads share the mux classifiers; readout.post_delay is the wait after herald.
    """

    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(label="Reset", description="Optional reset before each shot."),
    ] = None
    x180_qubit1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit1 Xπ",
            description="Flip Qubit1 in the excited spectator branch when Qubit2 is the Ramsey target.",
        ),
    ]
    x180_qubit2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit2 Xπ",
            description="Flip Qubit2 in the excited spectator branch when Qubit1 is the Ramsey target.",
        ),
    ]
    x90_qubit1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit1 Xπ/2",
            description="Prepare and project Ramsey coherence when Qubit1 is the target.",
        ),
    ]
    x90_qubit2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit2 Xπ/2",
            description="Prepare and project Ramsey coherence when Qubit2 is the target.",
        ),
    ]
    drive: Annotated[
        SweptQubitDriveCfg,
        FieldMeta(
            label="Dual-line drive",
            description="Use both calibrated CZ drive lines for each gate in the Ramsey train.",
        ),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="Full-group mux readout with calibrated frequencies, gains and timing. "
            + POINT_ALIAS_DESCRIPTION,
        ),
    ]
    qubit1_tone: Annotated[
        int,
        FieldMeta(
            label="Qubit1 tone index",
            description="Zero-based Qubit1 position in the full mux readout.",
        ),
    ]
    qubit2_tone: Annotated[
        int,
        FieldMeta(
            label="Qubit2 tone index",
            description="Zero-based Qubit2 position in the full mux readout.",
        ),
    ]
    target: Annotated[
        Literal["qubit1", "qubit2"],
        FieldMeta(
            label="Ramsey target",
            description="Choose which qubit receives Xπ/2 Ramsey pulses. The other is the spectator.",
        ),
    ] = "qubit1"
    gate_count: Annotated[
        list[Annotated[int, Field(strict=True, ge=1)]],
        FieldMeta(
            label="CZ gates per shot",
            unit="count",
            description="Enter consecutive CZ train lengths to compare phase accumulation.",
        ),
    ] = Field(default_factory=lambda: [1, 2, 3], min_length=2)
    frequency: Annotated[
        SweepCfg,
        FieldMeta(
            unit="MHz",
            label="Common CZ drive frequency",
            description="Sweep the shared RF carrier of both CZ drives across the expected π crossing.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Attempted shots per sweep point in one round, before herald selection.",
        ),
    ] = Field(gt=0)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Number of acquisitions accumulated at each sweep point.",
        ),
    ] = 1
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay",
            unit="µs",
            description="Delay before the first repetition.",
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relaxation delay",
            unit="µs",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ]

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Reject mismatched qubit generators, nonfinite frequencies and repeated n."""
        validate_conditional_phase_inputs(
            self.readout,
            (self.qubit1_tone, self.qubit2_tone),
            self.drive,
            ((self.x180_qubit1, self.x90_qubit1), (self.x180_qubit2, self.x90_qubit2)),
            self.frequency,
            self.gate_count,
        )
        return self
