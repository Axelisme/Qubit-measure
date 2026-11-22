"""Shared configuration for the paired RB experiments."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, field_validator, model_validator
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.program.v2 import Pulse
from zcu_tools.program.v2.modules import MuxReadout

from .._cz_phase import CZPhasePair
from .._drive import QubitDriveCfg
from .._rb_scheduling import validate_x_pulses
from .._readout import POINT_ALIAS_DESCRIPTION
from .._sums import validate_qubit_tones


class RBInputs[ResetT](ConfigBase):
    """Shared RB inputs, parameterized by the experiment's accepted reset cfg.

    The tone indices map Q0 to Qubit1 and Q1 to Qubit2 in the full mux group.
    phase_pair supplies the required logical Z angles in that order. Delays are us, pulse
    frequency MHz and phase degrees. No DC device is driven. reset=None emits
    no reset. Qubit1/Qubit2 need circle classifiers with radius from GE Post before Run.
    depth counts random Cliffords, not primitive gates. seed gives integer
    random.Random identities shared across depths/arms/rounds. Both lists are
    deduplicated and sorted before use or saving. The 31-bit seed bound
    preserves exact axes.
    reps is shots per acquisition round; rounds sets how many rounds to sum.
    """

    reset: Annotated[
        ResetT | None,
        FieldMeta(
            label="Reset",
            description="Select the preparation module before the sequence; leave blank for no reset.",
        ),
    ] = None
    x90_qubit1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit1 Xπ/2",
            description="Enter the calibrated Qubit1 Xπ/2 pulse from Rabi for the Clifford rotations.",
        ),
    ]
    x90_qubit2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit2 Xπ/2",
            description="Enter the calibrated Qubit2 Xπ/2 pulse from Rabi for the Clifford rotations.",
        ),
    ]
    x180_qubit1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit1 Xπ",
            description="Enter the calibrated Qubit1 Xπ pulse from Rabi for the Clifford rotations.",
        ),
    ]
    x180_qubit2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit2 Xπ",
            description="Enter the calibrated Qubit2 Xπ pulse from Rabi for the Clifford rotations.",
        ),
    ]
    drive: Annotated[
        QubitDriveCfg,
        FieldMeta(
            label="Two-line CZ drive",
            description="Enter the calibrated coupler pulses inserted after each interleaved Clifford.",
        ),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="Enter the calibrated full-group mux readout for state classification. "
            + POINT_ALIAS_DESCRIPTION,
        ),
    ]
    qubit1_tone: Annotated[
        int,
        FieldMeta(
            label="Qubit1 tone index",
            description="Zero-based position in the full mux readout.",
        ),
    ]
    qubit2_tone: Annotated[
        int,
        FieldMeta(
            label="Qubit2 tone index",
            description="Zero-based position in the full mux readout.",
        ),
    ]
    phase_pair: Annotated[
        CZPhasePair,
        FieldMeta(
            label="Local Z corrections",
            description="Enter the pair from Local Z to correct the logical frames after each CZ.",
        ),
    ]
    depth: Annotated[
        list[Annotated[int, Field(strict=True, ge=0)]],
        FieldMeta(
            label="Random Clifford depths",
            unit="count",
            description="Enter the Clifford counts before recovery; 0 runs recovery and readout without random Cliffords.",
        ),
    ] = Field(default_factory=lambda: [0, 1, 2, 3], min_length=1)
    seed: Annotated[
        list[Annotated[int, Field(strict=True, ge=0, le=2**31 - 1)]],
        FieldMeta(
            label="Sequence seeds",
            description="Enter integer sequence identifiers (0 to 2³¹−1) shared between arms; duplicates do not add sequences.",
        ),
    ] = Field(default_factory=lambda: [0, 1], min_length=1)
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter the shots per depth in each acquisition round.",
        ),
    ] = Field(gt=0)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter the acquisition rounds to sum for each seed and arm.",
        ),
    ] = Field(default=1, gt=0)
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay",
            unit="µs",
            description="Set the wait (µs) before the first shot in each round.",
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Enter the relaxation wait (µs) after each shot before the next repetition.",
        ),
    ]

    @field_validator("depth", "seed")
    @classmethod
    def sort_unique(cls, values: list[int]) -> list[int]:
        """Return sorted unique values so the saved cfg matches the data axes."""
        return sorted(set(values))

    @model_validator(mode="after")
    def validate_sequence_inputs(self) -> Self:
        """Reject invalid tone indices, classifiers, pulse phases and generators."""
        validate_qubit_tones(self.readout, self.qubit1_tone, self.qubit2_tone)
        validate_x_pulses(
            (self.x90_qubit1, self.x90_qubit2),
            (self.x180_qubit1, self.x180_qubit2),
            self.drive,
        )
        return self
