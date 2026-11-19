"""Shared configuration for the paired RB experiments."""

from __future__ import annotations

from collections import Counter
from typing import Annotated, Self

import numpy as np
from pydantic import Field, model_validator
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.program.v2 import Pulse
from zcu_tools.program.v2.modules import MuxReadout

from .._drive import DualLineCfg
from .._readout import JointRoles
from .._sums import validate_classifiers
from .program import CZPhasePair


class RBInputs[ResetT](ConfigBase):
    """Shared RB inputs, parameterized by the experiment's accepted reset cfg.

    roles maps Q0 to F1 and Q1 to F2 at their original full mux tone positions.
    phase_pair is required, ordered and component-named. Delays are us, pulse
    frequency MHz and phase degrees. No DC device is driven. reset=None emits
    no reset. F1/F2 need circle classifiers with radius from GE Post before Run.
    depth counts random Cliffords, not primitive gates; strictly increasing
    nonnegative integers. seed contains distinct integer random.Random identities
    shared across depths/arms/rounds. The 31-bit bound preserves exact axes.
    reps is shots per host round; rounds remains a separate saved axis.
    """

    reset: Annotated[
        ResetT | None,
        FieldMeta(
            label="Reset", description="Optional reset at the start of every shot."
        ),
    ] = None
    x90_f1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F1 Xπ/2",
            description="Calibrated F1 X90 pulse, frequency MHz, gain [-1, 1] and duration us.",
        ),
    ]
    x90_f2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F2 Xπ/2",
            description="Calibrated F2 X90 pulse, frequency MHz, gain [-1, 1] and duration us.",
        ),
    ]
    drive: Annotated[
        DualLineCfg,
        FieldMeta(
            label="Dual-line drive",
            description="F1/F2 charge-line CZ pulses from the calibrated coupler slots.",
        ),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="Full-group mux readout with calibrated frequencies, gains and timing.",
        ),
    ]
    roles: JointRoles
    phase_pair: CZPhasePair
    depth: list[Annotated[int, Field(strict=True, ge=0)]] = Field(
        default_factory=lambda: [0, 1, 2, 3], min_length=1
    )
    seed: list[Annotated[int, Field(strict=True, ge=0, le=2**31 - 1)]] = Field(
        default_factory=lambda: [0, 1], min_length=1
    )
    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = Field(gt=0)
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = Field(default=1, gt=0)
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay",
            unit="us",
            description="Delay before the first repetition.",
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ]

    @model_validator(mode="after")
    def validate_sequence_inputs(self) -> Self:
        """Reject swapped phases, non-increasing depths, repeated seeds or wrong generators."""
        validate_classifiers(self.readout, self.roles)
        self.phase_pair.validate_roles(self.roles)
        depth_counts = Counter(self.depth)
        duplicates = sorted(depth for depth, count in depth_counts.items() if count > 1)
        if duplicates:
            raise ValueError(
                f"RB depths must be strictly increasing; duplicate values: {duplicates}"
            )
        if self.depth != sorted(self.depth):
            raise ValueError("RB depths must be strictly increasing")
        if len(set(self.seed)) != len(self.seed):
            raise ValueError("RB seeds must be distinct complete integer identities")
        for drive, x90 in (
            (self.drive.f1, self.x90_f1),
            (self.drive.f2, self.x90_f2),
        ):
            if drive.ch != x90.ch:
                raise ValueError("CZ and X90 must use the same captured role generator")
            if not isinstance(x90.phase, (int, float)) or not np.isfinite(x90.phase):
                raise ValueError("X90 phase must be a finite scalar")
        return self
