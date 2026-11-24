"""Four-input CZ return conditioned on a measured pre-CZ herald."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import Experiment
from zcu_tools.program.v2 import Pulse, Reset, SweepCfg
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support.paired_counts import POINT_ALIAS_DESCRIPTION
from zcu_lab.v2.support.paired_drive import QubitDriveCfg


class HeraldedCZAmplitudeCfg(ConfigBase):
    """Four nominal preparations followed by a visible herald, double Xπ, CZ,
    double Xπ and terminal mux. Each of the two tone fields is a zero-based
    position in the full mux tone list, not an ADC channel number. The same
    calibrated readout is used twice. Reset is ordinary only (None omits it).
    Signed drive gains share one increasing nonnegative dimensionless scale.
    Reps are attempts per preparation and scale per round.
    """

    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select an ordinary reset before nominal preparation, or leave blank.",
        ),
    ] = None
    x180_qubit1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit1 Xπ",
            description="Calibrated Qubit1 Xπ for nominal preparation and both inversion pairs.",
        ),
    ]
    x180_qubit2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit2 Xπ",
            description="Calibrated Qubit2 Xπ for nominal preparation and both inversion pairs.",
        ),
    ]
    drive: Annotated[
        QubitDriveCfg,
        FieldMeta(
            label="Two-line CZ drive",
            description="Simultaneous CZ pulses on the two qubit charge lines. Multiply their signed source gains by the common sweep multiplier; keep their relative phase fixed.",
        ),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="Full-group mux readout with GE Post radii and the wait after each read. "
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
    scale: Annotated[
        SweepCfg,
        FieldMeta(
            label="Common CZ gain multiplier",
            unit="a.u.",
            description="Multiply both signed CZ gains by each value; 0 omits drive and 1 uses the source gains. Include 0 and scan through the first positive return.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Shots attempted for each nominal preparation and gain point in each round.",
        ),
    ] = Field(gt=0)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Number of complete gain sweeps whose class counts are added.",
        ),
    ] = Field(default=1, gt=0)
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


@dataclass(frozen=True)
class HeraldedCZAmplitudeResult:
    """Dimensionless scales and float64 counts from completed rounds.

    sums has axes (prepared Qubit1, prepared Qubit2, scale, herald Qubit1,
    herald Qubit2, terminal Qubit1, terminal Qubit2). Preparations are I, Xπ;
    each classification is g, e, Other. NaN means unmeasured, not zero counts.
    """

    scales: NDArray[np.float64]
    sums: NDArray[np.float64]


class HeraldedCZAmplitudeExp(
    Experiment[HeraldedCZAmplitudeResult, HeraldedCZAmplitudeCfg]
):
    """Acquire and analyze the four measured CZ return branches."""
