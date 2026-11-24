"""Input checks common to conditional phase and heralded conditional phase."""

import numpy as np
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.program.v2 import Pulse, SweepCfg
from zcu_tools.program.v2.modules import MuxReadout

from .paired_counts import validate_qubit_tones
from .paired_drive import SweptQubitDriveCfg


def validate_conditional_phase_inputs(
    readout: MuxReadout.Cfg,
    tones: tuple[int, int],
    drive: SweptQubitDriveCfg,
    rotations: tuple[tuple[Pulse.Cfg, Pulse.Cfg], tuple[Pulse.Cfg, Pulse.Cfg]],
    frequency: SweepCfg,
    gate_counts: list[int],
) -> None:
    """Validate conditional phase inputs without changing the requested scan.

    Supply full mux readout, selected tone indices, CZ drive and (Xπ, Xπ/2)
    pulse pairs in Qubit1/Qubit2 order. Frequency is a sweep in MHz. Raise
    ValueError for invalid classifiers, generators that disagree within a role,
    a nonfinite or nonscalar Xπ/2 phase, nonincreasing scan points or repeated
    train lengths.
    """
    validate_qubit_tones(readout, *tones)
    for drive_pulse, (pi, half_pi) in zip(
        (drive.qubit1, drive.qubit2), rotations, strict=True
    ):
        if drive_pulse.ch != pi.ch or drive_pulse.ch != half_pi.ch:
            raise ValueError("CZ, Xπ and X90 must use the same qubit generator")
        if not isinstance(half_pi.phase, (int, float)) or not np.isfinite(
            half_pi.phase
        ):
            raise ValueError("X90 phase must be a finite scalar")
    if np.any(np.diff(sweep2array(frequency)) <= 0):
        raise ValueError("Common frequency coordinates must be strictly increasing")
    if len(set(gate_counts)) != len(gate_counts):
        raise ValueError("Gate counts must be unique")
