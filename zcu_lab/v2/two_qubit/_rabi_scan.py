"""Shared configuration and raw results for two-line duration scans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.program.v2 import PulseCfg, SweepCfg
from zcu_tools.program.v2.modules import MuxReadoutCfg

from ._drive import DualLineCfg
from ._readout import JointRoles
from .joint_readout.core import GECenters


class DurationScanCfg(ConfigBase):
    """Shared B11 → two-line drive → double Xπ → mux inputs.

    duration is the requested positive envelope length in us on both lines.
    drive carries the source pair: common MHz frequency, signed gains, phases
    in degrees, explicit waveforms and timing. Only scalar length-settable
    envelopes are supported; arbitrary assets are not resized. roles and GE
    centers are saved capture values, never live lookup. Every shot is kept.
    """

    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
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
    ] = 1.0
    x180_f1: Annotated[
        PulseCfg,
        FieldMeta(
            label="F1 Xπ",
            description="填入 F1 x180 slot，先做 amplitude／length Rabi。frequency MHz、gain a.u. within [-1, 1]、duration us。",
        ),
    ]
    x180_f2: Annotated[
        PulseCfg,
        FieldMeta(
            label="F2 Xπ",
            description="填入 F2 x180 slot，先做 amplitude／length Rabi。frequency MHz、gain a.u. within [-1, 1]、duration us。",
        ),
    ]
    readout: Annotated[
        MuxReadoutCfg,
        FieldMeta(
            label="Mux readout",
            description="填入 full-group mux readout slot，先用 lookback 取得 timing us，再做 mux onetone。",
        ),
    ]
    roles: JointRoles
    drive: Annotated[
        DualLineCfg,
        FieldMeta(
            label="Dual-line drive",
            description="填入 F1/F2 charge-line pulses，先以 conditional spectroscopy 選頻，再確認 cz_f1／cz_f2 slots。",
        ),
    ]
    duration: Annotated[
        SweepCfg,
        FieldMeta(
            unit="us",
            label="Duration",
            description="填入兩條 charge-line 的共同 duration 起訖值，單位 us。先做 conditional spectroscopy 選頻，再由操作者確認 chevron 時間範圍。",
        ),
    ]
    f1_centers: Annotated[
        GECenters,
        FieldMeta(
            label="F1 GE centers",
            description="填入 singleshot/ge 取得的 F1 normalized complex IQ centers。",
        ),
    ]
    f2_centers: Annotated[
        GECenters,
        FieldMeta(
            label="F2 GE centers",
            description="填入 singleshot/ge 取得的 F2 normalized complex IQ centers。",
        ),
    ]

    @model_validator(mode="after")
    def validate_duration_scan(self) -> Self:
        """Reject invalid tones, routes, durations and unscalable waveforms."""
        self.roles.validate_readout(self.readout)
        if self.drive.f1.ch != self.x180_f1.ch or self.drive.f2.ch != self.x180_f2.ch:
            raise ValueError("Each drive must use its role's calibrated Xπ generator")
        durations = sweep2array(self.duration)
        if not np.all(np.isfinite(durations)) or np.any(durations <= 0):
            raise ValueError("Drive durations must be finite and positive")
        for pulse in (self.drive.f1, self.drive.f2):
            probe = pulse.model_copy(deep=True)
            probe.set_param("length", float(durations[0]))
        return self


@dataclass(frozen=True)
class RabiScanResult:
    """Raw (coordinate,duration,rounds*reps,N) complex IQ in original tone order.

    coordinates are relative phase degrees or common MHz frequency, according
    to the leaf's native AxesSpec. durations are requested us. shot_indices are
    round-major, tone_indices are original full mux indices. Unwritten whole
    rounds remain complex NaN; acquired nonfinite IQ is an error, not a sentinel.
    """

    coordinates: NDArray[np.float64]
    durations: NDArray[np.float64]
    shot_indices: NDArray[np.int64]
    tone_indices: NDArray[np.int64]
    signals: NDArray[np.complex128]
