"""Shared inputs, summed-count acquisition and results for two-line duration scans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator
from qick import QickConfig
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.program.v2 import (
    Classify,
    Pulse,
    PulseCfg,
    SweepCfg,
    TimelineBuilder,
    TimelineProgramV2,
    shot_sums,
)
from zcu_tools.program.v2.modules import MuxReadout, MuxReadoutCfg, Reset

from zcu_lab.v2.support import live_line

from ._drive import DualLineCfg, dual_line_drive
from ._readout import JointRoles
from ._sums import (
    joint_probabilities,
    terminal_counts,
    validate_classifiers,
)


class DurationScanCfg(ConfigBase):
    """Shared B11 → two-line drive → double Xπ → mux inputs.

    duration is the requested positive envelope length in us on both lines.
    drive carries the source pair: common MHz frequency, signed gains, phases
    in degrees, explicit waveforms and timing. Only scalar length-settable
    envelopes are supported; arbitrary assets are not resized. roles are saved
    capture values. F1/F2 require circle classifiers with radius from GE Post.
    reset=None emits no reset; otherwise reset precedes preparation and herald.
    Acquisition accumulates rounds without storing individual shots.
    """

    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Optional reset before each shot's reads and preparation.",
        ),
    ] = None
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

    @model_validator(mode="after")
    def validate_duration_scan(self) -> Self:
        """Reject invalid tones, routes, durations and unscalable waveforms."""
        validate_classifiers(self.readout, self.roles)
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
    """Coordinates, requested durations in us and unchanged float64 class counts.

    coordinates are relative phase degrees or common MHz frequency, as named by
    the leaf AXES_SPEC. sums is (Nc,Nd,3,3), or (Nc,Nd,3,3,3,3) with herald.
    Class order is g/e/other. Axis order is terminal F1/F2, preceded by herald
    F1/F2 when present. Unmeasured points remain NaN; completed rounds survive
    cancellation. There is no shot axis or IQ. Check the RunContext stop signal
    for execution errors before accepting partial results.
    """

    coordinates: NDArray[np.float64]
    durations: NDArray[np.float64]
    sums: NDArray[np.float64]


def build_program(
    cfg: DurationScanCfg,
    soccfg: QickConfig,
    drive: DualLineCfg,
    duration: float,
    *,
    heralded: bool,
) -> TimelineProgramV2:
    """Compile optional reset → optional herald → B11 → drive → double Xπ → terminal.

    drive is the coordinate-specific pair, duration is its positive common
    envelope length in us. Inputs remain unchanged. The same mux object emits
    herald and terminal. Every build creates fresh modules; invalid classifier,
    envelope or resource configuration fails before acquisition.
    """
    validate_classifiers(cfg.readout, cfg.roles)
    point_drive = drive.model_copy(deep=True)
    for pulse in (point_drive.f1, point_drive.f2):
        pulse.set_param("length", duration)
    builder = TimelineBuilder(
        soccfg,
        reps=cfg.reps,
        initial_delay=cfg.initial_delay,
        relax_delay=cfg.relax_delay,
    )
    readout = MuxReadout("readout", cfg.readout)
    if cfg.reset is not None:
        builder.emit(Reset("reset", cfg.reset))
    if heralded:
        builder.emit(readout, tag="herald")
    builder.emit(Pulse("prepare_f1", cfg.x180_f1))
    builder.emit(Pulse("prepare_f2", cfg.x180_f2))
    builder.emit(dual_line_drive(point_drive, frequency=float(point_drive.f1.freq)))
    builder.emit(Pulse("contrast_f1", cfg.x180_f1))
    builder.emit(Pulse("contrast_f2", cfg.x180_f2))
    builder.emit(readout, tag="terminal")
    return builder.build()


def acquire_counts(
    config: DurationScanCfg,
    coordinates: NDArray[np.float64],
    drives: list[DualLineCfg],
    *,
    context: RunContext,
    heralded: bool,
) -> RabiScanResult:
    """Scan coordinate-specific drives and durations, keeping 9/81-class sums.

    coordinates has one value per drive. Schedule owns cfg copies and stop
    handling; run_acquire_summed accumulates rounds on the acquisition side.
    Live traces show P00 per coordinate, conditioned on herald (g,g) if enabled.
    Counts include other in the denominator. No host round or shot axis is made.
    """
    validate_classifiers(config.readout, config.roles)
    if len(drives) != len(coordinates) or not np.all(np.isfinite(coordinates)):
        raise ValueError("Each finite coordinate requires one explicit drive pair")
    durations = sweep2array(config.duration)
    viewer = live_line(
        context.plots,
        "measurement",
        "Drive duration (us)",
        "Terminal P00 | herald (g,g)" if heralded else "Terminal P00",
        num_lines=len(coordinates),
    )

    def update_view(sums: NDArray[np.float64]) -> None:
        counts = terminal_counts(sums, ("g", "g")) if heralded else sums
        viewer.update(durations, joint_probabilities(counts)[..., 0, 0])

    class_shape = (3, 3, 3, 3) if heralded else (3, 3)
    buffer = SignalBuffer(
        (len(coordinates), len(durations), *class_shape),
        dtype=np.float64,
        on_update=update_view,
    )
    with Schedule(config, buffer, scope=context.scope) as sched:
        for index, step in sched.scan("coordinate", list(range(len(drives)))):
            for duration, duration_step in step.scan("duration", durations.tolist()):
                cfg = duration_step.cfg
                prog = build_program(
                    cfg, context.soccfg, drives[index], duration, heralded=heralded
                )
                duration_step.run_acquire_summed(
                    prog,
                    context.soc,
                    processor=shot_sums(
                        prog,
                        [
                            Classify(label, tone)
                            for label in (
                                ("herald", "terminal") if heralded else ("terminal",)
                            )
                            for tone in (cfg.roles.f1_tone, cfg.roles.f2_tone)
                        ],
                    ),
                    rounds=cfg.rounds,
                    options=RunOptions(),
                )
    buffer.trigger_update(flush=True)
    return RabiScanResult(coordinates, durations, buffer.array)
