"""Common signed-gain scaling of an ordinary simultaneous two-line CZ pulse."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Annotated, ClassVar, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Pulse, SweepCfg
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_line

from .._drive import DualLineCfg, dual_line_drive
from .._readout import JointRoles, collect_joint_shots
from ..joint_readout.core import GECenters
from .analysis import CZAmplitudeAnalysis, analyze_return


class CZAmplitudeCfg(ConfigBase):
    """B11 → simultaneous CZ → Xπ⊗Xπ → full-tone mux, without heralding.

    modules contains calibrated preparation/contrast pulses and the complete mux.
    roles fixes component identities and original tone positions from capture.
    drive is the scalar two-line template (MHz, µs, degrees, signed gains).
    scales is a finite, strictly increasing nonnegative common gain multiplier;
    neither line's relative phase, waveform, frequency nor gain ratio changes.
    f1_centers/f2_centers are saved nearest-GE centers; every shot is classified.
    reps shots per host round are retained in round-major order. No bias device,
    reset, radius filtering, confusion inversion or pure-state assumption is used.
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
        Pulse.Cfg,
        FieldMeta(
            label="F1 Xπ",
            description="填入 F1 x180 slot，先做 amplitude／length Rabi。frequency MHz、gain a.u. within [-1, 1]、duration us。",
        ),
    ]
    x180_f2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F2 Xπ",
            description="填入 F2 x180 slot，先做 amplitude／length Rabi。frequency MHz、gain a.u. within [-1, 1]、duration us。",
        ),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="填入 full-group mux readout slot。先用 lookback 取得 timing，再做 mux onetone，確認 frequency MHz、gain a.u. within [-1, 1] 與 timing us。",
        ),
    ]
    roles: JointRoles
    drive: Annotated[
        DualLineCfg,
        FieldMeta(
            label="Dual-line drive",
            description="填入 F1/F2 charge-line pulses。先做 conditional spectroscopy／chevron，再確認 coupler 的 cz_f1／cz_f2 slots。",
        ),
    ]
    scales: Annotated[
        SweepCfg,
        FieldMeta(
            label="Gain scale",
            unit="dimensionless",
            description="填入 common gain scale 起訖值。先用 chevron 選定 CZ pulse，再由操作者確認幅度範圍。",
        ),
    ]
    f1_centers: Annotated[
        GECenters,
        FieldMeta(
            label="F1 GE centers",
            description="填入 singleshot/ge 取得的 F1 ground／excited complex IQ centers，使用 normalized raw ADC units。",
        ),
    ]
    f2_centers: Annotated[
        GECenters,
        FieldMeta(
            label="F2 GE centers",
            description="填入 singleshot/ge 取得的 F2 ground／excited complex IQ centers，使用 normalized raw ADC units。",
        ),
    ]

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Reject tone/line mismatches and invalid or degenerate gain axes before I/O."""
        self.roles.validate_readout(self.readout)
        if self.drive.f1.ch != self.x180_f1.ch or self.drive.f2.ch != self.x180_f2.ch:
            raise ValueError("Each CZ line must use its role's calibrated Xπ generator")
        scales = sweep2array(self.scales)
        if (
            len(scales) < 2
            or not np.all(np.isfinite(scales))
            or np.any(scales < 0)
            or np.any(np.diff(scales) <= 0)
        ):
            raise ValueError(
                "Common gain scales must be finite, nonnegative and strictly increasing"
            )
        return self


@dataclass(frozen=True)
class CZAmplitudeResult:
    """Raw IQ (scale, rounds*reps, full tones), normalized by owner collect_raw_iq.

    scales are the requested dimensionless coordinates. shot_indices are int64
    [0..rounds*reps-1] in round-major/reps order. tone_indices are int64 [0..N-1]
    in original program order. Cancelled unwritten whole rounds remain complex
    NaN. Acquired nonfinite IQ is an error, not a missing-data sentinel.
    """

    scales: NDArray[np.float64]
    shot_indices: NDArray[np.int64]
    tone_indices: NDArray[np.int64]
    signals: NDArray[np.complex128]


class CZAmplitudeExp(Experiment[CZAmplitudeResult, CZAmplitudeCfg]):
    """Stateless amplitude scan with native raw/cfg persistence."""

    AXES_SPEC: ClassVar[AxesSpec[CZAmplitudeResult, CZAmplitudeCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("tone_indices", "Tone", "", dtype=np.int64),
            AxesSpec.Axis("shot_indices", "Shot", "", dtype=np.int64),
            AxesSpec.Axis("scales", "Common gain scale", ""),
        ),
        z=AxesSpec.Z("signals", "Raw IQ", "a.u."),
        result_type=CZAmplitudeResult,
        cfg_type=CZAmplitudeCfg,
        cfg_schema_version="3.0",
        tag="two_qubit/cz_amplitude",
    )

    def run(self, config: CZAmplitudeCfg, *, context: RunContext) -> CZAmplitudeResult:
        """Retain same-shot full tones in fixed scale/host-round slots.

        Only the runner-owned copy uses rounds=1. Schedule preserves completed
        slots on cancel; final raw conversion overwrites, never appends. Caller
        must check cancel_signal.raise_if_error() to distinguish acquisition
        faults from cancellation. No library lookup, writeback or saving occurs.
        """
        cfg = config.model_copy(deep=True, update={"rounds": 1})
        scales = sweep2array(cfg.scales)
        ntones = len(cfg.readout.tones)
        viewer = live_line(
            context.plots,
            "measurement",
            "Common gain scale",
            "Completed shots",
            num_lines=1,
        )

        def update_view(raw: NDArray[np.complex128]) -> None:
            viewer.update(scales, np.all(np.isfinite(raw), axis=-1).sum(axis=(1, 2)))

        buffer = SignalBuffer(
            (len(scales), config.rounds, cfg.reps, ntones), on_update=update_view
        )
        with Schedule(cfg, buffer, scope=context.scope) as sched:
            for scale, step in sched.scan("scale", scales.tolist()):
                builder = step.prog_builder(
                    context.soc,
                    context.soccfg,
                    reps=step.cfg.reps,
                    initial_delay=step.cfg.initial_delay,
                    relax_delay=step.cfg.relax_delay,
                )
                m = cfg
                builder.add_pulse("prepare_f1", m.x180_f1)
                builder.add_pulse("prepare_f2", m.x180_f2)
                drive = DualLineCfg(
                    f1=cfg.drive.f1.with_updates(gain=float(cfg.drive.f1.gain) * scale),
                    f2=cfg.drive.f2.with_updates(gain=float(cfg.drive.f2.gain) * scale),
                )
                builder.add(
                    dual_line_drive(drive, frequency=float(drive.f1.freq), name="cz")
                )
                builder.add_pulse("contrast_f1", m.x180_f1)
                builder.add_pulse("contrast_f2", m.x180_f2)
                builder.add(MuxReadout("readout", m.readout))
                prog = builder.build()
                for _, round_step in step.repeat("round", config.rounds):
                    _ = round_step.prog_builder(
                        context.soc,
                        context.soccfg,
                        reps=round_step.cfg.reps,
                        initial_delay=round_step.cfg.initial_delay,
                        relax_delay=round_step.cfg.relax_delay,
                    ).run_program(
                        prog,
                        raw2signal_fn=partial(collect_joint_shots, prog),
                        rounds=round_step.cfg.rounds,
                    )
        buffer.trigger_update(flush=True)
        return CZAmplitudeResult(
            scales,
            np.arange(config.rounds * cfg.reps, dtype=np.int64),
            np.arange(ntones, dtype=np.int64),
            buffer.array.reshape(len(scales), config.rounds * cfg.reps, ntones),
        )

    def analyze(
        self,
        source: RunRecord[CZAmplitudeCfg, CZAmplitudeResult],
        options: None,
        *,
        plots: Plots,
    ) -> CZAmplitudeAnalysis:
        """Fit the saved-source zero-phase P00 first-return model, never current cfg.

        Native cfg/raw validation failures raise ValueError. Incomplete coverage,
        numerical failure, degeneracy or nonfinite uncertainty instead retain
        diagnostics with a reason and no calibration scale. Only measured 0 through
        a full first positive period can supply a scale. Uncertainty is reported
        using fit_quality, not a new statistical or fidelity acceptance gate.
        """
        del options
        return analyze_return(source, plots=plots)
