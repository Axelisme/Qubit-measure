"""Monitor simultaneous T1 and Hahn echo with T1-based recovery delays."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from zcu_tools.experiment.metadata import RunSnapshot
from zcu_tools.experiment.workflows import Live, WorkflowEnv, workflow
from zcu_tools.program.v2 import ResetCfg
from zcu_tools.program.v2.modules import MuxReadoutCfg

from zcu_lab.components.qubit import QubitFluxSchema
from zcu_lab.v2.support.lifetime_models import LifetimeModelRecord
from zcu_lab.workflows.common.calibration_results import StageStatus
from zcu_lab.workflows.common.relaxation import echo_relaxation
from zcu_lab.workflows.common.stages import StageEnv
from zcu_lab.workflows.common.summary import RunSummary, SummaryAccumulator
from zcu_lab.workflows.common.summary_points import mux_summary_point
from zcu_lab.workflows.overnight_mux.pulses import FixedMuxPulses, OvernightMuxTone
from zcu_lab.workflows.overnight_mux.samples import append_point_samples
from zcu_lab.workflows.overnight_mux.stages import (
    EchoSettings,
    EchoTiming,
    T1Settings,
    measure_mux_echo,
    measure_mux_t1,
)
from zcu_lab.workflows.overnight_mux.trend import draw_overnight_mux_trend


class OvernightMuxPlan(BaseModel):
    """Fixed mux pulses and finite monitoring schedule; no flux setting.

    flux_frames maps each participant's fixed qubit name to its Start-captured
    device calibration block in native units, fixed across rounds and exports.
    An empty block means that qubit had no calibration; missing names are invalid.
    tones/readout retain the complete original readout order. participants
    selects original indices in acquisition order. Each selected tone supplies
    calibrated x180 drive and x90 pi2_pulse on its own generator. reset is the
    optional shared reset. outer_rounds counts T1/Echo pairs; start_interval_s
    spaces actual starts without catch-up. points is the common delay count
    per member. snapshot is the host's historical capture. Native cores own
    pulse, readout, participant and channel validation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    participants: list[int] = Field(min_length=1, description="Original tone indices")
    tones: list[OvernightMuxTone] = Field(min_length=1, description="Full tone order")
    reset: ResetCfg | None = None
    readout: MuxReadoutCfg
    outer_rounds: int = Field(ge=1, description="Finite T1/Echo pair count")
    start_interval_s: float = Field(gt=0, description="Actual start spacing, seconds")
    points: int = Field(default=51, ge=5, description="Common delay samples per member")
    snapshot: RunSnapshot
    flux_frames: dict[str, QubitFluxSchema] = Field(
        description="Start-captured flux blocks keyed by participant qubit name; empty blocks mean no calibration"
    )
    result_path: Path | None = Field(
        default=None,
        description="Exact entry Result path captured at Start. None means no entry (e.g. headless); skip auto samples and use offline --result later.",
    )

    @model_validator(mode="after")
    def require_participant_frames(self) -> "OvernightMuxPlan":
        """Reject absent participant frames before acquisition; empty blocks are valid."""
        for index, tone in enumerate(self.tones):
            if index in self.participants and tone.name not in self.flux_frames:
                raise ValueError(f"{tone.name!r}: missing captured flux frame")
        return self


class OvernightMuxTunables(BaseModel):
    """Set fixed delay grids, T1-based relaxation, averaging and fit acceptance.

    uniform selects T1's linear delay grid; False concentrates delays within
    the same start/stop window. Echo always keeps its linear grid.
    reps/rounds are native acquisition averages, not outer monitoring rounds.
    t1_guess_us/t2_guess_us are positive longest-qubit delay estimates in
    microseconds. All participants share each member's delay grid.
    sweep_factor times each guess gives its grid endpoint. T1 relax uses
    relax_factor times t1_guess_us. Echo relax uses the largest accepted T1
    among this round's participants, falling back to t1_guess_us. Both delays
    are floored at relax_min_us. Fits do not change grids, drive or readout.
    max_rel_err and max_fit_residual are strict dimensionless
    upper bounds on error/lifetime and normalized residual RMS; rejected fits
    keep raw data but omit accepted lifetime/error scalars. min_lifetime_points
    defaults to 5 fitted delays within [0, lifetime], including endpoints.
    Invalid quality diagnostics reject regardless of scalar error/residual.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    uniform: bool = Field(default=True, description="Uniform (linear) sweep for T1")
    reps: int = Field(default=1000, ge=1, description="Native averages per acquisition")
    rounds: int = Field(default=10, ge=1, description="Native acquisition rounds")
    t1_guess_us: float = Field(
        default=20.0, gt=0, description="Fixed T1 timing guess, microseconds"
    )
    t2_guess_us: float = Field(
        default=10.0, gt=0, description="Fixed Echo timing guess, microseconds"
    )
    t1_acquire_retry: int = Field(
        default=3, ge=0, description="Retries after a transient acquisition failure"
    )
    t1_earlystop_snr: float | None = Field(
        default=20.0,
        gt=0,
        description="Completed-round trace SNR threshold; None disables early stop",
    )
    t2_acquire_retry: int = Field(
        default=3, ge=0, description="Retries after a transient acquisition failure"
    )
    t2_earlystop_snr: float | None = Field(
        default=20.0,
        gt=0,
        description="Completed-round trace SNR threshold; None disables early stop",
    )
    sweep_factor: float = Field(
        default=4.0, gt=0, description="Grid endpoint / timing guess"
    )
    relax_factor: float = Field(
        default=5.0, gt=0, description="Relaxation delay / T1 estimate"
    )
    relax_min_us: float = Field(
        default=1.0, ge=0, description="Relaxation lower bound in us"
    )
    max_model_window_factor: float = Field(
        default=2.0,
        gt=0,
        description="Diagnostic lifetime / maximum fitted delay upper bound; no feedback change",
    )
    min_lifetime_points: int = Field(
        default=5,
        ge=1,
        description="Minimum fitted delay points with 0 <= delay <= lifetime",
    )
    max_rel_err: float = Field(
        default=0.5, gt=0, description="Strict error/lifetime upper bound"
    )
    max_fit_residual: float = Field(
        default=0.1, gt=0, description="Strict normalized residual RMS upper bound"
    )


class OvernightMuxQubit(BaseModel):
    """One participant's accepted lifetimes/errors in us and rejection reasons.

    name is the captured qubit identity, not a readout label. Missing scalars
    remain None. Reasons are None on acceptance, stopped on partial acquisition,
    or the native/fit failure diagnosis. Unattempted members keep None fields.
    """

    name: str
    t1_status: StageStatus = Field(
        default="skipped",
        description="Accepted, rejected, failed, or skipped until attempted",
    )
    t1_us: float | None = None
    t1_error_us: float | None = None
    t1_reason: str | None = None
    t1_models: tuple[LifetimeModelRecord, ...] = ()
    t2_status: StageStatus = Field(
        default="skipped",
        description="Accepted, rejected, failed, or skipped until attempted",
    )
    t2_us: float | None = None
    t2_error_us: float | None = None
    t2_reason: str | None = None
    t2_models: tuple[LifetimeModelRecord, ...] = ()


class OvernightMuxPoint(BaseModel):
    """One committed mux pair with UTC times and one row per participant.

    round_index is zero-based. actual_start is None until the wait completes.
    Native filenames are relative to the iteration directory, absent on native
    failure, retained on partial/rejected fits. status stays collecting on an
    unexpected exception, skipped on iteration cancellation, completed otherwise
    (including fit/native failures). Finally commits already collected fields.
    echo_relax_delay_us is the shared Echo cfg relaxation in us. echo_t1_source
    is accepted_t1 for this round's largest accepted participant T1, otherwise
    guess. Both stay None until Echo cfg setup.
    """

    model_config = ConfigDict(extra="forbid")
    round_index: int
    scheduled_start: datetime
    actual_start: datetime | None = None
    status: Literal["collecting", "completed", "skipped"] = "collecting"
    t1_file: str | None = None
    t2_file: str | None = None
    echo_relax_delay_us: float | None = None
    echo_t1_source: Literal["accepted_t1", "guess"] | None = None
    qubits: list[OvernightMuxQubit]


def _t1_settings(current: OvernightMuxTunables) -> T1Settings:
    """Copy this stage's settings from one captured tunables revision."""
    return T1Settings(
        uniform=current.uniform,
        reps=current.reps,
        rounds=current.rounds,
        acquire_retry=current.t1_acquire_retry,
        earlystop_snr=current.t1_earlystop_snr,
        sweep_factor=current.sweep_factor,
        relax_factor=current.relax_factor,
        relax_min_us=current.relax_min_us,
        max_rel_err=current.max_rel_err,
        max_fit_residual=current.max_fit_residual,
        min_lifetime_points=current.min_lifetime_points,
        max_model_window_factor=current.max_model_window_factor,
    )


def _echo_settings(current: OvernightMuxTunables) -> EchoSettings:
    """Copy this stage's settings from one captured tunables revision."""
    return EchoSettings(
        reps=current.reps,
        rounds=current.rounds,
        acquire_retry=current.t2_acquire_retry,
        earlystop_snr=current.t2_earlystop_snr,
        sweep_factor=current.sweep_factor,
        max_rel_err=current.max_rel_err,
        max_fit_residual=current.max_fit_residual,
        min_lifetime_points=current.min_lifetime_points,
        max_model_window_factor=current.max_model_window_factor,
    )


@workflow("overnight_mux", "Monitor simultaneous mux T1 and Hahn-echo pairs.")
async def overnight_mux[C](  # noqa: PLR0915  # tolerated: workflow flow reads top to bottom
    env: WorkflowEnv[C], plan: OvernightMuxPlan, tun: Live[OvernightMuxTunables]
) -> RunSummary:
    """Measure all participants' T1, then all participants' zero-detune Echo.

    Averaging, T1 sampling and acceptance follow overnight. Enter the longest
    qubit's T1/Echo guesses for their shared timing windows. Echo relax uses the
    largest accepted participant T1 from this round, otherwise t1_guess_us,
    times relax_factor and floored at relax_min_us. T1 relax uses the same
    floor with t1_guess_us. Fits do not alter grids, pulses, or later rounds.
    No flux setup, context read or calibration occurs. Save Completed raw before
    fitting; partial results are not fitted. Native/fit failures do not skip Echo.
    Iteration cancellation skips remaining work; Stop/errors propagate after
    finally commits collected fields, then appends one sample per participant
    to the captured entry when result_path is set. Append failures propagate
    without undoing commits or earlier rows. Pause retains local progress. Pair starts
    follow max(now, previous completed pair's actual start + interval), matching
    overnight. Trends retain scalars only and are not automatically saved.
    """
    summary = SummaryAccumulator(plan.outer_rounds)
    stage_env = StageEnv(plots=env.plots, snapshot=plan.snapshot, summary=summary)
    pulses = FixedMuxPulses(
        tones=tuple(plan.tones),
        readout=plan.readout,
        participants=tuple(plan.participants),
        reset=plan.reset,
    )
    last_actual_start: datetime | None = None
    trend_points: tuple[OvernightMuxPoint, ...] = ()
    for round_index, iteration in zip(
        range(plan.outer_rounds), env.iterations(), strict=False
    ):
        env.pbar("overnight mux rounds", total=plan.outer_rounds).set_progress(
            round_index
        )
        draw_overnight_mux_trend(
            env.plots, trend_points, start_interval_s=plan.start_interval_s
        )
        scheduled = datetime.now(UTC)
        if last_actual_start is not None:
            scheduled = max(
                scheduled, last_actual_start + timedelta(seconds=plan.start_interval_s)
            )
        point_record = OvernightMuxPoint(
            round_index=round_index,
            scheduled_start=scheduled,
            qubits=[
                OvernightMuxQubit(name=plan.tones[i].name) for i in plan.participants
            ],
        )
        try:
            await iteration.wait_until(scheduled)
            if iteration.cancelled:
                point_record.status = "skipped"
                continue
            actual = datetime.now(UTC)
            point_record.actual_start = actual
            current = tun.current
            t1 = await measure_mux_t1(
                iteration,
                stage_env,
                pulses,
                settings=_t1_settings(current),
                points=plan.points,
                guess_us=current.t1_guess_us,
            )
            point_record.t1_file = t1.file
            qubits = dict(zip(plan.participants, point_record.qubits, strict=True))
            for fit in t1.qubits:
                qubit = qubits[fit.index]
                qubit.t1_status = fit.status
                qubit.t1_us = fit.lifetime_us
                qubit.t1_error_us = fit.error_us
                qubit.t1_reason = fit.reason
                qubit.t1_models = fit.models
            if iteration.cancelled:
                point_record.status = "skipped"
                continue
            current = tun.current
            point_record.echo_relax_delay_us, point_record.echo_t1_source = (
                echo_relaxation(
                    (q.t1_us for q in point_record.qubits),
                    t1_guess_us=current.t1_guess_us,
                    relax_factor=current.relax_factor,
                    relax_min_us=current.relax_min_us,
                )
            )
            echo = await measure_mux_echo(
                iteration,
                stage_env,
                pulses,
                settings=_echo_settings(current),
                points=plan.points,
                timing=EchoTiming(
                    guess_us=current.t2_guess_us,
                    relax_delay_us=point_record.echo_relax_delay_us,
                ),
            )
            point_record.t2_file = echo.file
            for fit in echo.qubits:
                qubit = qubits[fit.index]
                qubit.t2_status = fit.status
                qubit.t2_us = fit.lifetime_us
                qubit.t2_error_us = fit.error_us
                qubit.t2_reason = fit.reason
                qubit.t2_models = fit.models
            point_record.status = "skipped" if iteration.cancelled else "completed"
            last_actual_start = actual
            trend_points += (point_record,)
            draw_overnight_mux_trend(
                env.plots, trend_points, start_interval_s=plan.start_interval_s
            )
        finally:
            iteration.commit(point_record)
            append_point_samples(env.identity.run_id, plan, point_record, iteration.dir)
            summary.points.append(mux_summary_point(point_record))
    env.pbar("overnight mux rounds", total=plan.outer_rounds).set_progress(
        plan.outer_rounds
    )
    return summary.finish()
