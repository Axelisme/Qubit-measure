"""Fixed-flux T1 followed by zero-detune Hahn echo, committed as one pair."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from zcu_tools.experiment.metadata import RunSnapshot
from zcu_tools.experiment.workflows import (
    Live,
    WorkflowEnv,
    workflow,
)
from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg

from zcu_lab.v2.support.lifetime_models import LifetimeModelRecord
from zcu_lab.workflows.common.calibration_results import StageStatus
from zcu_lab.workflows.common.devices import set_flux
from zcu_lab.workflows.common.gain import GainClamp
from zcu_lab.workflows.common.relaxation import echo_relaxation
from zcu_lab.workflows.common.stages import StageEnv
from zcu_lab.workflows.common.summary import RunSummary, SummaryAccumulator
from zcu_lab.workflows.common.summary_points import overnight_summary_point
from zcu_lab.workflows.overnight.pulses import FixedPulses
from zcu_lab.workflows.overnight.samples import append_point_sample
from zcu_lab.workflows.overnight.stages import (
    EchoSettings,
    EchoTiming,
    T1Settings,
    measure_echo,
    measure_t1,
)
from zcu_lab.workflows.overnight.trend import draw_overnight_trend


class OvernightPlan(BaseModel):
    """Monitor T1 and Echo drift at fixed flux without calibration.

    flux_dev names the connected driver; required flux_unit is A, V or native
    (Fake), with no conversion. flux is its absolute setpoint. outer_rounds counts
    complete T1/Echo pairs; start_interval_s is the positive wall-clock spacing
    between actual pair starts in seconds. points is the uniform delay-grid size
    (at least five). qub_pulse is the already calibrated pi pulse; Echo uses the
    same waveform with half gain for pi/2. readout is the fixed pulse-backed
    readout. snapshot is host-captured historical entry/point/roles, not live
    state or output paths. Native cfg validation and driver limits still apply.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    flux_dev: str = Field(min_length=1, description="Connected flux driver name")
    flux_unit: Literal["A", "V", "native"] = Field(
        description="Explicit flux unit; no conversion"
    )
    flux: float = Field(description="Fixed absolute setpoint in flux_unit")
    outer_rounds: int = Field(
        ge=1, description="Finite number of committed T1/Echo pairs"
    )
    start_interval_s: float = Field(
        gt=0, description="Spacing between actual pair starts, seconds"
    )
    points: int = Field(
        default=51, ge=5, description="Uniform delay samples per acquisition"
    )
    qub_pulse: PulseCfg = Field(description="Fixed calibrated pi drive")
    readout: PulseReadoutCfg = Field(description="Fixed pulse-backed readout")
    snapshot: RunSnapshot = Field(description="Historical host capture")
    result_path: Path | None = Field(
        default=None,
        description="Exact entry Result path captured at Start. None means no entry (e.g. headless); skip auto samples and use offline --result later.",
    )


class OvernightTunables(BaseModel):
    """Set fixed delay grids, T1-based relaxation, averaging and fit acceptance.

    uniform selects the native T1 linear delay sweep; False concentrates delays
    within the same start/stop window around the decay. Echo is unchanged.
    reps/rounds are native acquisition averages, not outer monitoring rounds.
    t1_guess_us/t2_guess_us are positive delay estimates in microseconds.
    sweep_factor times each guess gives its grid endpoint. T1 relax uses
    relax_factor times t1_guess_us. Echo relax uses the accepted T1 from
    this round, falling back to t1_guess_us. Both delays are floored at
    relax_min_us. Fits do not change grids, drive or readout.
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


class OvernightPoint(BaseModel):
    """Summarize one T1/Echo pair and its T1-based Echo relaxation.

    round_index is zero-based; scheduled_start/actual_start are aware UTC
    wall-clock times. actual_start remains None until the wait completes.
    flux is in the plan's explicit unit. t1_us/t2_us and their
    errors are accepted lifetimes and marginal uncertainties in microseconds,
    or None on rejection/native failure. t1_reason/t2_reason are None on
    acceptance, otherwise stopped for partial, a fit rejection, or the Schedule/
    expected fitting failure diagnosis. Unexpected exceptions stop the run
    after finally commits the collected fields. t1_file/t2_file are relative
    native filenames for each completed member, or None after native failure;
    rejected fits retain their raw acquisition filenames. echo_relax_delay_us
    is the Echo cfg relaxation in us; echo_t1_source is accepted_t1 for this
    round's accepted T1, otherwise guess. Both stay None until Echo cfg setup.
    gain_clamped records any truncated Echo pi/2 gain before/after.
    """

    model_config = ConfigDict(extra="forbid")

    round_index: int
    scheduled_start: datetime
    actual_start: datetime | None = None
    flux: float
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
    t1_file: str | None = None
    t2_file: str | None = None
    echo_relax_delay_us: float | None = None
    echo_t1_source: Literal["accepted_t1", "guess"] | None = None
    gain_clamped: list[GainClamp] = Field(default_factory=list)
    status: Literal["collecting", "completed", "skipped"] = "collecting"


@workflow("overnight", "Monitor fixed-flux T1 and Hahn-echo lifetime pairs.")
async def overnight[C](  # noqa: PLR0915  # tolerated: workflow flow reads top to bottom
    env: WorkflowEnv[C],
    plan: OvernightPlan,
    tun: Live[OvernightTunables],
) -> RunSummary:
    """Acquire T1 then zero-detune Echo at fixed flux, committing complete pairs.

    Echo relax uses this round's accepted T1, or t1_guess_us if unavailable,
    times relax_factor and floored at relax_min_us. T1 relax uses the same
    floor with t1_guess_us. Fits do not alter grids, pulses, or later rounds.
    No calibration or context read occurs. A native Failed
    outcome is recorded per member; the other member is still attempted.
    Completed members save before analysis. Partial records stopped without
    fitting; expected fitting failures record their reason without estimates.
    Neither fit/native failure skips the next member. Iteration cancellation
    skips it, while finally still commits the collected pair fields, then appends
    a sample to the captured entry when result_path is set. Append failures
    propagate without undoing the committed pair or earlier rows.
    Unexpected acquisition/fit/plot/save/device errors propagate. Each pair starts
    at max(now, previous actual start + interval), without catch-up bursts.
    Pause resumes the pending operation; it does not freeze wall time.
    Read timing and acceptance from tun.current. The trend retains scalar
    pairs, not Figures, and is not automatically saved.
    To revisit past points, start a new run with a sliced plan and seed local
    history from the previous run using the saved_run reader.
    """
    _ = env.soc
    _ = env.devices
    summary = SummaryAccumulator(plan.outer_rounds, plan.flux_unit)
    stage_env = StageEnv(plots=env.plots, snapshot=plan.snapshot, summary=summary)
    pulses = FixedPulses(pi=plan.qub_pulse, readout=plan.readout)
    qubit_role = plan.snapshot.roles.get("qubit")
    qubit_name = qubit_role if isinstance(qubit_role, str) else None
    last_actual_start: datetime | None = None
    trend_points: tuple[OvernightPoint, ...] = ()
    for round_index, iteration in zip(
        range(plan.outer_rounds), env.iterations(), strict=False
    ):
        env.pbar("overnight rounds", total=plan.outer_rounds).set_progress(round_index)
        draw_overnight_trend(
            env.plots, trend_points, start_interval_s=plan.start_interval_s
        )
        # Space actual pair starts without catching up after overruns.
        scheduled = datetime.now(UTC)
        if last_actual_start is not None:
            scheduled = max(
                scheduled,
                last_actual_start + timedelta(seconds=plan.start_interval_s),
            )
        point_record = OvernightPoint(
            round_index=round_index,
            scheduled_start=scheduled,
            flux=plan.flux,
        )
        try:
            await iteration.wait_until(scheduled)
            if iteration.cancelled:
                point_record.status = "skipped"
                continue
            actual = datetime.now(UTC)
            point_record.actual_start = actual
            # Set the fixed flux before acquiring either member.
            await set_flux(iteration, plan.flux_dev, plan.flux, unit=plan.flux_unit)
            if iteration.cancelled:
                point_record.status = "skipped"
                continue
            # Acquire and save T1 before fitting its scalar summary.
            current = tun.current
            t1 = await measure_t1(
                iteration,
                stage_env,
                pulses,
                T1Settings(
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
                ),
                points=plan.points,
                guess_us=current.t1_guess_us,
            )
            point_record.t1_status = t1.status
            point_record.t1_us = t1.lifetime_us
            point_record.t1_error_us = t1.error_us
            point_record.t1_reason = t1.reason
            point_record.t1_models = t1.models
            point_record.t1_file = t1.file
            if iteration.cancelled:
                point_record.status = "skipped"
                continue
            # Acquire zero-detune Echo even if the T1 member failed.
            current = tun.current
            point_record.echo_relax_delay_us, point_record.echo_t1_source = (
                echo_relaxation(
                    (point_record.t1_us,),
                    t1_guess_us=current.t1_guess_us,
                    relax_factor=current.relax_factor,
                    relax_min_us=current.relax_min_us,
                )
            )
            echo = await measure_echo(
                iteration,
                stage_env,
                pulses,
                EchoSettings(
                    reps=current.reps,
                    rounds=current.rounds,
                    acquire_retry=current.t2_acquire_retry,
                    earlystop_snr=current.t2_earlystop_snr,
                    sweep_factor=current.sweep_factor,
                    max_rel_err=current.max_rel_err,
                    max_fit_residual=current.max_fit_residual,
                    min_lifetime_points=current.min_lifetime_points,
                    max_model_window_factor=current.max_model_window_factor,
                ),
                points=plan.points,
                timing=EchoTiming(
                    guess_us=current.t2_guess_us,
                    relax_delay_us=point_record.echo_relax_delay_us,
                ),
            )
            point_record.t2_status = echo.status
            point_record.t2_us = echo.lifetime_us
            point_record.t2_error_us = echo.error_us
            point_record.t2_reason = echo.reason
            point_record.t2_models = echo.models
            point_record.t2_file = echo.file
            point_record.gain_clamped.extend(echo.gain_clamped)
            point_record.status = "skipped" if iteration.cancelled else "completed"
            last_actual_start = actual
            trend_points += (point_record,)
            draw_overnight_trend(
                env.plots, trend_points, start_interval_s=plan.start_interval_s
            )
        finally:
            iteration.commit(point_record)
            append_point_sample(env.identity.run_id, plan, point_record, iteration.dir)
            summary.points.append(overnight_summary_point(point_record, qubit_name))
    env.pbar("overnight rounds", total=plan.outer_rounds).set_progress(
        plan.outer_rounds
    )
    return summary.finish()
