"""Fixed-flux T1 followed by zero-detune Hahn echo, committed as one pair."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from zcu_tools.experiment.metadata import RunSnapshot
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import (
    Failed,
    Live,
    WorkflowEnv,
    workflow,
)
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg

from zcu_lab.v2.twotone.time_domain.t1.core import T1Cfg, T1Exp
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoCfg, T2EchoExp
from zcu_lab.workflows.common.analysis import DecayFit, analyze_t1, lifetime_rejection
from zcu_lab.workflows.common.devices import set_flux
from zcu_lab.workflows.common.relaxation import echo_relaxation
from zcu_lab.workflows.common.saving import save_native
from zcu_lab.workflows.overnight.analysis import analyze_t2echo
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
    reps: int = Field(default=100, ge=1, description="Native averages per acquisition")
    rounds: int = Field(default=2, ge=1, description="Native acquisition rounds")
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
        default=None,
        gt=0,
        description="Completed-round trace SNR threshold; None disables early stop",
    )
    t2_acquire_retry: int = Field(
        default=3, ge=0, description="Retries after a transient acquisition failure"
    )
    t2_earlystop_snr: float | None = Field(
        default=None,
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
    """

    model_config = ConfigDict(extra="forbid")

    round_index: int
    scheduled_start: datetime
    actual_start: datetime | None = None
    flux: float
    t1_us: float | None = None
    t1_error_us: float | None = None
    t1_reason: str | None = None
    t2_us: float | None = None
    t2_error_us: float | None = None
    t2_reason: str | None = None
    t1_file: str | None = None
    t2_file: str | None = None
    echo_relax_delay_us: float | None = None
    echo_t1_source: Literal["accepted_t1", "guess"] | None = None
    status: Literal["collecting", "completed", "skipped"] = "collecting"


def _summary(
    fit: DecayFit | Failed, tun: OvernightTunables, name: str
) -> tuple[float | None, float | None, str | None]:
    """Apply the shared scalar acceptance bounds to either lifetime fit.

    fit contains microsecond lifetime/error and residual quality; tun holds
    strict dimensionless bounds. name identifies rejected members. Expected
    fitting failures supply Failed(reason). Return accepted lifetime/error or
    None values with a diagnosis, without feedback.
    """
    if isinstance(fit, Failed):
        return None, None, fit.reason
    rejection = lifetime_rejection(
        fit.quality, fit.times, fit.lifetime_us, tun.min_lifetime_points
    )
    if rejection is not None:
        return None, None, f"{name} fit rejected: {rejection}"
    residual = fit.quality.normalized_residual_rms
    if not (
        isfinite(fit.lifetime_us)
        and fit.lifetime_us > 0
        and isfinite(fit.error_us)
        and fit.error_us >= 0
        and fit.error_us / fit.lifetime_us < tun.max_rel_err
        and residual is not None
        and residual < tun.max_fit_residual
    ):
        return None, None, f"{name} fit rejected"
    return fit.lifetime_us, fit.error_us, None


async def run_t1_stage(
    iteration: Scope,
    cfg: T1Cfg,
    point_record: OvernightPoint,
    *,
    snapshot: RunSnapshot,
    tun: Live[OvernightTunables],
) -> bool:
    """Save T1 before fitting and retain its summary on point cancellation."""
    t1_record = await iteration.run(T1Exp().run, cfg, name="T1")
    if not isinstance(t1_record, Failed):
        save_native(
            t1_record,
            iteration.dir / "t1.h5",
            spec=T1Exp.AXES_SPEC,
            snapshot=snapshot,
            identity=iteration.identity,
        )
        point_record.t1_file = "t1.h5"
    if isinstance(t1_record, Failed):
        summary = (None, None, t1_record.reason)
    elif t1_record.partial_result:
        summary = (None, None, "stopped")
    else:
        summary = _summary(analyze_t1(t1_record.value.result), tun.current, "T1")
    point_record.t1_us, point_record.t1_error_us, point_record.t1_reason = summary
    if iteration.cancelled:
        point_record.status = "skipped"
    return iteration.cancelled


async def run_echo_stage(
    iteration: Scope,
    cfg: T2EchoCfg,
    point_record: OvernightPoint,
    *,
    snapshot: RunSnapshot,
    tun: Live[OvernightTunables],
) -> bool:
    """Save T2 Echo before fitting and retain its summary on point cancellation."""
    t2_record = await iteration.run(T2EchoExp().run, cfg, name="Echo")
    if not isinstance(t2_record, Failed):
        save_native(
            t2_record,
            iteration.dir / "t2echo.h5",
            spec=T2EchoExp.AXES_SPEC,
            snapshot=snapshot,
            identity=iteration.identity,
        )
        point_record.t2_file = "t2echo.h5"
    if isinstance(t2_record, Failed):
        summary = (None, None, t2_record.reason)
    elif t2_record.partial_result:
        summary = (None, None, "stopped")
    else:
        summary = _summary(
            analyze_t2echo(t2_record.value.result), tun.current, "T2 Echo"
        )
    point_record.t2_us, point_record.t2_error_us, point_record.t2_reason = summary
    if iteration.cancelled:
        point_record.status = "skipped"
    return iteration.cancelled


@workflow("overnight", "Monitor fixed-flux T1 and Hahn-echo lifetime pairs.")
async def overnight[C](
    env: WorkflowEnv[C],
    plan: OvernightPlan,
    tun: Live[OvernightTunables],
) -> None:
    """Acquire T1 then zero-detune Echo at fixed flux, committing complete pairs.

    Echo relax uses this round's accepted T1, or t1_guess_us if unavailable,
    times relax_factor and floored at relax_min_us. T1 relax uses the same
    floor with t1_guess_us. Fits do not alter grids, pulses, or later rounds.
    No calibration or context read occurs. A native Failed
    outcome is recorded per member; the other member is still attempted.
    Completed members save before analysis. Partial records stopped without
    fitting; expected fitting failures record their reason without estimates.
    Neither fit/native failure skips the next member. Iteration cancellation
    skips it, while finally still commits the collected pair fields.
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
            t1_stop = tun.current.sweep_factor * tun.current.t1_guess_us
            t1_cfg = T1Cfg(
                uniform=tun.current.uniform,
                reps=tun.current.reps,
                rounds=tun.current.rounds,
                acquire_retry=tun.current.t1_acquire_retry,
                earlystop_snr=tun.current.t1_earlystop_snr,
                relax_delay=max(
                    tun.current.relax_min_us,
                    tun.current.relax_factor * tun.current.t1_guess_us,
                ),
                pi_pulse=plan.qub_pulse,
                readout=plan.readout,
                length=SweepCfg(
                    start=0.0,
                    stop=t1_stop,
                    expts=plan.points,
                    step=t1_stop / (plan.points - 1),
                ),
            )
            if await run_t1_stage(
                iteration, t1_cfg, point_record, snapshot=plan.snapshot, tun=tun
            ):
                continue
            # Acquire zero-detune Echo even if the T1 member failed.
            point_record.echo_relax_delay_us, point_record.echo_t1_source = (
                echo_relaxation(
                    (point_record.t1_us,),
                    t1_guess_us=tun.current.t1_guess_us,
                    relax_factor=tun.current.relax_factor,
                    relax_min_us=tun.current.relax_min_us,
                )
            )
            t2_stop = tun.current.sweep_factor * tun.current.t2_guess_us
            t2_cfg = T2EchoCfg(
                reps=tun.current.reps,
                rounds=tun.current.rounds,
                acquire_retry=tun.current.t2_acquire_retry,
                earlystop_snr=tun.current.t2_earlystop_snr,
                relax_delay=point_record.echo_relax_delay_us,
                pi_pulse=plan.qub_pulse,
                pi2_pulse=plan.qub_pulse.with_updates(gain=plan.qub_pulse.gain / 2),
                readout=plan.readout,
                length=SweepCfg(
                    start=0.0,
                    stop=t2_stop,
                    expts=plan.points,
                    step=t2_stop / (plan.points - 1),
                ),
            )
            await run_echo_stage(
                iteration, t2_cfg, point_record, snapshot=plan.snapshot, tun=tun
            )
            point_record.status = "skipped" if iteration.cancelled else "completed"
            last_actual_start = actual
            trend_points += (point_record,)
            draw_overnight_trend(
                env.plots, trend_points, start_interval_s=plan.start_interval_s
            )
        finally:
            iteration.commit(point_record)
    env.pbar("overnight rounds", total=plan.outer_rounds).set_progress(
        plan.outer_rounds
    )
