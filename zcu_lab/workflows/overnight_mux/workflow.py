"""Monitor simultaneous T1 and Hahn echo without flux changes or feedback."""

from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from zcu_tools.experiment.metadata import RunSnapshot
from zcu_tools.experiment.workflows import Live, WorkflowEnv, workflow
from zcu_tools.program.v2 import ResetCfg, SweepCfg
from zcu_tools.program.v2.modules import MuxReadoutCfg, PulseCfg

from zcu_lab.v2.mux.t1.core import MuxT1Cfg, MuxT1Exp, MuxT1ToneCfg
from zcu_lab.v2.mux.t2echo.core import MuxEchoToneCfg, MuxT2EchoCfg, MuxT2EchoExp
from zcu_lab.workflows.overnight_mux.analysis import record_echo, record_t1
from zcu_lab.workflows.overnight_mux.trend import draw_overnight_mux_trend


class OvernightMuxTone(MuxT1ToneCfg):
    """Original readout position: name, x180 drive and independent x90 pulse.

    Both pulses are required for participants by the native cores. Inactive
    positions may leave either pulse None. No pulse is inferred or calibrated.
    """

    pi2_pulse: PulseCfg | None = None


class OvernightMuxPlan(BaseModel):
    """Fixed mux pulses and finite monitoring schedule; no flux setting.

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


class OvernightMuxTunables(BaseModel):
    """Resolve lifetime drift using fixed timing, averaging and fit acceptance.

    uniform selects T1's linear delay grid; False concentrates delays within
    the same start/stop window. Echo always keeps its linear grid.
    reps/rounds are native acquisition averages, not outer monitoring rounds.
    t1_guess_us/t2_guess_us are positive longest-qubit delay estimates in
    microseconds. All participants share each member's delay grid.
    sweep_factor times each guess gives its grid endpoint; relax_factor times
    that guess gives its relaxation delay. No accepted fit changes any timing,
    drive or readout. max_rel_err and max_fit_residual are strict dimensionless
    upper bounds on error/lifetime and normalized residual RMS; rejected fits
    keep raw data but omit accepted lifetime/error scalars.
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
        default=5.0, gt=0, description="Relaxation delay / timing guess"
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
    t1_us: float | None = None
    t1_error_us: float | None = None
    t1_reason: str | None = None
    t2_us: float | None = None
    t2_error_us: float | None = None
    t2_reason: str | None = None


class OvernightMuxPoint(BaseModel):
    """One committed mux pair with UTC times and one row per participant.

    round_index is zero-based. actual_start is None until the wait completes.
    Native filenames are relative to the iteration directory, absent on native
    failure, retained on partial/rejected fits. status stays collecting on an
    unexpected exception, skipped on iteration cancellation, completed otherwise
    (including fit/native failures). Finally commits already collected fields.
    """

    model_config = ConfigDict(extra="forbid")
    round_index: int
    scheduled_start: datetime
    actual_start: datetime | None = None
    status: Literal["collecting", "completed", "skipped"] = "collecting"
    t1_file: str | None = None
    t2_file: str | None = None
    qubits: list[OvernightMuxQubit]


@workflow("overnight_mux", "Monitor simultaneous mux T1 and Hahn-echo pairs.")
async def overnight_mux[C](
    env: WorkflowEnv[C], plan: OvernightMuxPlan, tun: Live[OvernightMuxTunables]
) -> None:
    """Measure all participants' T1, then all participants' zero-detune Echo.

    Averaging, T1 sampling and acceptance follow overnight. Enter the longest
    qubit's T1/Echo guesses for their shared timing windows. No flux setup, context
    read, calibration or fit feedback occurs. Save Completed raw before fitting;
    partial results are not fitted. Native/fit failures do not skip Echo.
    Iteration cancellation skips remaining work; Stop/errors propagate after
    finally commits collected fields. Pause retains local progress. Pair starts
    follow max(now, previous completed pair's actual start + interval), matching
    overnight. Trends retain scalars only and are not automatically saved.
    """
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
            t1_stop = tun.current.sweep_factor * tun.current.t1_guess_us
            t1_cfg = MuxT1Cfg(
                uniform=tun.current.uniform,
                reps=tun.current.reps,
                rounds=tun.current.rounds,
                acquire_retry=tun.current.t1_acquire_retry,
                earlystop_snr=tun.current.t1_earlystop_snr,
                relax_delay=tun.current.relax_factor * tun.current.t1_guess_us,
                reset=plan.reset,
                readout=plan.readout,
                tones=[MuxT1ToneCfg(name=t.name, drive=t.drive) for t in plan.tones],
                participants=plan.participants,
                delays=SweepCfg(
                    start=0.0,
                    stop=t1_stop,
                    expts=plan.points,
                    step=t1_stop / (plan.points - 1),
                ),
            )
            t1_record = await iteration.run(MuxT1Exp().run, t1_cfg, name="Mux T1")
            record_t1(iteration, t1_record, point_record, plan, tun.current, env.plots)
            if iteration.cancelled:
                point_record.status = "skipped"
                continue
            t2_stop = tun.current.sweep_factor * tun.current.t2_guess_us
            t2_cfg = MuxT2EchoCfg(
                reps=tun.current.reps,
                rounds=tun.current.rounds,
                acquire_retry=tun.current.t2_acquire_retry,
                earlystop_snr=tun.current.t2_earlystop_snr,
                relax_delay=tun.current.relax_factor * tun.current.t2_guess_us,
                reset=plan.reset,
                readout=plan.readout,
                tones=[
                    MuxEchoToneCfg(
                        name=t.name, drive=t.pi2_pulse, pi_pulse=t.drive, detune=0.0
                    )
                    for t in plan.tones
                ],
                participants=plan.participants,
                delay=SweepCfg(
                    start=0.0,
                    stop=t2_stop,
                    expts=plan.points,
                    step=t2_stop / (plan.points - 1),
                ),
            )
            t2_record = await iteration.run(MuxT2EchoExp().run, t2_cfg, name="Mux Echo")
            record_echo(
                iteration, t2_record, point_record, plan, tun.current, env.plots
            )
            point_record.status = "skipped" if iteration.cancelled else "completed"
            last_actual_start = actual
            trend_points += (point_record,)
            draw_overnight_mux_trend(
                env.plots, trend_points, start_interval_s=plan.start_interval_s
            )
        finally:
            iteration.commit(point_record)
    env.pbar("overnight mux rounds", total=plan.outer_rounds).set_progress(
        plan.outer_rounds
    )
