"""Fixed-flux T1 followed by zero-detune Hahn echo, committed as one pair."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from zcu_tools.datafile import RunSnapshot
from zcu_tools.experiment.workflows import (
    Done,
    Failed,
    InitEnv,
    Next,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg

from zcu_lab.v2.twotone.time_domain.t1.core import T1Cfg, T1ModuleCfg, T1SweepCfg
from zcu_lab.v2.twotone.time_domain.t2echo.core import (
    T2EchoCfg,
    T2EchoModuleCfg,
    T2EchoSweepCfg,
)
from zcu_lab.workflows.acquire import run_t1, run_t2echo
from zcu_lab.workflows.analysis import DecayFit, analyze_t1, analyze_t2echo
from zcu_lab.workflows.devices import set_flux
from zcu_lab.workflows.saving import save_t1, save_t2echo


class OvernightPlan(BaseModel):
    """Fixed measurement choices, with no calibration or fit feedback.

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
    """Acquisition averaging, fixed timing and scalar summary acceptance.

    reps/rounds are native acquisition averages, not outer monitoring rounds.
    t1_guess_us/t2_guess_us are positive delay estimates in microseconds.
    sweep_factor times each guess gives its grid endpoint; relax_factor times
    that guess gives its relaxation delay. No accepted fit changes any timing,
    drive or readout. max_rel_err and max_fit_residual are strict dimensionless
    upper bounds on error/lifetime and normalized residual RMS; rejected fits
    keep raw data but omit accepted lifetime/error scalars.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    reps: int = Field(default=100, ge=1, description="Native averages per acquisition")
    rounds: int = Field(default=2, ge=1, description="Native acquisition rounds")
    t1_guess_us: float = Field(
        default=20.0, gt=0, description="Fixed T1 timing guess, microseconds"
    )
    t2_guess_us: float = Field(
        default=10.0, gt=0, description="Fixed Echo timing guess, microseconds"
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


@dataclass
class OvernightState:
    """Committed pair cursor and wall-clock timing, copied by Engine.

    round_index is the next zero-based pair index. last_actual_start is the
    preceding committed pair's aware UTC start, or None before the first pair.
    Uncommitted changes disappear on Pause/Stop; Resume repeats that pair.
    """

    round_index: int = 0
    last_actual_start: datetime | None = None


@dataclass(frozen=True)
class OvernightPoint:
    """One committed pair's scalar evidence, never a trace or calibration.

    round_index is zero-based; scheduled_start/actual_start are aware UTC
    wall-clock times. flux is in the plan's explicit unit. t1_us/t2_us and their
    errors are accepted lifetimes and marginal uncertainties in microseconds,
    or None on rejection/native failure. t1_reason/t2_reason are None on
    acceptance, otherwise the rejection or Schedule failure diagnosis. Exceptions
    stop the run without committing this point. t1_file/t2_file are relative
    native filenames for each completed member, or None after native failure;
    rejected fits retain their raw acquisition filenames.
    """

    round_index: int
    scheduled_start: datetime
    actual_start: datetime
    flux: float
    t1_us: float | None
    t1_error_us: float | None
    t1_reason: str | None
    t2_us: float | None
    t2_error_us: float | None
    t2_reason: str | None
    t1_file: str | None
    t2_file: str | None


def init_overnight[C](_env: InitEnv[C], _plan: OvernightPlan) -> OvernightState:
    """Initialize a fresh pair cursor without hardware, timing or capture reads."""
    return OvernightState()


def _summary(
    fit: DecayFit, tun: OvernightTunables, name: str
) -> tuple[float | None, float | None, str | None]:
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


def overnight[C](
    env: WorkflowEnv[C],
    plan: OvernightPlan,
    tun: OvernightTunables,
    state: OvernightState,
) -> Step[OvernightState, OvernightPoint]:
    """Acquire T1 then Echo at fixed flux, publishing a complete pair via Next.

    Does not read or produce context; it may be any value, including None.
    All hardware/wait effects use Engine cancellation. A native Failed outcome
    is recorded per member; the other member is still attempted. Unexpected
    acquisition/analysis/plot/saver/device errors propagate and stop the run.
    Pause/Stop discard uncommitted pairs while retaining workflow-written native files.
    Wait until max(now, previous committed actual start + interval). Overruns
    start immediately, without catching up to an original-start grid. Pause
    does not freeze wall time; Resume repeats the discarded pair under the same
    rule. Progress counts committed pairs only. Drive, readout and timing are fixed;
    fit acceptance only affects scalar summaries, never the next native cfg.
    """
    _ = env.soc
    _ = env.devices
    env.pbar("overnight rounds", total=plan.outer_rounds).set_progress(
        state.round_index
    )
    if state.round_index == plan.outer_rounds:
        return Done()
    scheduled = datetime.now(UTC)
    if state.last_actual_start is not None:
        scheduled = max(
            scheduled,
            state.last_actual_start + timedelta(seconds=plan.start_interval_s),
        )
    yield from env.wait_until(scheduled)
    actual = datetime.now(UTC)
    yield from set_flux(env, plan.flux_dev, plan.flux, unit=plan.flux_unit)
    t1_stop = tun.sweep_factor * tun.t1_guess_us
    t1_cfg = T1Cfg(
        reps=tun.reps,
        rounds=tun.rounds,
        relax_delay=tun.relax_factor * tun.t1_guess_us,
        modules=T1ModuleCfg(pi_pulse=plan.qub_pulse, readout=plan.readout),
        sweep=T1SweepCfg(
            length=SweepCfg(
                start=0.0,
                stop=t1_stop,
                expts=plan.points,
                step=t1_stop / (plan.points - 1),
            )
        ),
    )
    t1 = yield from env.run(run_t1, t1_cfg)
    t1_file = None
    if not isinstance(t1, Failed):
        save_t1(t1, env.iteration_dir / "t1.h5", snapshot=plan.snapshot)
        t1_file = "t1.h5"
    t1_summary = (
        (None, None, t1.reason)
        if isinstance(t1, Failed)
        else _summary(analyze_t1(t1.result), tun, "T1")
    )
    t2_stop = tun.sweep_factor * tun.t2_guess_us
    t2_cfg = T2EchoCfg(
        reps=tun.reps,
        rounds=tun.rounds,
        relax_delay=tun.relax_factor * tun.t2_guess_us,
        modules=T2EchoModuleCfg(
            pi_pulse=plan.qub_pulse,
            pi2_pulse=plan.qub_pulse.with_updates(gain=plan.qub_pulse.gain / 2),
            readout=plan.readout,
        ),
        sweep=T2EchoSweepCfg(
            length=SweepCfg(
                start=0.0,
                stop=t2_stop,
                expts=plan.points,
                step=t2_stop / (plan.points - 1),
            )
        ),
    )
    t2 = yield from env.run(run_t2echo, t2_cfg)
    t2_file = None
    if not isinstance(t2, Failed):
        save_t2echo(t2, env.iteration_dir / "t2echo.h5", snapshot=plan.snapshot)
        t2_file = "t2echo.h5"
    t2_summary = (
        (None, None, t2.reason)
        if isinstance(t2, Failed)
        else _summary(analyze_t2echo(t2.result), tun, "T2 Echo")
    )
    point = OvernightPoint(
        state.round_index,
        scheduled,
        actual,
        plan.flux,
        *t1_summary,
        *t2_summary,
        t1_file,
        t2_file,
    )
    state.round_index += 1
    state.last_actual_start = actual
    return Next(point, state)


# The decorator's return fixes invariant C, losing the function's parametric context.
# Declare the original function in place instead of assigning that return.
workflow(
    name="overnight",
    plan=OvernightPlan,
    tunables=OvernightTunables,
    init=init_overnight,
)(overnight)
