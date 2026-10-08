"""Native T1 flux-dependence plan, state and complete cfg generation.

The workflow owns generation, calibration policy and accepted scalar history.
Engine owns copied inputs, cancellation, exact destinations and publication.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from matplotlib.axes import Axes
from pydantic import BaseModel, ConfigDict, Field
from zcu_tools.datafile import RunSnapshot
from zcu_tools.experiment.workflows import (
    Done,
    Failed,
    InitEnv,
    Live1D,
    Next,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg

from zcu_lab.v2.twotone.time_domain.t1.core import T1Cfg
from zcu_lab.workflows.acquire import run_t1
from zcu_lab.workflows.analysis import analyze_t1
from zcu_lab.workflows.calibration import (
    CalCarry,
    Calibrated,
    CalibrationFailed,
    CalibrationSavers,
    CalibrationTunables,
    apply_readout_calibration,
    calibrate,
)
from zcu_lab.workflows.saving import save_frequency, save_rabi, save_readout, save_t1


class T1Plan(BaseModel):
    """Fixed setpoints, two resolved module choices and historical capture.

    flux_dev names the connected device. fluxes retain absolute device units
    and order; points >= 5 controls each uniform delay grid. qub_pulse/readout
    are the shared host-resolved drive and pulse-backed readout selections.
    Drive routing/waveform/gain are fixed; runtime overwrites frequency/pi
    duration and applies accepted readout frequency/gain on shared routing.
    snapshot retains the captured entry identity, work point and roles for
    each acquisition; it does not choose destinations or read live state.
    No complete native cfg, averaging, timing policy or sweep lives in plan.
    Engine copies the plan before runtime constructs per-acquisition cfgs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    flux_dev: str = Field(min_length=1, description="Connected flux device name")
    fluxes: tuple[float, ...] = Field(
        description="Ordered absolute setpoints in native device units"
    )
    points: int = Field(default=51, ge=5, description="Uniform T1 delay samples")
    qub_pulse: PulseCfg = Field(
        description="Resolved shared spectroscopy/Rabi/pi drive"
    )
    readout: PulseReadoutCfg = Field(description="Resolved shared pulse-backed readout")
    snapshot: RunSnapshot = Field(description="Host-captured historical entry state")


class T1Tunables(BaseModel):
    """Captured calibration, T1 generation and acceptance knobs.

    cal controls calibration only. reps/rounds control T1 averaging.
    guess_us seeds lifetime history; sweep_factor/relax_factor multiply the
    current lifetime guess into delay-grid end/relax delay in microseconds.
    max_rel_err and max_fit_residual are strict upper bounds for error/lifetime
    and normalized residual RMS. Defaults are provisional, confirmed at CP-B/C.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    cal: CalibrationTunables = Field(default_factory=CalibrationTunables)
    reps: int = Field(default=100, ge=1, description="T1 averages")
    rounds: int = Field(default=10, ge=1, description="T1 averaging rounds")
    guess_us: float = Field(default=20.0, gt=0, description="Initial lifetime in us")
    sweep_factor: float = Field(
        default=5.0, gt=0, description="Sweep end divided by lifetime guess"
    )
    relax_factor: float = Field(
        default=3.0, gt=0, description="Relax delay divided by lifetime guess"
    )
    max_rel_err: float = Field(
        default=0.3, gt=0, le=1, description="Strict T1 error/lifetime upper bound"
    )
    max_fit_residual: float = Field(
        default=0.2, gt=0, le=1, description="Strict normalized residual RMS bound"
    )


@dataclass
class T1State:
    """Committed data for a T1 flux-dependence step.

    remaining contains unattempted absolute device setpoints in order.
    cal is the scalar calibration seed and all-success carry.
    t1_by_flux contains ordered (native setpoint, accepted lifetime in us) pairs.
    """

    remaining: list[float]
    cal: CalCarry
    t1_by_flux: list[tuple[float, float]] = field(default_factory=list)


def init_t1(env: InitEnv[Calibrated], plan: T1Plan) -> T1State:
    """Create fresh scalar carry/history and copy the plan's remaining setpoints.

    env.context must supply the deeply read-only Calibrated seed; absent
    context raises MissingCapability. No hardware, files or plots are touched.
    """
    return T1State(list(plan.fluxes), CalCarry(env.context))


def t1_cfg(plan: T1Plan, tun: T1Tunables, cal: Calibrated, lifetime_us: float) -> T1Cfg:
    """Build a detached complete uniform native T1 cfg, without acquisition.

    lifetime_us is the positive finite lifetime guess in microseconds;
    invalid values raise ValueError. Delay ranges from zero through
    tun.sweep_factor * lifetime_us, with plan.points samples. Relax delay is
    tun.relax_factor * lifetime_us. reps/rounds come from tun.
    Pi drive uses plan.qub_pulse waveform/gain and cal's accepted frequency
    and duration. Readout uses cal's frequency/gain on plan.readout routing.
    Native initial_delay/dev defaults remain owned by the native cfg model;
    no reset or extra module role is selected. Flux is set by the workflow,
    never patched into a cfg device mapping.
    Inputs are unchanged; native cfg construction/set_param errors propagate.
    """
    if not math.isfinite(lifetime_us) or lifetime_us <= 0:
        raise ValueError("T1 lifetime guess must be positive and finite")
    cfg = plan.t1.model_copy(deep=True)
    cfg.reps, cfg.rounds = tun.reps, tun.rounds
    cfg.relax_delay = tun.relax_factor * lifetime_us
    cfg.modules.pi_pulse = plan.calibration.rabi.modules.qub_pulse.model_copy(deep=True)
    cfg.modules.pi_pulse.set_param("freq", cal.f01_mhz)
    cfg.modules.pi_pulse.set_param("length", cal.pi_length_us)
    apply_readout_calibration(cfg.modules.readout, cal)
    stop = tun.sweep_factor * lifetime_us
    cfg.sweep.length = SweepCfg(
        start=0.0, stop=stop, expts=plan.points, step=stop / (plan.points - 1)
    )
    return cfg


@dataclass(frozen=True)
class T1Point:
    """Committed outcome of one absolute flux setpoint.

    index is zero-based in plan.fluxes. flux uses the connected device's units.
    calibration is the accepted scalar calibration or its stage/reason failure.
    lifetime_us and error_us are accepted lifetime/error in microseconds, or both
    None when acquisition/fit failed. reason is None on success, otherwise the
    calibration, Schedule or fit-rejection reason. Raw traces live in native files.
    """

    index: int
    flux: float
    calibration: Calibrated | CalibrationFailed
    lifetime_us: float | None
    error_us: float | None
    reason: str | None


def _save_plot(ax: Axes, destination: Path) -> None:
    figure = ax.get_figure(root=True)
    if figure is None:
        raise ValueError("Cannot save axes without a root figure")
    figure.savefig(destination)


@workflow(
    "T1_fluxdep",
    plan=T1Plan,
    tunables=T1Tunables,
    state=T1State,
    record=T1Point,
    init=init_t1,
    requires=("soc", "devices", "context"),
)
def t1_fluxdep(
    env: WorkflowEnv[Calibrated],
    plan: T1Plan,
    tun: T1Tunables,
    state: T1State,
) -> Step[T1State, T1Point]:
    """Attempt the next setpoint with native acquisitions and exact-path saving.

    Engine supplies detached state and captured tunables. Calibration updates
    carry only on full success. Failed calibration/acquisition or rejected T1
    fit commits a point with its reason, without accepted lifetime history.
    Accepted T1 appends lifetime/error and saves its decay plot. The final step
    saves a summary built from committed lifetime history and returns Done.
    Pause/stop discards the current step through Engine. Native adapter, fitting,
    plotting and saver errors propagate; already published files remain.
    """
    index = len(plan.fluxes) - len(state.remaining)
    env.pbar("flux points", total=len(plan.fluxes)).set_progress(index)
    summary = env.axes("t1/vs_flux")
    summary.cla()
    if state.t1_by_flux:
        fluxes, lifetimes = zip(*state.t1_by_flux, strict=True)
        summary.scatter(fluxes, lifetimes)
    summary.set(xlabel="flux setpoint", ylabel="T1 (us)")
    if not state.remaining:
        _save_plot(summary, env.iter_dir / "summary.png")
        return Done()

    flux = state.remaining.pop(0)
    yield from env.set_device(plan.flux_dev, flux)
    cal = yield from calibrate(
        env,
        plan.qub_pulse,
        plan.readout,
        tun.cal,
        flux,
        state.cal,
        savers=CalibrationSavers(
            partial(save_frequency, snapshot=plan.snapshot),
            partial(save_rabi, snapshot=plan.snapshot),
            partial(save_readout, snapshot=plan.snapshot),
        ),
    )
    if isinstance(cal, CalibrationFailed):
        return Next(T1Point(index, flux, cal, None, None, cal.reason), state)

    guess = tun.guess_us
    for _, lifetime in state.t1_by_flux:
        guess = 0.5 * lifetime + 0.5 * guess
    ax = env.axes("t1/decay")
    ax.cla()
    (line,) = ax.plot([], [])
    outcome = yield from env.run(
        run_t1,
        t1_cfg(plan, tun, cal, guess),
        save=partial(save_t1, snapshot=plan.snapshot),
        live=Live1D(line, y=lambda signals: signals.real),
    )
    if isinstance(outcome, Failed):
        return Next(T1Point(index, flux, cal, None, None, outcome.reason), state)
    fit = analyze_t1(outcome.result)
    residual = fit.quality.normalized_residual_rms
    if not (
        math.isfinite(fit.lifetime_us)
        and fit.lifetime_us > 0
        and math.isfinite(fit.error_us)
        and fit.error_us >= 0
        and fit.error_us / fit.lifetime_us < tun.max_rel_err
        and residual is not None
        and residual < tun.max_fit_residual
    ):
        return Next(T1Point(index, flux, cal, None, None, "T1 fit rejected"), state)
    ax.plot(outcome.result.times, fit.fitted, label="fit")
    ax.set_title(f"T1 = {fit.lifetime_us:.2f} us")
    _save_plot(ax, env.iter_dir / "decay.png")
    state.t1_by_flux.append((flux, fit.lifetime_us))
    return Next(T1Point(index, flux, cal, fit.lifetime_us, fit.error_us, None), state)
