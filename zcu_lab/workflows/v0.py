"""Phase-0 offline workflow demonstrations, not a hardware measurement entry.

Calibration values and traces are fake deterministic values, not physical policy.
The JSON savers only exercise the exact-path saver seam, even for .h5 paths.
They are not the official/native data format and are replaced during phase 1.
Import declares functions only; it does not register workflows or start a run.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from zcu_tools.experiment.workflows import (
    Completed,
    Done,
    Effect,
    Failed,
    InitEnv,
    Live1D,
    Next,
    Run,
    Step,
    WorkflowEnv,
    WorkflowRegistry,
    workflow,
)


@dataclass(frozen=True)
class DemoContext:
    """Fake initial calibration: f01/readout frequency in MHz, pi length in us."""

    f01_mhz: float
    pi_length_us: float
    readout_freq_mhz: float


class FluxPlan(BaseModel):
    """Ordered device setpoints and fixed per-curve sample count."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    flux_dev: str = Field(min_length=1, description="Connected flux device name")
    fluxes: tuple[float, ...] = Field(
        description="Ordered absolute setpoints in device units"
    )
    points: int = Field(
        default=51, ge=3, description="Samples per decay curve for this run"
    )


class CalTunables(BaseModel):
    """Calibration knobs captured once per workflow step.

    qf_reps/rabi_reps/ro_reps are scan averages. qf_widen contains positive
    span multipliers; span_mhz is the base frequency span. min_snr is the
    acceptance threshold. ro_every counts successful calibrations between
    readout scans. The first successful calibration always scans readout.
    """

    model_config = ConfigDict(extra="forbid")
    qf_reps: int = Field(
        default=100, ge=1, le=10000, description="Frequency scan averages"
    )
    rabi_reps: int = Field(
        default=100, ge=1, le=10000, description="Rabi scan averages"
    )
    ro_reps: int = Field(
        default=100, ge=1, le=10000, description="Readout scan averages"
    )
    qf_widen: tuple[Annotated[float, Field(gt=0)], ...] = Field(
        default=(1.0, 2.0, 4.0), min_length=1
    )
    span_mhz: float = Field(default=30.0, ge=1.0, le=500.0)
    min_snr: float = Field(default=5.0, ge=1.0, le=50.0)
    ro_every: int = Field(default=5, ge=1, le=1000)


class DecayKnobs(BaseModel):
    """reps and rounds control acquire averaging, not workflow points or rounds."""

    model_config = ConfigDict(extra="forbid")
    reps: int = Field(default=100, ge=1, le=10000)
    rounds: int = Field(default=10, ge=1, le=1000)


class T1Tunables(BaseModel):
    """T1 generation and acceptance knobs.

    cal controls calibration; t1 controls acquire averaging. guess_us seeds
    the lifetime estimate. sweep_factor/relax_factor scale that estimate into
    timing. max_rel_err is the strict upper bound on relative fit uncertainty.
    """

    model_config = ConfigDict(extra="forbid")
    cal: CalTunables = Field(default_factory=CalTunables)
    t1: DecayKnobs = Field(default_factory=DecayKnobs)
    guess_us: float = Field(default=20.0, ge=1.0, le=1000.0)
    sweep_factor: float = Field(default=5.0, ge=2.0, le=10.0)
    relax_factor: float = Field(default=3.0, ge=1.0, le=10.0)
    max_rel_err: float = Field(default=0.3, ge=0.01, le=1.0)


class T2EchoTunables(BaseModel):
    """Echo generation and acceptance knobs.

    cal controls calibration; t2echo controls acquire averaging. guess_us seeds
    the lifetime estimate. sweep_factor/relax_factor scale that estimate into
    timing. max_rel_err is the strict upper bound on relative fit uncertainty.
    """

    model_config = ConfigDict(extra="forbid")
    cal: CalTunables = Field(default_factory=CalTunables)
    t2echo: DecayKnobs = Field(default_factory=DecayKnobs)
    guess_us: float = Field(default=30.0, ge=1.0, le=1000.0)
    sweep_factor: float = Field(default=5.0, ge=2.0, le=10.0)
    relax_factor: float = Field(default=3.0, ge=1.0, le=10.0)
    max_rel_err: float = Field(default=0.3, ge=0.01, le=1.0)


@dataclass(frozen=True)
class Calibrated:
    """f01_mhz/readout_freq_mhz are frequencies; pi_length_us is pulse duration."""

    f01_mhz: float
    pi_length_us: float
    readout_freq_mhz: float


@dataclass(frozen=True)
class CalibrationFailed:
    """stage names the rejected scan; reason explains its calibration failure."""

    stage: Literal["qubit_freq", "pi_pulse", "readout"]
    reason: str


type Calibration = Calibrated | CalibrationFailed


@dataclass
class CalCarry:
    """Committed calibration history, updated only after full success.

    seed holds the initial values; last is the latest accepted calibration or
    None. f01_by_flux contains ordered (native setpoint, MHz) pairs.
    n_calibrated counts successful calibration calls for readout cadence.
    """

    seed: Calibrated
    last: Calibrated | None = None
    f01_by_flux: list[tuple[float, float]] = field(default_factory=list)
    n_calibrated: int = 0


def initial_carry(env: InitEnv[DemoContext]) -> CalCarry:
    """Copy the host's initial calibration into workflow-owned pure data."""
    ctx = env.context
    return CalCarry(Calibrated(ctx.f01_mhz, ctx.pi_length_us, ctx.readout_freq_mhz))


@dataclass(frozen=True)
class CalCfg:
    """Fake calibration input for qubit_freq, pi_pulse, or readout.

    stage selects the scan; center/span are in MHz for frequency/readout and
    us for pi_pulse. points is the sample count; reps is acquire averaging.
    """

    stage: Literal["qubit_freq", "pi_pulse", "readout"]
    center: float
    span: float
    points: int
    reps: int


@dataclass(frozen=True)
class CalResult:
    """Deterministic calibration output.

    axis and complex signals are matching 1-D samples. value is the accepted
    center in the cfg stage's units; snr is the dimensionless quality estimate.
    """

    axis: NDArray[np.float64]
    signals: NDArray[np.complex128]
    value: float
    snr: float


@dataclass(frozen=True)
class DecayCfg:
    """Fake decay input; kind selects t1 or t2echo and cal holds calibration.

    points is the sample count. stop_us is the final delay and relax_delay_us
    is the relaxation wait. reps/rounds describe acquire averaging, not outer
    workflow iterations. All timing is in microseconds.
    """

    kind: Literal["t1", "t2echo"]
    cal: Calibrated
    points: int
    stop_us: float
    relax_delay_us: float
    reps: int
    rounds: int


@dataclass(frozen=True)
class DecayResult:
    """times_us and signals are matching 1-D delay and complex sample arrays."""

    times_us: NDArray[np.float64]
    signals: NDArray[np.complex128]


def fake_calibration(run: Run[CalCfg]) -> CalResult:
    """Fill a trace without hardware; preserve the real buffer/schedule seam."""
    cfg = run.cfg
    axis = np.linspace(cfg.center - cfg.span / 2, cfg.center + cfg.span / 2, cfg.points)
    buffer = run.buffer((cfg.points,), axes=(axis,))
    with run.schedule(buffer) as sched:
        for x, step in sched.scan("sample", axis):
            value = np.exp(-(((x - cfg.center) / max(cfg.span / 8, 1e-9)) ** 2))
            step.set_data(np.complex128(value), flush=True)
    return CalResult(axis, buffer.array.copy(), cfg.center, 20.0)


def run_qubit_freq(run: Run[CalCfg]) -> CalResult:
    """Offline substitute for a qubit-frequency scan."""
    return fake_calibration(run)


def run_len_rabi(run: Run[CalCfg]) -> CalResult:
    """Offline substitute for a Rabi scan."""
    return fake_calibration(run)


def run_ro_landscape(run: Run[CalCfg]) -> CalResult:
    """Offline substitute for readout optimization."""
    return fake_calibration(run)


def fake_decay(run: Run[DecayCfg], lifetime_us: float) -> DecayResult:
    """Fill a complex decay trace; Schedule supplies cancellation checks."""
    cfg = run.cfg
    times = np.linspace(0.5, cfg.stop_us, cfg.points)
    buffer = run.buffer((cfg.points,), axes=(times,))
    with run.schedule(buffer) as sched:
        for delay, step in sched.scan("delay", times):
            step.set_data(np.complex128(np.exp(-delay / lifetime_us)), flush=True)
    return DecayResult(times, buffer.array.copy())


def run_t1(run: Run[DecayCfg]) -> DecayResult:
    """Offline T1 acquire with a known 20 us lifetime."""
    return fake_decay(run, 20.0)


def run_t2echo(run: Run[DecayCfg]) -> DecayResult:
    """Offline echo acquire with a known 30 us lifetime."""
    return fake_decay(run, 30.0)


def save_cal(source: Completed[CalCfg, CalResult], destination: Path) -> None:
    """Write a diagnostic fake payload to the engine's exact temporary path."""
    result = source.result
    payload = {
        "cfg": asdict(source.cfg),
        "axis": result.axis.tolist(),
        "real": result.signals.real.tolist(),
        "imag": result.signals.imag.tolist(),
        "value": result.value,
        "snr": result.snr,
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, allow_nan=False)


def save_decay(source: Completed[DecayCfg, DecayResult], destination: Path) -> None:
    """Fake persistence adapter; not the phase-1 native HDF5 writer."""
    result = source.result
    payload = {
        "cfg": asdict(source.cfg),
        "times_us": result.times_us.tolist(),
        "real": result.signals.real.tolist(),
        "imag": result.signals.imag.tolist(),
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, allow_nan=False)


@dataclass(frozen=True)
class DecayValue:
    """value_us/err_us hold lifetime/uncertainty; snr is a dimensionless estimate."""

    value_us: float
    err_us: float
    snr: float

    @property
    def rel_err(self) -> float:
        """Return uncertainty divided by the positive fitted lifetime."""
        return self.err_us / self.value_us


def analyze_decay(_cfg: DecayCfg, result: DecayResult) -> DecayValue | None:
    """Offline analytic fit; return None for a non-finite or non-decaying trace."""
    y = result.signals.real
    if not np.all(np.isfinite(y)) or np.any(y <= 0):
        return None
    slope = np.polyfit(result.times_us, np.log(y), 1)[0]
    if not np.isfinite(slope) or slope >= 0:
        return None
    lifetime = float(-1.0 / slope)
    return DecayValue(lifetime, lifetime * 0.01, 20.0)


def smooth_last(values: list[float], default: float, *, alpha: float = 0.5) -> float:
    """Compute EWMA from pure history; empty history returns the explicit default."""
    estimate = default
    for value in values:
        estimate = alpha * value + (1 - alpha) * estimate
    return estimate


def save_plot(ax: Axes, destination: Path) -> None:
    """Save the axes' root figure to the exact destination, propagating I/O errors."""
    figure = ax.get_figure(root=True)
    if figure is None:
        raise ValueError("Cannot save axes without a root figure")
    figure.savefig(destination)


def redraw(
    env: WorkflowEnv[DemoContext], name: str, points: list[tuple[float, float]]
) -> None:
    """Rebuild a scatter plot from committed workflow data."""
    ax = env.axes(name)
    ax.cla()
    if points:
        x, y = zip(*points, strict=True)
        ax.scatter(x, y)


def calibrate(
    env: WorkflowEnv[DemoContext],
    tun: CalTunables,
    flux: float,
    carry: CalCarry,
) -> Generator[Effect, None, Calibration]:
    """Return Calibration after yielding effects, not a workflow Next.

    On full success, mutate carry.last/history/count. Failure leaves carry unchanged.
    """
    prior = carry.last if carry.last is not None else carry.seed
    hint = smooth_last([v for _, v in carry.f01_by_flux], prior.f01_mhz)
    stages = env.pbar("calibration", total=3)
    ax = env.axes("calibration/qubit_freq")
    stages.set_progress(0)
    f01 = None
    for widen in tun.qf_widen:
        ax.cla()
        ax.axvline(hint, linestyle="--", label="expected")
        ax.axvline(prior.f01_mhz, linestyle=":", label="previous")
        (line,) = ax.plot([], [])
        outcome = yield from env.run(
            run_qubit_freq,
            CalCfg("qubit_freq", hint, tun.span_mhz * widen, 101, tun.qf_reps),
            save=save_cal,
            live=Live1D(line, y=lambda signals: signals.real),
        )
        if isinstance(outcome, Failed):
            continue
        if outcome.result.snr >= tun.min_snr:
            f01 = outcome.result.value
            break
    if f01 is None:
        return CalibrationFailed("qubit_freq", "no accepted frequency fit")

    stages.set_progress(1)
    outcome = yield from env.run(
        run_len_rabi,
        CalCfg("pi_pulse", prior.pi_length_us, prior.pi_length_us, 51, tun.rabi_reps),
        save=save_cal,
    )
    if isinstance(outcome, Failed):
        return CalibrationFailed("pi_pulse", outcome.reason)
    if outcome.result.snr < tun.min_snr:
        return CalibrationFailed("pi_pulse", "SNR below threshold")
    pi_length = outcome.result.value

    stages.set_progress(2)
    readout = prior.readout_freq_mhz
    if carry.last is None or carry.n_calibrated % tun.ro_every == 0:
        outcome = yield from env.run(
            run_ro_landscape,
            CalCfg("readout", readout, 10.0, 51, tun.ro_reps),
            save=save_cal,
        )
        if isinstance(outcome, Failed):
            return CalibrationFailed("readout", outcome.reason)
        if outcome.result.snr < tun.min_snr:
            return CalibrationFailed("readout", "SNR below threshold")
        readout = outcome.result.value

    cal = Calibrated(f01, pi_length, readout)
    carry.last = cal
    carry.f01_by_flux.append((flux, f01))
    carry.n_calibrated += 1
    stages.set_progress(3)
    return cal


@dataclass(frozen=True)
class T1Point:
    """One attempted T1 point.

    index is zero-based; flux is the native setpoint. cal records calibration
    success/failure. t1 is an accepted estimate or None; reason explains a
    missing estimate and is None on success.
    """

    index: int
    flux: float
    cal: Calibration
    t1: DecayValue | None
    reason: str | None


@dataclass
class T1State:
    """T1 loop state: remaining native setpoints and calibration carry in cal.

    t1_by_flux contains ordered (setpoint, accepted lifetime in us) pairs.
    """

    remaining: list[float]
    cal: CalCarry
    t1_by_flux: list[tuple[float, float]]


def init_t1(env: InitEnv[DemoContext], plan: FluxPlan) -> T1State:
    """Create detached T1 state from the fixed plan and initial host context."""
    return T1State(list(plan.fluxes), initial_carry(env), [])


def t1_cfg(plan: FluxPlan, tun: T1Tunables, cal: Calibrated, guess: float) -> DecayCfg:
    """Build complete T1 cfg using a positive lifetime guess in microseconds."""
    return DecayCfg(
        "t1",
        cal,
        plan.points,
        tun.sweep_factor * guess,
        tun.relax_factor * guess,
        tun.t1.reps,
        tun.t1.rounds,
    )


@workflow(
    "T1_fluxdep",
    plan=FluxPlan,
    tunables=T1Tunables,
    state=T1State,
    record=T1Point,
    init=init_t1,
    requires=("devices", "context"),
)
def t1_fluxdep(
    env: WorkflowEnv[DemoContext],
    plan: FluxPlan,
    tun: T1Tunables,
    state: T1State,
) -> Step[T1State, T1Point]:
    """Attempt the next flux point; commit missing fits with an explicit reason.

    The engine supplies detached state and captured tunables. Effects acquire
    and save offline traces; calibration updates carry only after full success.
    After all points, save a summary figure and return Done without a record.
    Adapter, plotting, analysis, and saver errors propagate to the engine.
    """
    index = len(plan.fluxes) - len(state.remaining)
    env.pbar("flux points", total=len(plan.fluxes)).set_progress(index)
    redraw(env, "t1/vs_flux", state.t1_by_flux)
    if not state.remaining:
        save_plot(env.axes("t1/vs_flux"), env.iter_dir / "summary.png")
        return Done()
    flux = state.remaining.pop(0)
    yield from env.set_device(plan.flux_dev, flux)
    cal = yield from calibrate(env, tun.cal, flux, state.cal)
    if isinstance(cal, CalibrationFailed):
        return Next(T1Point(index, flux, cal, None, cal.reason), state)

    guess = smooth_last([v for _, v in state.t1_by_flux], tun.guess_us)
    ax = env.axes("t1/decay")
    ax.cla()
    (line,) = ax.plot([], [])
    outcome = yield from env.run(
        run_t1,
        t1_cfg(plan, tun, cal, guess),
        save=save_decay,
        live=Live1D(line, y=lambda signals: signals.real),
    )
    if isinstance(outcome, Failed):
        return Next(T1Point(index, flux, cal, None, outcome.reason), state)
    fit = analyze_decay(outcome.cfg, outcome.result)
    if fit is None or fit.rel_err >= tun.max_rel_err:
        return Next(T1Point(index, flux, cal, None, "fit rejected"), state)

    ax.plot(outcome.result.times_us, np.exp(-outcome.result.times_us / fit.value_us))
    ax.set_title(f"T1 = {fit.value_us:.2f} us")
    save_plot(ax, env.iter_dir / "decay.png")
    state.t1_by_flux.append((flux, fit.value_us))
    return Next(T1Point(index, flux, cal, fit, None), state)


@dataclass(frozen=True)
class T2EchoPoint:
    """One attempted echo point.

    index is zero-based; flux is the native setpoint. cal records calibration
    success/failure. t2echo is accepted or None; reason explains missing values
    and is None on success.
    """

    index: int
    flux: float
    cal: Calibration
    t2echo: DecayValue | None
    reason: str | None


@dataclass
class T2EchoState:
    """Echo loop state: remaining native setpoints and calibration carry in cal.

    t2echo_by_flux contains ordered (setpoint, accepted lifetime in us) pairs.
    """

    remaining: list[float]
    cal: CalCarry
    t2echo_by_flux: list[tuple[float, float]]


def init_echo(env: InitEnv[DemoContext], plan: FluxPlan) -> T2EchoState:
    """Create detached echo state from the fixed plan and initial host context."""
    return T2EchoState(list(plan.fluxes), initial_carry(env), [])


@workflow(
    "T2echo_fluxdep",
    plan=FluxPlan,
    tunables=T2EchoTunables,
    state=T2EchoState,
    record=T2EchoPoint,
    init=init_echo,
    requires=("devices", "context"),
)
def t2echo_fluxdep(
    env: WorkflowEnv[DemoContext],
    plan: FluxPlan,
    tun: T2EchoTunables,
    state: T2EchoState,
) -> Step[T2EchoState, T2EchoPoint]:
    """Attempt the next echo point; commit missing fits with an explicit reason.

    The engine supplies detached state and captured tunables. Effects acquire
    and save offline traces. Return Done after the fixed setpoints are exhausted.
    Adapter, plotting, analysis, and saver errors propagate to the engine.
    """
    index = len(plan.fluxes) - len(state.remaining)
    env.pbar("echo points", total=len(plan.fluxes)).set_progress(index)
    redraw(env, "t2echo/vs_flux", state.t2echo_by_flux)
    if not state.remaining:
        return Done()
    flux = state.remaining.pop(0)
    yield from env.set_device(plan.flux_dev, flux)
    cal = yield from calibrate(env, tun.cal, flux, state.cal)
    if isinstance(cal, CalibrationFailed):
        return Next(T2EchoPoint(index, flux, cal, None, cal.reason), state)

    guess = smooth_last([v for _, v in state.t2echo_by_flux], tun.guess_us)
    cfg = DecayCfg(
        "t2echo",
        cal,
        plan.points,
        tun.sweep_factor * guess,
        tun.relax_factor * guess,
        tun.t2echo.reps,
        tun.t2echo.rounds,
    )
    ax = env.axes("t2echo/decay")
    ax.cla()
    (line,) = ax.plot([], [])
    outcome = yield from env.run(
        run_t2echo,
        cfg,
        save=save_decay,
        live=Live1D(line, y=lambda signals: signals.real),
    )
    if isinstance(outcome, Failed):
        return Next(T2EchoPoint(index, flux, cal, None, outcome.reason), state)
    fit = analyze_decay(outcome.cfg, outcome.result)
    if fit is None or fit.rel_err >= tun.max_rel_err:
        return Next(T2EchoPoint(index, flux, cal, None, "fit rejected"), state)
    state.t2echo_by_flux.append((flux, fit.value_us))
    return Next(T2EchoPoint(index, flux, cal, fit, None), state)


class OvernightPlan(BaseModel):
    """Fixed overnight plan.

    flux_dev names the device; flux is its native setpoint. period_s separates
    scheduled starts from the original UTC start. rounds counts outer steps;
    points is samples per decay curve.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    flux_dev: str = Field(min_length=1)
    flux: float = Field(description="Absolute setpoint in the device's native units")
    period_s: float = Field(
        default=600.0, gt=0, description="Interval between scheduled starts"
    )
    rounds: int = Field(
        default=12, ge=1, description="Outer workflow iterations, not acquire rounds"
    )
    points: int = Field(default=51, ge=3)


class OvernightTunables(BaseModel):
    """Overnight knobs with no result feedback.

    cal controls calibration; t1 controls acquire averaging. stop_us and
    relax_delay_us are fixed timing. max_rel_err is the strict upper bound on
    relative fit uncertainty. Changing cal does not invalidate saved calibration.
    """

    model_config = ConfigDict(extra="forbid")
    cal: CalTunables = Field(default_factory=CalTunables)
    t1: DecayKnobs = Field(default_factory=DecayKnobs)
    stop_us: float = Field(default=100.0, gt=0.5)
    relax_delay_us: float = Field(default=60.0, gt=0)
    max_rel_err: float = Field(default=0.3, ge=0.01, le=1.0)


@dataclass(frozen=True)
class T1Round:
    """One attempted overnight round.

    index is zero-based; cal records calibration success/failure. t1 is an
    accepted estimate or None; reason explains missing estimates and is None
    on success.
    """

    index: int
    cal: Calibration
    t1: DecayValue | None
    reason: str | None


@dataclass
class OvernightState:
    """Overnight state retained between committed steps.

    index is the next zero-based round. start is the original aware UTC origin.
    carry holds calibration history; cal is None until full success, then reused.
    """

    index: int
    start: datetime
    carry: CalCarry
    cal: Calibrated | None = None


def init_overnight(env: InitEnv[DemoContext], _plan: OvernightPlan) -> OvernightState:
    """Create round-zero state retaining the original UTC start and context."""
    return OvernightState(0, env.started_at, initial_carry(env))


@workflow(
    "T1_overnight",
    plan=OvernightPlan,
    tunables=OvernightTunables,
    state=OvernightState,
    record=T1Round,
    init=init_overnight,
    requires=("devices", "context"),
)
def t1_overnight(
    env: WorkflowEnv[DemoContext],
    plan: OvernightPlan,
    tun: OvernightTunables,
    state: OvernightState,
) -> Step[OvernightState, T1Round]:
    """Attempt a round at its original absolute deadline, without fit feedback.

    The engine supplies detached state and captured tunables. Set flux on round
    zero, reuse the first accepted calibration, and commit rejected fits with a
    reason. Return Done after plan.rounds. Adapter, analysis, and saver errors
    propagate to the engine.
    """
    env.pbar("rounds", total=plan.rounds).set_progress(state.index)
    if state.index >= plan.rounds:
        return Done()
    index = state.index
    state.index += 1
    yield from env.wait_until(state.start + timedelta(seconds=index * plan.period_s))
    if index == 0:
        yield from env.set_device(plan.flux_dev, plan.flux)

    cal = state.cal
    if cal is None:
        cal = yield from calibrate(env, tun.cal, plan.flux, state.carry)
        if isinstance(cal, CalibrationFailed):
            return Next(T1Round(index, cal, None, cal.reason), state)
        state.cal = cal
    cfg = DecayCfg(
        "t1",
        cal,
        plan.points,
        tun.stop_us,
        tun.relax_delay_us,
        tun.t1.reps,
        tun.t1.rounds,
    )
    outcome = yield from env.run(run_t1, cfg, save=save_decay)
    if isinstance(outcome, Failed):
        return Next(T1Round(index, cal, None, outcome.reason), state)
    fit = analyze_decay(outcome.cfg, outcome.result)
    if fit is None or fit.rel_err >= tun.max_rel_err:
        return Next(T1Round(index, cal, None, "fit rejected"), state)
    return Next(T1Round(index, cal, fit, None), state)


def register_all(registry: WorkflowRegistry) -> None:
    """Add the three offline workflows to a host-owned catalog.

    Import does not create a registry. Duplicate names raise ValueError from the
    supplied registry; the host creates a fresh registry for each reload.
    """
    registry.add(t1_fluxdep)
    registry.add(t2echo_fluxdep)
    registry.add(t1_overnight)
