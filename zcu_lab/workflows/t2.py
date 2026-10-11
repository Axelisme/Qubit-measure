"""Calibrated native Echo and Ramsey flux sweeps with old T2 fit policy."""

from collections.abc import Generator
from dataclasses import dataclass, field
from functools import partial
from typing import Literal

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from zcu_tools.datafile import RunSnapshot
from zcu_tools.experiment.workflows import (
    Done,
    Effect,
    Failed,
    InitEnv,
    Next,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg

from zcu_lab.v2.twotone.time_domain.t2echo.core import (
    T2EchoCfg,
    T2EchoModuleCfg,
    T2EchoSweepCfg,
)
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import (
    T2RamseyCfg,
    T2RamseyModuleCfg,
    T2RamseySweepCfg,
)

from .acquire import run_t2echo, run_t2ramsey
from .analysis import DecayFit, analyze_t2echo_fluxdep, analyze_t2ramsey
from .calibration import (
    CalCarry,
    Calibrated,
    CalibratedPi2,
    CalibrationFailed,
    CalibrationSavers,
    CalibrationTunables,
    apply_readout_calibration,
    calibrate,
)
from .devices import set_flux
from .saving import save_frequency, save_rabi, save_readout, save_t2echo, save_t2ramsey


class T2Plan(BaseModel):
    """Fixed setpoints, native calibration modules and historical snapshot.

    flux_dev names the connected driver; flux_unit is explicit A/V/native with
    no conversion. fluxes are ordered absolute setpoints. points is the number
    of uniform delay samples. qub_pulse supplies shared spectroscopy/Rabi/pi
    waveform and gain; readout supplies pulse-backed routing. snapshot is the
    Start capture for native files, never a live source or destination chooser.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    flux_dev: str = Field(min_length=1)
    flux_unit: Literal["A", "V", "native"]
    fluxes: tuple[float, ...]
    points: int = Field(default=51, ge=5)
    qub_pulse: PulseCfg
    readout: PulseReadoutCfg
    snapshot: RunSnapshot


class T2EchoTunables(BaseModel):
    """Calibration and generation knobs, captured per T2 invocation.

    cal controls native calibration; reps/rounds control T2 averaging. guess_us
    seeds the delay-history EWMA in microseconds; sweep_factor multiplies it
    into maximum delay. relax_factor times guess_us fixes relax timing without
    an additional T1 estimate. Acceptance retains the old two T2 rejection
    rules and is not replaced by T1 relative-error or RMS thresholds.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    cal: CalibrationTunables = Field(default_factory=CalibrationTunables)
    reps: int = Field(default=100, ge=1)
    rounds: int = Field(default=10, ge=1)
    guess_us: float = Field(default=20.0, gt=0)
    sweep_factor: float = Field(default=5.0, gt=0)
    relax_factor: float = Field(default=3.0, gt=0)


class T2RamseyTunables(T2EchoTunables):
    """Ramsey generation adds caller-required detune_ratio, in cycles per step.

    Native detune in MHz equals detune_ratio divided by the requested delay
    step in microseconds. The finite ratio may be zero or signed; there is no
    guessed detune. All inherited fields retain their Echo timing meanings.
    """

    detune_ratio: float


@dataclass
class T2State:
    """Committed setpoint progress, scalar calibration carry and T2 history.

    remaining holds unattempted absolute setpoints; cal is all-success native
    calibration carry. lifetimes holds accepted (flux, lifetime_us) pairs only,
    used for display and the next delay grid, not for a separate T1 estimate.
    """

    remaining: list[float]
    cal: CalCarry
    lifetimes: list[tuple[float, float]] = field(default_factory=list)


@dataclass(frozen=True)
class T2Point:
    """Scalar committed T2 outcome; raw trace remains in native acquisition.

    index is zero-based in plan.fluxes; flux is the device-unit setpoint.
    calibration is fitted CalibratedPi2 or a stage/reason failure. lifetime_us
    and error_us are fitted microsecond values, both None after rejection or
    native failure. reason is None on acceptance, otherwise a specific diagnosis.
    """

    index: int
    flux: float
    calibration: CalibratedPi2 | CalibrationFailed
    lifetime_us: float | None
    error_us: float | None
    reason: str | None


def init_t2(env: InitEnv[Calibrated], plan: T2Plan) -> T2State:
    """Copy fixed setpoints and initialize calibration carry from scalar context.

    Missing Calibrated context raises via env.context; no I/O or plots occur.
    """
    return T2State(list(plan.fluxes), CalCarry(env.context))


def t2echo_cfg(
    plan: T2Plan, tun: T2EchoTunables, cal: CalibratedPi2, guess_us: float
) -> T2EchoCfg:
    """Build detached zero-detune Echo cfg from the successful fitted pi/pi2.

    guess_us is the positive finite delay-grid estimate in microseconds.
    Template waveform/gain is identical for pi and pi/2, with separate fitted
    durations. Readout scalars use calibration. Delay ends at sweep_factor times
    guess_us, relax at relax_factor times tun.guess_us. Invalid grid estimates
    raise ValueError; no setup/acquisition occurs or caller models change.
    """
    sweep = _sweep(plan, tun, guess_us)
    pi = plan.qub_pulse.model_copy(deep=True)
    pi.set_param("freq", cal.f01_mhz)
    pi.set_param("length", cal.pi_length_us)
    pi2 = plan.qub_pulse.model_copy(deep=True)
    pi2.set_param("freq", cal.f01_mhz)
    pi2.set_param("length", cal.pi2_length_us)
    ro = plan.readout.model_copy(deep=True)
    apply_readout_calibration(ro, cal)
    return T2EchoCfg(
        reps=tun.reps,
        rounds=tun.rounds,
        relax_delay=tun.relax_factor * tun.guess_us,
        detune=0.0,
        modules=T2EchoModuleCfg(pi_pulse=pi, pi2_pulse=pi2, readout=ro),
        sweep=T2EchoSweepCfg(length=sweep),
    )


def t2ramsey_cfg(
    plan: T2Plan, tun: T2RamseyTunables, cal: CalibratedPi2, guess_us: float
) -> T2RamseyCfg:
    """Build detached Ramsey cfg with native detune_ratio/length_step MHz.

    Inputs share t2echo_cfg's fixed modules and positive finite microsecond grid
    contract. Pi/2 uses fit.pi2_length_us at the same waveform/gain, never half
    the fitted pi. No hardware, plotting or saving occurs; cfg errors propagate.
    """
    sweep = _sweep(plan, tun, guess_us)
    pi2 = plan.qub_pulse.model_copy(deep=True)
    pi2.set_param("freq", cal.f01_mhz)
    pi2.set_param("length", cal.pi2_length_us)
    ro = plan.readout.model_copy(deep=True)
    apply_readout_calibration(ro, cal)
    return T2RamseyCfg(
        reps=tun.reps,
        rounds=tun.rounds,
        relax_delay=tun.relax_factor * tun.guess_us,
        detune=tun.detune_ratio / sweep.step,
        modules=T2RamseyModuleCfg(pi2_pulse=pi2, readout=ro),
        sweep=T2RamseySweepCfg(length=sweep),
    )


def _sweep(plan: T2Plan, tun: T2EchoTunables, guess: float) -> SweepCfg:
    if not np.isfinite(guess) or guess <= 0:
        raise ValueError("T2 grid guess must be positive and finite")
    stop = tun.sweep_factor * guess
    return SweepCfg(
        start=0.0, stop=stop, expts=plan.points, step=stop / (plan.points - 1)
    )


def _reject(fit: DecayFit, maximum: float) -> str | None:
    if fit.lifetime_us > 2 * maximum:
        return "T2 lifetime exceeds twice the maximum delay"
    if np.mean(np.abs(fit.observed - fit.fitted)) > 0.1 * np.ptp(fit.fitted):
        return "T2 mean fit error exceeds 0.1 peak-to-peak fit"
    return None


def _measure(
    env: WorkflowEnv[Calibrated],
    plan: T2Plan,
    tun: T2EchoTunables,
    cal: CalibratedPi2,
    guess: float,
) -> Generator[Effect, None, tuple[DecayFit, NDArray[np.float64]] | Failed]:
    kind = "ramsey" if isinstance(tun, T2RamseyTunables) else "echo"
    name = f"t2{kind}/decay"
    axes = None
    if name in env.plots:

        def rebuild(figure: Figure) -> None:
            figure.clear()
            figure.subplots()

        env.plots.draw(name, rebuild)
        axes = env.plots[name].axes[0]
    plot = env.plots.liveplot_1d(name, "Time (us)", "Signal", axes=axes)
    if isinstance(tun, T2RamseyTunables):
        outcome = yield from env.run(
            run_t2ramsey,
            t2ramsey_cfg(plan, tun, cal, guess),
            save=partial(save_t2ramsey, snapshot=plan.snapshot),
            live=lambda axes, samples: plot.update(axes[0], samples.real),
        )
        if isinstance(outcome, Failed):
            return outcome
        return analyze_t2ramsey(outcome.result), outcome.result.times
    echo = yield from env.run(
        run_t2echo,
        t2echo_cfg(plan, tun, cal, guess),
        save=partial(save_t2echo, snapshot=plan.snapshot),
        live=lambda axes, samples: plot.update(axes[0], samples.real),
    )
    if isinstance(echo, Failed):
        return echo
    return analyze_t2echo_fluxdep(echo.result), echo.result.times


def _step(
    env: WorkflowEnv[Calibrated], plan: T2Plan, tun: T2EchoTunables, state: T2State
) -> Step[T2State, T2Point]:
    _ = env.soc
    _ = env.devices
    _ = env.context
    kind = "ramsey" if isinstance(tun, T2RamseyTunables) else "echo"
    index = len(plan.fluxes) - len(state.remaining)
    env.pbar("flux points", total=len(plan.fluxes)).set_progress(index)
    summary_name = f"t2{kind}/vs_flux"
    if summary_name not in env.plots:
        env.plots.subplots(summary_name)

    def summary(figure: Figure) -> None:
        figure.clear()
        ax = figure.subplots()
        if state.lifetimes:
            fluxes, lifetimes = zip(*state.lifetimes, strict=True)
            ax.scatter(fluxes, lifetimes)
        ax.set(xlabel="flux setpoint", ylabel=f"T2 {kind} (us)")

    env.plots.draw(summary_name, summary)
    if not state.remaining:
        env.plots.save(summary_name, env.iter_dir / "summary.png")
        return Done()
    flux = state.remaining.pop(0)
    yield from set_flux(env, plan.flux_dev, flux, unit=plan.flux_unit)
    cal = yield from calibrate(
        env,
        qub_pulse=plan.qub_pulse,
        readout=plan.readout,
        tun=tun.cal,
        flux=flux,
        carry=state.cal,
        t1_estimate_us=tun.guess_us,
        savers=CalibrationSavers(
            partial(save_frequency, snapshot=plan.snapshot),
            partial(save_rabi, snapshot=plan.snapshot),
            partial(save_readout, snapshot=plan.snapshot),
        ),
    )
    if isinstance(cal, CalibrationFailed):
        return Next(T2Point(index, flux, cal, None, None, cal.reason), state)
    guess = tun.guess_us
    for _, lifetime in state.lifetimes:
        guess = 0.5 * lifetime + 0.5 * guess
    measured = yield from _measure(env, plan, tun, cal, guess)
    if isinstance(measured, Failed):
        return Next(T2Point(index, flux, cal, None, None, measured.reason), state)
    fit, times = measured
    name = f"t2{kind}/decay"
    reason = _reject(fit, float(np.max(times)))
    if reason is not None:
        return Next(T2Point(index, flux, cal, None, None, reason), state)

    def overlay(figure: Figure) -> None:
        ax = figure.axes[0]
        ax.plot(times, fit.fitted, label="fit")
        ax.set_title(f"T2 {kind} = {fit.lifetime_us:.2f} us")

    env.plots.draw(name, overlay)
    env.plots.save(name, env.iter_dir / "decay.png")
    state.lifetimes.append((flux, fit.lifetime_us))
    return Next(T2Point(index, flux, cal, fit.lifetime_us, fit.error_us, None), state)


@workflow("T2echo_fluxdep", plan=T2Plan, tunables=T2EchoTunables, init=init_t2)
def t2echo_fluxdep(
    env: WorkflowEnv[Calibrated], plan: T2Plan, tun: T2EchoTunables, state: T2State
) -> Step[T2State, T2Point]:
    """Calibrate each setpoint, acquire zero-detune Echo and commit scalar evidence.

    Required soc/devices/context are checked before effects. Whole-trace decay
    fit rejects lifetime >2*max delay or mean absolute residual >0.1 fit ptp.
    Rejects/native failures commit None values and diagnosis while retaining raw
    files. Success updates T2 delay history only; no extra T1 is measured.
    Native/fit/plot/save exceptions propagate. Engine owns cancellation/commit.
    """
    return (yield from _step(env, plan, tun, state))


@workflow("T2ramsey_fluxdep", plan=T2Plan, tunables=T2RamseyTunables, init=init_t2)
def t2ramsey_fluxdep(
    env: WorkflowEnv[Calibrated], plan: T2Plan, tun: T2RamseyTunables, state: T2State
) -> Step[T2State, T2Point]:
    """Calibrate each setpoint and acquire Ramsey with required detune_ratio.

    Uses the same effects/commit/failure contract as t2echo_fluxdep, but fits the
    whole trace with fit_decay_fringe. The native phase ramp is ratio/delay step;
    fitted pi/2 uses the original waveform and gain, not pi duration divided by
    two. Old T2 rejection rules apply unchanged; unexpected exceptions propagate.
    """
    return (yield from _step(env, plan, tun, state))
