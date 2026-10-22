"""Concrete lifetime acquisition, acceptance and independent delay histories."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from zcu_tools.experiment.workflows import Failed, WorkflowEnv
from zcu_tools.plotting.plots import LinePlot

from zcu_lab.v2.twotone.time_domain.t1.core import T1Exp
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoExp
from zcu_lab.workflows.coherence_fluxdep.acquire import run_t2ramsey
from zcu_lab.workflows.coherence_fluxdep.analysis import (
    analyze_t2echo_fluxdep,
    analyze_t2ramsey,
)
from zcu_lab.workflows.coherence_fluxdep.saving import save_t2ramsey
from zcu_lab.workflows.coherence_fluxdep.t1 import T1Plan, T1Tunables, t1_cfg
from zcu_lab.workflows.coherence_fluxdep.t2 import (
    T2EchoTunables,
    T2RamseyTunables,
    t2echo_cfg,
    t2ramsey_cfg,
)
from zcu_lab.workflows.common.analysis import DecayFit, analyze_t1
from zcu_lab.workflows.common.calibration import Calibrated, CalibratedPi2, StageStatus
from zcu_lab.workflows.common.saving import save_t1, save_t2echo

type LifetimeKind = Literal["t1", "t2ramsey", "t2echo"]


type LifetimeHistory = list[tuple[float, float]]


@dataclass(frozen=True)
class LifetimeResult:
    """An enabled lifetime's scalar outcome and immutable native reference.

    kind is t1/t2ramsey/t2echo, in that acquisition order. status is accepted,
    rejected (Completed fit), failed (Schedule) or skipped (missing current
    frequency/Rabi). reason is None only on acceptance. lifetime_us/error_us
    contain accepted microseconds only, otherwise both None. file is relative
    to the iteration and exists only after Completed, including rejected fits.
    Disabled lifetimes have no result or placeholder in the point tuple.
    Unexpected fit/adapter/writer/plot exceptions propagate, not a status.
    """

    kind: LifetimeKind
    status: StageStatus
    reason: str | None
    lifetime_us: float | None
    error_us: float | None
    file: str | None


def history_guess(initial_us: float, history: LifetimeHistory) -> float:
    """Return this lifetime's existing alpha=.5 delay-grid EWMA in microseconds.

    initial_us is its positive validated guess, history contains only its
    accepted (setpoint, microseconds) pairs. No other lifetime enters this grid.
    The caller retains history; this pure projection does not change it.
    """
    guess = initial_us
    for _, lifetime in history:
        guess = 0.5 * lifetime + 0.5 * guess
    return guess


def _plot(env: WorkflowEnv[Calibrated], kind: LifetimeKind) -> LinePlot:
    name = f"{kind}/decay"
    if name not in env.plots:
        env.plots.subplots(name)

    def rebuild(figure: Figure) -> None:
        figure.clear()
        figure.subplots()

    env.plots.draw(name, rebuild)
    return env.plots.liveplot_1d(
        name, "Time (us)", "Signal", axes=env.plots[name].axes[0]
    )


def _t2_rejection(fit: DecayFit, maximum: float) -> str | None:
    if fit.lifetime_us > 2 * maximum:
        return "T2 lifetime exceeds twice the maximum delay"
    if np.mean(np.abs(fit.observed - fit.fitted)) > 0.1 * np.ptp(fit.fitted):
        return "T2 mean fit error exceeds 0.1 peak-to-peak fit"
    return None


def _finish(
    env: WorkflowEnv[Calibrated],
    kind: LifetimeKind,
    fit: DecayFit,
    times: NDArray[np.float64],
    reason: str | None,
    *,
    directory: Path,
) -> LifetimeResult:
    filename = f"{kind}.h5"
    if reason is not None:
        return LifetimeResult(kind, "rejected", reason, None, None, filename)

    def overlay(figure: Figure) -> None:
        ax = figure.axes[0]
        ax.plot(times, fit.fitted, label="fit")
        ax.set_title(f"{kind} = {fit.lifetime_us:.2f} us")

    env.plots.draw(f"{kind}/decay", overlay)
    env.plots.save(f"{kind}/decay", directory / f"{kind}-decay.png")
    return LifetimeResult(
        kind, "accepted", None, fit.lifetime_us, fit.error_us, filename
    )


async def measure_t1(
    env: WorkflowEnv[Calibrated],
    plan: T1Plan,
    tun: T1Tunables,
    cal: CalibratedPi2,
    guess_us: float,
    *,
    directory: Path,
) -> LifetimeResult:
    """Measure energy relaxation to estimate T1 independently of dephasing.

    plan supplies fixed routing/grid resolution/snapshot, tun supplies native
    averaging/timing/fit gates, and cal is this point's accepted frequency/pi
    with retained legal readout. directory is this point's allocated commit path. save precedes fit, so rejected fits retain raw.
    Normal Schedule failures return failed without a file. Unexpected errors
    propagate; Engine owns cancellation. History updates belong to the caller.
    """
    # Acquire this lifetime on its own delay grid, with a live trace.
    plot = _plot(env, "t1")
    outcome = await env.run(T1Exp().run, t1_cfg(plan, tun, cal, guess_us))
    if isinstance(outcome, Failed):
        return LifetimeResult("t1", "failed", outcome.reason, None, None, None)
    plot.update(outcome.value.result.times, outcome.value.result.signals.real)
    # Save every completed trace before fitting and applying acceptance gates.
    save_t1(outcome, directory / "t1.h5", snapshot=plan.snapshot, identity=env.identity)
    fit = analyze_t1(outcome.value.result)
    residual = fit.quality.normalized_residual_rms
    accepted = (
        np.isfinite(fit.lifetime_us)
        and fit.lifetime_us > 0
        and np.isfinite(fit.error_us)
        and fit.error_us >= 0
        and fit.error_us / fit.lifetime_us < tun.max_rel_err
        and residual is not None
        and residual < tun.max_fit_residual
    )
    reason = None if accepted else "T1 fit rejected"
    return _finish(
        env, "t1", fit, outcome.value.result.times, reason, directory=directory
    )


async def measure_ramsey(
    env: WorkflowEnv[Calibrated],
    plan: T1Plan,
    tun: T2RamseyTunables,
    cal: CalibratedPi2,
    guess_us: float,
    *,
    directory: Path,
) -> LifetimeResult:
    """Measure Ramsey fringes to estimate dephasing without an echo pulse.

    plan supplies fixed routing, grid resolution and historical snapshot; tun
    supplies averaging, timing and detuning. cal contains this point's accepted
    frequency/pi/pi2 and retained legal readout. guess_us is this lifetime's
    positive microsecond delay estimate. directory is this point's allocated
    commit path. Save precedes fit; Schedule failures
    return failed without a file and unexpected errors propagate. Engine owns
    cancellation; history updates belong to the caller. Fitted pi2 is not pi/2.
    Detune remains detune_ratio/length_step; native actual detune is retained.
    Relax remains relax_factor*tun.guess_us, not the current T1 or delay EWMA.
    Original overlong/mean-absolute-error gates apply, with rejected raw retained.
    """
    # Acquire this lifetime on its own delay grid, with a live trace.
    plot = _plot(env, "t2ramsey")
    outcome = await env.run(run_t2ramsey, t2ramsey_cfg(plan, tun, cal, guess_us))
    if isinstance(outcome, Failed):
        return LifetimeResult("t2ramsey", "failed", outcome.reason, None, None, None)
    plot.update(outcome.value.result.times, outcome.value.result.signals.real)
    # Save every completed trace before fitting and applying acceptance gates.
    save_t2ramsey(
        outcome,
        directory / "t2ramsey.h5",
        snapshot=plan.snapshot,
        identity=env.identity,
    )
    fit = analyze_t2ramsey(outcome.value.result)
    return _finish(
        env,
        "t2ramsey",
        fit,
        outcome.value.result.times,
        _t2_rejection(fit, float(np.max(outcome.value.result.times))),
        directory=directory,
    )


async def measure_echo(
    env: WorkflowEnv[Calibrated],
    plan: T1Plan,
    tun: T2EchoTunables,
    cal: CalibratedPi2,
    guess_us: float,
    *,
    directory: Path,
) -> LifetimeResult:
    """Measure Hahn-echo decay to estimate coherence after refocusing slow noise.

    plan supplies fixed routing, grid resolution and historical snapshot; tun
    supplies averaging and fixed relaxation timing. cal contains this point's
    accepted frequency/pi/pi2 and retained legal readout. guess_us is Echo's
    positive microsecond delay estimate. directory is this point's allocated
    commit path. Save precedes fit; Schedule failures
    return failed without a file and unexpected errors propagate. Engine owns
    cancellation. Whole-trace fluxdep projection and T2 gates remain unchanged.
    No extra T1 measurement or fallback is made; history belongs to the caller.
    """
    # Acquire this lifetime on its own delay grid, with a live trace.
    plot = _plot(env, "t2echo")
    outcome = await env.run(T2EchoExp().run, t2echo_cfg(plan, tun, cal, guess_us))
    if isinstance(outcome, Failed):
        return LifetimeResult("t2echo", "failed", outcome.reason, None, None, None)
    plot.update(outcome.value.result.times, outcome.value.result.signals.real)
    # Save every completed trace before fitting and applying acceptance gates.
    save_t2echo(
        outcome, directory / "t2echo.h5", snapshot=plan.snapshot, identity=env.identity
    )
    fit = analyze_t2echo_fluxdep(outcome.value.result)
    return _finish(
        env,
        "t2echo",
        fit,
        outcome.value.result.times,
        _t2_rejection(fit, float(np.max(outcome.value.result.times))),
        directory=directory,
    )
