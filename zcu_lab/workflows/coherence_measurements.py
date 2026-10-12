"""Concrete lifetime acquisition, acceptance and independent delay histories."""

from collections.abc import Generator
from dataclasses import dataclass
from typing import Literal

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from zcu_tools.experiment.workflows import Effect, Failed, WorkflowEnv
from zcu_tools.plotting.plots import LinePlot

from .acquire import run_t1, run_t2echo, run_t2ramsey
from .analysis import DecayFit, analyze_t1, analyze_t2echo_fluxdep, analyze_t2ramsey
from .calibration import Calibrated, CalibratedPi2, StageStatus
from .saving import save_t1, save_t2echo, save_t2ramsey
from .t1 import T1Plan, T1Tunables, t1_cfg
from .t2 import T2EchoTunables, T2RamseyTunables, t2echo_cfg, t2ramsey_cfg

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


def _t1_rejection(fit: DecayFit, tun: T1Tunables) -> str | None:
    residual = fit.quality.normalized_residual_rms
    if not (
        np.isfinite(fit.lifetime_us)
        and fit.lifetime_us > 0
        and np.isfinite(fit.error_us)
        and fit.error_us >= 0
        and fit.error_us / fit.lifetime_us < tun.max_rel_err
        and residual is not None
        and residual < tun.max_fit_residual
    ):
        return "T1 fit rejected"
    return None


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
) -> LifetimeResult:
    filename = f"{kind}.h5"
    if reason is not None:
        return LifetimeResult(kind, "rejected", reason, None, None, filename)

    def overlay(figure: Figure) -> None:
        ax = figure.axes[0]
        ax.plot(times, fit.fitted, label="fit")
        ax.set_title(f"{kind} = {fit.lifetime_us:.2f} us")

    env.plots.draw(f"{kind}/decay", overlay)
    env.plots.save(f"{kind}/decay", env.iteration_dir / f"{kind}-decay.png")
    return LifetimeResult(
        kind, "accepted", None, fit.lifetime_us, fit.error_us, filename
    )


def measure_t1(
    env: WorkflowEnv[Calibrated],
    plan: T1Plan,
    tun: T1Tunables,
    cal: CalibratedPi2,
    guess_us: float,
) -> Generator[Effect, None, LifetimeResult]:
    """Acquire/save/fit T1 with its own positive microsecond grid estimate.

    plan supplies fixed routing/grid resolution/snapshot, tun supplies native
    averaging/timing/fit gates, and cal is this point's accepted frequency/pi
    with retained legal readout. save precedes fit, so rejected fits retain raw.
    Normal Schedule failures return failed without a file. Unexpected errors
    propagate; Engine owns cancellation. History updates belong to the caller.
    """
    plot = _plot(env, "t1")
    outcome = yield from env.run(
        run_t1,
        t1_cfg(plan, tun, cal, guess_us),
        live=lambda axes, samples: plot.update(axes[0], samples.real),
    )
    if isinstance(outcome, Failed):
        return LifetimeResult("t1", "failed", outcome.reason, None, None, None)
    save_t1(outcome, env.iteration_dir / "t1.h5", snapshot=plan.snapshot)
    fit = analyze_t1(outcome.result)
    return _finish(env, "t1", fit, outcome.result.times, _t1_rejection(fit, tun))


def measure_ramsey(
    env: WorkflowEnv[Calibrated],
    plan: T1Plan,
    tun: T2RamseyTunables,
    cal: CalibratedPi2,
    guess_us: float,
) -> Generator[Effect, None, LifetimeResult]:
    """Acquire/save/fringe-fit Ramsey on its independent microsecond delay grid.

    Inputs share measure_t1's native/exception contract. Fitted pi2 is not pi/2.
    Detune remains detune_ratio/length_step; native actual detune is retained.
    Relax remains relax_factor*tun.guess_us, not the current T1 or delay EWMA.
    Original overlong/mean-absolute-error gates apply, with rejected raw retained.
    """
    plot = _plot(env, "t2ramsey")
    outcome = yield from env.run(
        run_t2ramsey,
        t2ramsey_cfg(plan, tun, cal, guess_us),
        live=lambda axes, samples: plot.update(axes[0], samples.real),
    )
    if isinstance(outcome, Failed):
        return LifetimeResult("t2ramsey", "failed", outcome.reason, None, None, None)
    save_t2ramsey(outcome, env.iteration_dir / "t2ramsey.h5", snapshot=plan.snapshot)
    fit = analyze_t2ramsey(outcome.result)
    return _finish(
        env,
        "t2ramsey",
        fit,
        outcome.result.times,
        _t2_rejection(fit, float(np.max(outcome.result.times))),
    )


def measure_echo(
    env: WorkflowEnv[Calibrated],
    plan: T1Plan,
    tun: T2EchoTunables,
    cal: CalibratedPi2,
    guess_us: float,
) -> Generator[Effect, None, LifetimeResult]:
    """Acquire/save/decay-fit zero-detune Echo on its independent delay grid.

    Inputs share measure_ramsey's native timing, fitted-pi2 and error contract.
    The original whole-trace fluxdep projection and T2 gates remain unchanged.
    No extra T1 measurement or fallback is made; history belongs to the caller.
    """
    plot = _plot(env, "t2echo")
    outcome = yield from env.run(
        run_t2echo,
        t2echo_cfg(plan, tun, cal, guess_us),
        live=lambda axes, samples: plot.update(axes[0], samples.real),
    )
    if isinstance(outcome, Failed):
        return LifetimeResult("t2echo", "failed", outcome.reason, None, None, None)
    save_t2echo(outcome, env.iteration_dir / "t2echo.h5", snapshot=plan.snapshot)
    fit = analyze_t2echo_fluxdep(outcome.result)
    return _finish(
        env,
        "t2echo",
        fit,
        outcome.result.times,
        _t2_rejection(fit, float(np.max(outcome.result.times))),
    )
