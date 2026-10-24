"""Concrete lifetime acquisition, acceptance and independent delay histories."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from zcu_tools.experiment.metadata import RunSnapshot
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import Failed
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.support import LinePlot, live_line
from zcu_lab.v2.twotone.time_domain.t1.core import T1Cfg, T1Exp
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoCfg, T2EchoExp
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import T2RamseyCfg
from zcu_lab.workflows.coherence_fluxdep.acquire import run_t2ramsey
from zcu_lab.workflows.coherence_fluxdep.analysis import (
    analyze_t2echo_fluxdep,
    analyze_t2ramsey,
)
from zcu_lab.workflows.coherence_fluxdep.saving import save_t2ramsey
from zcu_lab.workflows.coherence_fluxdep.t1 import T1Tunables
from zcu_lab.workflows.common.analysis import DecayFit, analyze_t1
from zcu_lab.workflows.common.calibration import StageStatus
from zcu_lab.workflows.common.saving import save_native

type LifetimeKind = Literal["t1", "t2ramsey", "t2echo"]


type LifetimeHistory = list[tuple[float, float, float]]


@dataclass(frozen=True)
class LifetimeResult:
    """An enabled lifetime's scalar outcome and immutable native reference.

    kind is t1/t2ramsey/t2echo, in that acquisition order. status is accepted,
    rejected (fit gates or stopped partial), failed (Schedule or expected fit
    failure) or skipped (missing current frequency/Rabi).
    reason is None only on acceptance. lifetime_us/error_us
    contain accepted microseconds only, otherwise both None. file is relative
    to the iteration and exists only after Completed, including rejected fits.
    Disabled lifetimes have no result or placeholder in the point tuple.
    Partial saves then returns rejected/stopped without fitting or acceptance.
    Expected fit failures return failed/reason after saving. Unexpected fit/
    adapter/writer/plot exceptions propagate, not a status.
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
    accepted (setpoint, lifetime_us, error_us) triples. No other lifetime enters this grid.
    The caller retains history; this pure projection does not change it.
    """
    guess = initial_us
    for _, lifetime, _error in history:
        guess = 0.5 * lifetime + 0.5 * guess
    return guess


def _plot(plots: Plots, kind: LifetimeKind) -> LinePlot:
    if kind not in plots:
        plots.subplots(kind, nrows=2)

    def rebuild(figure: Figure) -> None:
        figure.set_layout_engine("constrained")
        # A new axis gives this point its own live handle; cleared axes still
        # belong to the previous point\'s registered live template.
        figure.axes[1].remove()
        figure.add_subplot(2, 1, 2)

    plots.draw(kind, rebuild)
    return live_line(
        plots, kind, "Time (us)", "Signal (raw I)", axes=plots[kind].axes[1]
    )


def _t2_rejection(fit: DecayFit, maximum: float) -> str | None:
    if fit.lifetime_us > 2 * maximum:
        return "T2 lifetime exceeds twice the maximum delay"
    if np.mean(np.abs(fit.observed - fit.fitted)) > 0.1 * np.ptp(fit.fitted):
        return "T2 mean fit error exceeds 0.1 peak-to-peak fit"
    return None


def _finish(
    plots: Plots,
    kind: LifetimeKind,
    fit: DecayFit,
    times: NDArray[np.float64],
    reason: str | None,
) -> LifetimeResult:
    filename = f"{kind}.h5"

    def overlay(figure: Figure) -> None:
        ax = figure.axes[1]
        # Replace raw I with the observations in the fit's projection.
        ax.clear()
        ax.plot(times, fit.observed, ".", label="observed")
        ax.plot(times, fit.fitted, label="fit")
        ax.set(
            xlabel="Time (us)",
            ylabel="Signal (projected)" if kind == "t1" else "Normalized signal",
            title=(
                f"{kind} = {fit.lifetime_us:.2f} ± {fit.error_us:.2f} us"
                + ("\nRejected fit" if reason is not None else "")
            ),
        )
        ax.legend()

    plots.draw(kind, overlay)
    if reason is not None:
        return LifetimeResult(kind, "rejected", reason, None, None, filename)
    return LifetimeResult(
        kind, "accepted", None, fit.lifetime_us, fit.error_us, filename
    )


async def measure_t1(
    scope: Scope,
    cfg: T1Cfg,
    *,
    tun: T1Tunables,
    snapshot: RunSnapshot,
    directory: Path,
    plots: Plots,
) -> LifetimeResult:
    """Measure energy relaxation to estimate T1 independently of dephasing.

    plan supplies fixed routing/grid resolution/snapshot, tun supplies native
    averaging/timing/fit gates, and cal is this point's accepted frequency/pi
    with retained legal readout. directory is this point's allocated commit path.
    Save precedes fit, so rejected fits retain raw.
    Partial saves then returns rejected/stopped without fitting. Schedule failures
    return failed without a file; expected fit failures return failed with the raw
    filename. Unexpected errors propagate. History updates belong to the caller.
    """
    # Acquire this lifetime on its own delay grid, with a live trace.
    plot = _plot(plots, "t1")
    outcome = await scope.run(T1Exp().run, cfg, name="T1")
    if isinstance(outcome, Failed):
        return LifetimeResult("t1", "failed", outcome.reason, None, None, None)
    # Save every completed trace before plotting or analyzing it.
    save_native(
        outcome,
        directory / "t1.h5",
        spec=T1Exp.AXES_SPEC,
        snapshot=snapshot,
        identity=scope.identity,
    )
    if outcome.partial_result:
        return LifetimeResult("t1", "rejected", "stopped", None, None, "t1.h5")
    plot.update(outcome.value.result.times, outcome.value.result.signals.real)
    fit = analyze_t1(outcome.value.result)
    if isinstance(fit, Failed):
        return LifetimeResult("t1", "failed", fit.reason, None, None, "t1.h5")
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
    return _finish(plots, "t1", fit, outcome.value.result.times, reason)


async def measure_ramsey(
    scope: Scope,
    cfg: T2RamseyCfg,
    *,
    snapshot: RunSnapshot,
    directory: Path,
    plots: Plots,
) -> LifetimeResult:
    """Measure Ramsey fringes to estimate dephasing without an echo pulse.

    plan supplies fixed routing, grid resolution and historical snapshot; tun
    supplies averaging, timing and detuning. cal contains this point's accepted
    frequency/pi/pi2 and retained legal readout. guess_us is this lifetime's
    positive microsecond delay estimate. directory is this point's allocated
    commit path. Partial saves then returns rejected/stopped without fitting.
    Schedule failures return failed without a file; expected fit failures return
    failed with the raw filename. Unexpected errors propagate. History updates
    belong to the caller. Fitted pi2 is not pi/2.
    Detune remains detune_ratio/length_step; native actual detune is retained.
    Relax remains relax_factor*tun.guess_us, not the current T1 or delay EWMA.
    Original overlong/mean-absolute-error gates apply, with rejected raw retained.
    """
    # Acquire this lifetime on its own delay grid, with a live trace.
    plot = _plot(plots, "t2ramsey")
    outcome = await scope.run(run_t2ramsey, cfg, name="Ramsey")
    if isinstance(outcome, Failed):
        return LifetimeResult("t2ramsey", "failed", outcome.reason, None, None, None)
    # Save every completed trace before plotting or analyzing it.
    save_t2ramsey(
        outcome,
        directory / "t2ramsey.h5",
        snapshot=snapshot,
        identity=scope.identity,
    )
    if outcome.partial_result:
        return LifetimeResult(
            "t2ramsey", "rejected", "stopped", None, None, "t2ramsey.h5"
        )
    plot.update(outcome.value.result.times, outcome.value.result.signals.real)
    fit = analyze_t2ramsey(outcome.value.result)
    if isinstance(fit, Failed):
        return LifetimeResult(
            "t2ramsey", "failed", fit.reason, None, None, "t2ramsey.h5"
        )
    return _finish(
        plots,
        "t2ramsey",
        fit,
        outcome.value.result.times,
        _t2_rejection(fit, float(np.max(outcome.value.result.times))),
    )


async def measure_echo(
    scope: Scope,
    cfg: T2EchoCfg,
    *,
    snapshot: RunSnapshot,
    directory: Path,
    plots: Plots,
) -> LifetimeResult:
    """Measure Hahn-echo decay to estimate coherence after refocusing slow noise.

    plan supplies fixed routing, grid resolution and historical snapshot; tun
    supplies averaging and fixed relaxation timing. cal contains this point's
    accepted frequency/pi/pi2 and retained legal readout. guess_us is Echo's
    positive microsecond delay estimate. directory is this point's allocated
    commit path. Partial saves then returns rejected/stopped without fitting.
    Schedule failures return failed without a file; expected fit failures return
    failed with the raw filename. Unexpected errors propagate.
    Whole-trace fluxdep projection and T2 gates remain unchanged.
    No extra T1 measurement or fallback is made; history belongs to the caller.
    """
    # Acquire this lifetime on its own delay grid, with a live trace.
    plot = _plot(plots, "t2echo")
    outcome = await scope.run(T2EchoExp().run, cfg, name="Echo")
    if isinstance(outcome, Failed):
        return LifetimeResult("t2echo", "failed", outcome.reason, None, None, None)
    # Save every completed trace before plotting or analyzing it.
    save_native(
        outcome,
        directory / "t2echo.h5",
        spec=T2EchoExp.AXES_SPEC,
        snapshot=snapshot,
        identity=scope.identity,
    )
    if outcome.partial_result:
        return LifetimeResult("t2echo", "rejected", "stopped", None, None, "t2echo.h5")
    plot.update(outcome.value.result.times, outcome.value.result.signals.real)
    fit = analyze_t2echo_fluxdep(outcome.value.result)
    if isinstance(fit, Failed):
        return LifetimeResult("t2echo", "failed", fit.reason, None, None, "t2echo.h5")
    return _finish(
        plots,
        "t2echo",
        fit,
        outcome.value.result.times,
        _t2_rejection(fit, float(np.max(outcome.value.result.times))),
    )
