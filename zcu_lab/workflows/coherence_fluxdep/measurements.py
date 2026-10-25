"""Concrete lifetime acquisition, acceptance and independent delay histories."""

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from matplotlib.figure import Figure
from zcu_tools.experiment.metadata import RunSnapshot
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import Failed
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.support import LinePlot, live_line
from zcu_lab.v2.support.lifetime_models import (
    LifetimeComparison,
    LifetimeModelRecord,
    draw_lifetime_comparison,
)
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoCfg, T2EchoExp
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import T2RamseyCfg
from zcu_lab.workflows.coherence_fluxdep.acquire import run_t2ramsey
from zcu_lab.workflows.coherence_fluxdep.analysis import (
    analyze_t2echo_fluxdep,
    analyze_t2ramsey,
)
from zcu_lab.workflows.coherence_fluxdep.saving import save_t2ramsey
from zcu_lab.workflows.coherence_fluxdep.t2 import T2EchoTunables
from zcu_lab.workflows.common.analysis import (
    DecayFit,
    compare_decay_fit,
    lifetime_rejection,
)
from zcu_lab.workflows.common.calibration import StageStatus
from zcu_lab.workflows.common.saving import save_native
from zcu_lab.workflows.common.summary import SummaryAccumulator, observe_operation

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
    Disabled or unreached lifetimes leave their named point field None.
    Partial saves then returns rejected/stopped without fitting or acceptance.
    Expected fit failures return failed/reason after saving. Unexpected fit/
    adapter/writer/plot exceptions propagate, not a status.
    detune_mhz is accepted Ramsey's physical frequency offset after subtracting
    the artificial phase ramp, or None for other/nonaccepted measurements.
    models records both diagnostic fits, including discarded/failed alternatives,
    but is empty on partial, dependency/acquisition or primary-fit failure.
    """

    kind: LifetimeKind
    status: StageStatus
    reason: str | None
    lifetime_us: float | None
    error_us: float | None
    file: str | None
    detune_mhz: float | None = None
    models: tuple[LifetimeModelRecord, ...] = ()


@dataclass(frozen=True, kw_only=True)
class RamseyResult(LifetimeResult):
    """Ramsey outcome with kind=t2ramsey and optional values defaulting to None.

    Status, reason, microsecond values, native file, detune and models retain
    LifetimeResult's meanings. Use keywords to record an enabled stage's skip.
    """

    kind: Literal["t2ramsey"] = "t2ramsey"
    reason: str | None = None
    lifetime_us: float | None = None
    error_us: float | None = None
    file: str | None = None


@dataclass(frozen=True, kw_only=True)
class EchoResult(LifetimeResult):
    """Echo outcome with kind=t2echo and optional values defaulting to None.

    Status, reason, microsecond values, native file and models retain
    LifetimeResult's meanings. Use keywords to record an enabled stage's skip.
    """

    kind: Literal["t2echo"] = "t2echo"
    reason: str | None = None
    lifetime_us: float | None = None
    error_us: float | None = None
    file: str | None = None


def lifetime_guess(initial_us: float, history: LifetimeHistory) -> float:
    """Return this lifetime's existing alpha=.5 delay-grid EWMA in microseconds.

    initial_us is its positive validated guess, history contains only its
    accepted (setpoint, lifetime_us, error_us) triples. No other lifetime enters this grid.
    The caller retains history; this pure projection does not change it.
    """
    guess = initial_us
    for _, lifetime, _error in history:
        guess = 0.5 * lifetime + 0.5 * guess
    return guess


# Existing T2 and singleshot callers retain their name during the T1-only pilot.
history_guess = lifetime_guess


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


def _t2_rejection(fit: DecayFit, min_lifetime_points: int) -> str | None:
    rejection = lifetime_rejection(
        fit.quality, fit.times, fit.lifetime_us, min_lifetime_points
    )
    if rejection is not None:
        return rejection
    if fit.lifetime_us > 2 * float(np.max(fit.times)):
        return "T2 lifetime exceeds twice the maximum delay"
    if np.mean(np.abs(fit.observed - fit.fitted)) > 0.1 * np.ptp(fit.fitted):
        return "T2 mean fit error exceeds 0.1 peak-to-peak fit"
    return None


def _finish[Result: LifetimeResult](
    plots: Plots,
    kind: LifetimeKind,
    fit: DecayFit,
    comparison: LifetimeComparison,
    reason: str | None,
    *,
    result_type: type[Result],
) -> Result:
    filename = f"{kind}.h5"

    def overlay(figure: Figure) -> None:
        ax = figure.axes[1]
        # Replace raw I with the observations in the fit's projection.
        ax.clear()
        ax.plot(fit.times, fit.observed, ".", label="observed")
        draw_lifetime_comparison(ax, fit.times, comparison)
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
        return result_type(
            kind=kind,
            status="rejected",
            reason=reason,
            lifetime_us=None,
            error_us=None,
            file=filename,
            models=comparison.records,
        )
    return result_type(
        kind=kind,
        status="accepted",
        reason=None,
        lifetime_us=fit.lifetime_us,
        error_us=fit.error_us,
        file=filename,
        models=comparison.records,
    )


async def measure_ramsey(
    scope: Scope,
    cfg: T2RamseyCfg,
    *,
    tun: T2EchoTunables,
    snapshot: RunSnapshot,
    plots: Plots,
    summary: SummaryAccumulator | None = None,
) -> RamseyResult:
    """Measure Ramsey fringes to estimate dephasing without an echo pulse.

    plan supplies fixed routing, grid resolution and historical snapshot; tun
    supplies averaging, timing and detuning. cal contains this point's accepted
    frequency/pi/pi2 and retained legal readout. guess_us is this lifetime's
    positive microsecond delay estimate. raw data saves in scope.dir.
    summary optionally collects delivered facts. Partial saves then returns rejected/stopped without fitting.
    Schedule failures return failed without a file; expected fit failures return
    failed with the raw filename. Unexpected errors propagate. History updates
    belong to the caller. Fitted pi2 is not pi/2.
    Detune remains detune_ratio/length_step; native actual detune is retained.
    Relax timing is already resolved in cfg from T1 feedback or fixed tunables.
    Accepted physical detune subtracts the actual artificial phase ramp.
    min_lifetime_points requires that many fitted delays in [0, lifetime],
    inclusive; invalid quality rejects too. Original overlong/mean-absolute-error
    gates apply, with rejected raw retained.
    """
    # Acquire this lifetime on its own delay grid, with a live trace.
    plot = _plot(plots, "t2ramsey")
    outcome = await scope.run(run_t2ramsey, cfg, name="Ramsey")
    observe_operation(summary, outcome)
    if isinstance(outcome, Failed):
        return RamseyResult(status="failed", reason=outcome.reason)
    # Save every completed trace before plotting or analyzing it.
    save_t2ramsey(
        outcome,
        scope.dir / "t2ramsey.h5",
        snapshot=snapshot,
        identity=scope.identity,
    )
    if outcome.partial_result:
        return RamseyResult(status="rejected", reason="stopped", file="t2ramsey.h5")
    plot.update(outcome.value.result.times, outcome.value.result.signals.real)
    fit = analyze_t2ramsey(outcome.value.result)
    if isinstance(fit, Failed):
        return RamseyResult(status="failed", reason=fit.reason, file="t2ramsey.h5")
    result = _finish(
        plots,
        "t2ramsey",
        fit,
        compare_decay_fit(
            fit, t1=False, fringe=True, max_window_factor=tun.max_model_window_factor
        ),
        _t2_rejection(fit, tun.min_lifetime_points),
        result_type=RamseyResult,
    )
    return replace(
        result, detune_mhz=fit.detune_mhz if result.status == "accepted" else None
    )


async def measure_echo(
    scope: Scope,
    cfg: T2EchoCfg,
    *,
    tun: T2EchoTunables,
    snapshot: RunSnapshot,
    plots: Plots,
    summary: SummaryAccumulator | None = None,
) -> EchoResult:
    """Measure Hahn-echo decay to estimate coherence after refocusing slow noise.

    plan supplies fixed routing, grid resolution and historical snapshot; tun
    supplies averaging and T1-derived or fixed relaxation timing. cal contains
    this point's accepted frequency/pi/pi2 and retained legal readout. guess_us is Echo's
    positive microsecond delay estimate. raw data saves in scope.dir.
    summary optionally collects delivered facts. Partial saves then returns rejected/stopped without fitting.
    Schedule failures return failed without a file; expected fit failures return
    failed with the raw filename. Unexpected errors propagate.
    min_lifetime_points requires that many fitted delays in [0, lifetime],
    inclusive; invalid quality rejects too. Whole-trace fluxdep projection and
    the original T2 gates remain unchanged. Nonzero
    artificial detune selects a fringe fit; zero selects pure decay.
    No extra T1 measurement or fallback is made; history belongs to the caller.
    """
    # Acquire this lifetime on its own delay grid, with a live trace.
    plot = _plot(plots, "t2echo")
    outcome = await scope.run(T2EchoExp().run, cfg, name="Echo")
    observe_operation(summary, outcome)
    if isinstance(outcome, Failed):
        return EchoResult(status="failed", reason=outcome.reason)
    # Save every completed trace before plotting or analyzing it.
    save_native(
        outcome,
        scope.dir / "t2echo.h5",
        spec=T2EchoExp.AXES_SPEC,
        snapshot=snapshot,
        identity=scope.identity,
    )
    if outcome.partial_result:
        return EchoResult(status="rejected", reason="stopped", file="t2echo.h5")
    plot.update(outcome.value.result.times, outcome.value.result.signals.real)
    fit = analyze_t2echo_fluxdep(outcome.value.result, fit_fringe=cfg.detune != 0.0)
    if isinstance(fit, Failed):
        return EchoResult(status="failed", reason=fit.reason, file="t2echo.h5")
    return _finish(
        plots,
        "t2echo",
        fit,
        compare_decay_fit(
            fit,
            t1=False,
            fringe=cfg.detune != 0.0,
            max_window_factor=tun.max_model_window_factor,
        ),
        _t2_rejection(fit, tun.min_lifetime_points),
        result_type=EchoResult,
    )
