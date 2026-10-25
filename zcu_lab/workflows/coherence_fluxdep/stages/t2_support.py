"""Shared T2 timing, acceptance comparisons and decay-plot mechanics."""

import math
from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import SweepCfg

from zcu_lab.v2.support import LinePlot, live_line
from zcu_lab.v2.support.lifetime_models import (
    LifetimeComparison,
    draw_lifetime_comparison,
)
from zcu_lab.workflows.coherence_fluxdep.t2 import T2EchoTunables
from zcu_lab.workflows.common.analysis import DecayFit, lifetime_rejection
from zcu_lab.workflows.common.gates import Gate, gate


@dataclass(frozen=True)
class T2Guess:
    """Point-local microsecond estimates, both required positive and finite.

    lifetime_us is this T2 stage's accepted-history EWMA or configured seed.
    t1_us is this point's accepted T1, otherwise T1's EWMA or seed, used only
    for automatic relaxation. Stage cfg construction validates both estimates.
    """

    lifetime_us: float
    t1_us: float


def delay_sweep(settings: T2EchoTunables, *, points: int, guess_us: float) -> SweepCfg:
    """Build the original auto-capped or fixed T2 delay grid in microseconds.

    points is the sample count; guess_us must be finite and positive. Reject
    a generated stop not greater than the configured start. Inputs are unchanged.
    """
    if not math.isfinite(guess_us) or guess_us <= 0:
        raise ValueError("T2 grid guess must be positive and finite")
    stop = (
        min(settings.max_length_us, settings.sweep_factor * guess_us)
        if settings.sweep_mode == "auto"
        else settings.sweep_stop_us
    )
    if stop <= settings.sweep_start_us:
        raise ValueError("T2 generated stop must exceed start")
    return SweepCfg(
        start=settings.sweep_start_us,
        stop=stop,
        expts=points,
        step=(stop - settings.sweep_start_us) / (points - 1),
    )


def relax_delay(settings: T2EchoTunables, t1_estimate_us: float) -> float:
    """Resolve T2 relaxation from positive finite T1 feedback or fixed timing.

    Auto mode floors relax_factor*t1_estimate_us at relax_min_us. Fixed mode
    returns relax_delay_us unchanged. Invalid estimates raise in either mode.
    """
    if not math.isfinite(t1_estimate_us) or t1_estimate_us <= 0:
        raise ValueError("T1 relaxation estimate must be positive and finite")
    return (
        max(settings.relax_min_us, settings.relax_factor * t1_estimate_us)
        if settings.relax_mode == "auto"
        else settings.relax_delay_us
    )


def decay_plot(plots: Plots, kind: str) -> LinePlot:
    """Prepare the lower raw-I panel without clearing accepted history above."""
    if kind not in plots:
        plots.subplots(kind, nrows=2)

    def rebuild(figure: Figure) -> None:
        figure.set_layout_engine("constrained")
        # A new axis belongs to this point, not the previous live template.
        figure.axes[1].remove()
        figure.add_subplot(2, 1, 2)

    plots.draw(kind, rebuild)
    return live_line(
        plots, kind, "Time (us)", "Signal (raw I)", axes=plots[kind].axes[1]
    )


def draw_decay_fit(
    plots: Plots,
    kind: str,
    fit: DecayFit,
    comparison: LifetimeComparison,
    reason: str | None,
) -> None:
    """Draw projected observations and both models, marking a rejected fit."""

    def overlay(figure: Figure) -> None:
        ax = figure.axes[1]
        ax.clear()
        ax.plot(fit.times, fit.observed, ".", label="observed")
        draw_lifetime_comparison(ax, fit.times, comparison)
        ax.set(
            xlabel="Time (us)",
            ylabel="Normalized signal",
            title=(
                f"{kind} = {fit.lifetime_us:.2f} ± {fit.error_us:.2f} us"
                + ("\nRejected fit" if reason is not None else "")
            ),
        )
        ax.legend()

    plots.draw(kind, overlay)


def t2_gates(fit: DecayFit, min_lifetime_points: int) -> tuple[Gate, ...]:
    """Record the existing T2 quality, sampling, window and mean-error gates.

    Delays are the primary fit's microseconds, including its sample exclusions.
    These comparisons do not introduce an uncertainty or residual-RMS threshold.
    """
    return (
        gate("positive_lifetime_us", ">", 0, fit.lifetime_us),
        gate("invalid_quality_count", "==", 0, len(fit.quality.invalid)),
        gate(
            "points_within_lifetime",
            ">=",
            min_lifetime_points,
            int(np.count_nonzero((fit.times >= 0) & (fit.times <= fit.lifetime_us))),
        ),
        gate("lifetime_window_us", "<=", 2 * float(np.max(fit.times)), fit.lifetime_us),
        gate(
            "mean_absolute_fit_error",
            "<=",
            0.1 * float(np.ptp(fit.fitted)),
            float(np.mean(np.abs(fit.observed - fit.fitted))),
        ),
    )


def t2_rejection_reason(
    fit: DecayFit, min_lifetime_points: int, gates: tuple[Gate, ...]
) -> str | None:
    """List failed comparisons after the original lifetime-quality diagnosis.

    The original thresholds and fit-point selection remain unchanged. Numeric
    gate diagnostics supplement quality paths and the within-lifetime count.
    """
    reasons = []
    lifetime_reason = lifetime_rejection(
        fit.quality, fit.times, fit.lifetime_us, min_lifetime_points
    )
    if lifetime_reason is not None:
        reasons.append(lifetime_reason)
    failed_comparison = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!="}
    reasons.extend(
        f"{item.name} {item.actual:.9g} {failed_comparison[item.comparison]} {item.threshold:.9g}"
        if item.actual is not None
        else f"{item.name} unavailable (requires {item.comparison} {item.threshold:.9g})"
        for item in gates
        if not item.passed
    )
    return "; ".join(reasons) or None
