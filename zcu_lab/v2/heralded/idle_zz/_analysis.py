"""Analyze each bias in actual spectator g/e order, after nominal completion masking."""

from __future__ import annotations

from dataclasses import dataclass, field
from textwrap import fill
from typing import TYPE_CHECKING, Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.analysis.fitting import (
    FitQuality,
    WeightedFringeTrace,
    fit_weighted_decay_fringe,
)
from zcu_tools.analysis.shot_uncertainty import class_probability_covariance
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.support.heralded_projection import align_terminal_counts
from zcu_lab.v2.support.paired_counts import (
    qubit_marginal_axis,
    qubit_sigma,
    qubit_sigma_fit_errors,
)

from ..counts import terminal_counts

if TYPE_CHECKING:
    from .core import IdleZZHeraldedCfg, IdleZZHeraldedResult


@dataclass(frozen=True)
class IdleZZFrequency:
    """One actual branch's fit; MHz throughout, with no placeholder estimates.

    magnitude_mhz and frequency_se_mhz are the fitted positive frequency and
    covariance standard error, or None if unavailable. signed_mhz is only set
    when the detuning sets a resolvable sign. sign_resolution_mhz is the fold
    guard distance, or None without a fit. reason explains an absent signed
    estimate. quality_by_herald reports per-trace residual/covariance metrics
    from the same fit, or is empty when no model could be fitted.
    """

    magnitude_mhz: float | None = None
    frequency_se_mhz: float | None = None
    signed_mhz: float | None = None
    sign_resolution_mhz: float | None = None
    reason: str | None = None
    quality_by_herald: dict[str, FitQuality] = field(default_factory=dict)


@dataclass(frozen=True)
class IdleZZActualBranch:
    """One actual spectator state, 'g' or 'e'.

    primary shares frequency across the two spectator-g herald classes.
    gg_only and spectator_e (keyed by herald Qubit1/Qubit2 label) are separate
    diagnostic fits and cannot contribute to ζ.
    """

    spectator: str
    primary: IdleZZFrequency
    gg_only: IdleZZFrequency
    spectator_e: dict[str, IdleZZFrequency]


@dataclass(frozen=True)
class IdleZZBiasAnalysis:
    """Signed conditional frequency difference at one saved bias coordinate.

    dc_value uses the cfg's DC unit, or 0 when not swept. branches are actual
    g then e. ζ and its independent-branch quadrature SE are in MHz, both None
    when a branch has no signed estimate; zeta_reason explains the gap. The
    ignored_herald_shots count sums all acquired heralds with either qubit Other.
    """

    dc_value: float
    branches: tuple[IdleZZActualBranch, ...]
    zeta_mhz: float | None
    zeta_se_mhz: float | None
    zeta_reason: str | None
    ignored_herald_shots: int


@dataclass(frozen=True)
class IdleZZHeraldedAnalysis:
    """Per-bias scalar estimates and diagnostics keyed by their `ramsey_<index>` figure.

    Bias entries follow saved DC order. No per-delay samples or curves are returned.
    """

    biases: dict[str, IdleZZBiasAnalysis]


@dataclass(frozen=True)
class _Trace:
    contrast: NDArray[np.float64]
    sem: NDArray[np.float64]
    sigma: NDArray[np.float64]
    complete: bool


@dataclass(frozen=True)
class _Fit:
    frequency: IdleZZFrequency
    models: dict[str, WeightedFringeTrace]


def shot_sem(
    counts: NDArray[np.float64], target: Literal["qubit1", "qubit2"]
) -> NDArray[np.float64]:
    """Return shot SEM of target e-minus-g contrast from (..., Q1, Q2) counts.

    Terminal Other is included in N. Unmeasured or fewer than two shots have
    NaN SEM. This error is for display, not the weights used in the fit.
    """
    marginal = counts.sum(axis=qubit_marginal_axis(target))
    cov = class_probability_covariance(marginal)
    return np.sqrt(np.maximum(cov[..., 0, 0] + cov[..., 1, 1] - 2 * cov[..., 0, 1], 0))


def _complete_points(
    cfg: IdleZZHeraldedCfg, result: IdleZZHeraldedResult
) -> NDArray[np.bool_]:
    dc = np.array([0.0]) if cfg.dc_device is None else sweep2array(cfg.dc)
    delays, sums = result.delays, result.sums
    if (
        not np.array_equal(result.dc_values, dc)
        or delays.shape != (cfg.delay.expts,)
        or not np.all(np.isfinite(delays) & (delays >= 0))
        or len(np.unique(delays)) != len(delays)
        or sums.shape != (len(dc), 2, len(delays), 3, 3, 3, 3)
    ):
        raise ValueError(
            "Heralded idle ZZ requires saved DC × I/Xπ × distinct delays × 81 classes"
        )
    axes = (-4, -3, -2, -1)
    complete = np.all(np.isfinite(sums) & (sums >= 0), axis=axes)
    missing = np.all(np.isnan(sums), axis=axes)
    if not np.all(complete | missing) or np.any(
        sums.sum(axis=axes) > cfg.reps * cfg.rounds
    ):
        raise ValueError(
            "Heralded idle ZZ counts must be nonnegative, within requested shots, or all NaN"
        )
    return complete


def _aligned_traces(
    cfg: IdleZZHeraldedCfg, result: IdleZZHeraldedResult, order: NDArray[np.intp]
) -> dict[str, NDArray[np.float64]]:
    # Dict order: two fixed primary classes, then two separate comparisons.
    heralds: tuple[tuple[Literal["g", "e"], Literal["g", "e"]], ...] = (
        (("g", "g"), ("e", "g"), ("g", "e"), ("e", "e"))
        if cfg.target == "qubit1"
        else (("g", "g"), ("g", "e"), ("e", "g"), ("e", "e"))
    )
    complete = _complete_points(cfg, result)[:, :, order]
    aligned = {}
    for herald in heralds:
        counts = terminal_counts(result.sums, herald)[:, :, order]
        masked = np.where(complete[..., None, None], counts, np.nan)
        aligned["".join(herald)] = align_terminal_counts(
            masked, cfg.target, herald, spectator_axis=1
        )
    return aligned


def _trace(counts: NDArray[np.float64], cfg: IdleZZHeraldedCfg) -> _Trace:
    return _Trace(
        contrast=qubit_sigma(counts, cfg.target),
        sem=shot_sem(counts, cfg.target),
        sigma=qubit_sigma_fit_errors(counts, cfg.target),
        complete=bool(np.all(np.isfinite(counts))),
    )


def _unavailable_reason(
    traces: dict[str, _Trace], delays: NDArray[np.float64]
) -> str | None:
    if not (len(delays) >= 6 and np.allclose(np.diff(delays), np.diff(delays)[0])):
        return "Need at least six distinct regularly spaced delays"
    for label, trace in traces.items():
        if not trace.complete:
            return f"{label}: no completed acquisition at a required delay"
        if not np.all(np.isfinite(trace.contrast)):
            return f"{label}: no accepted shots at a required delay"
        if np.ptp(trace.contrast) <= np.finfo(float).eps:
            return f"{label}: no identifiable Ramsey contrast"
    return None


def _fit(traces: dict[str, _Trace], delays: NDArray[np.float64], detune: float) -> _Fit:
    reason = _unavailable_reason(traces, delays)
    if reason is not None:
        return _Fit(IdleZZFrequency(reason=reason), {})
    try:
        fit = fit_weighted_decay_fringe(
            delays,
            tuple(t.contrast for t in traces.values()),
            tuple(t.sigma for t in traces.values()),
        )
    except (
        AnalysisFailed,
        ValueError,
        RuntimeError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as exc:
        return _Fit(IdleZZFrequency(reason=f"Fringe fit failed: {exc}"), {})

    magnitude, se = fit.frequency, fit.frequency_se
    resolution = max(1 / float(np.ptp(delays)), se)
    if not fit.shared.diagnostics.valid:
        reason = "Shared fringe optimizer did not converge"
    elif not np.isfinite(se) or not np.isfinite(magnitude):
        reason = "Frequency or its fit SE is not finite; sign unidentifiable"
    elif detune == 0 or min(magnitude, abs(magnitude - 2 * abs(detune))) <= resolution:
        reason = "Sign unidentifiable near 0 or twice |detuning|; possible folding"
    models = dict(zip(traces, fit.traces, strict=True))
    return _Fit(
        IdleZZFrequency(
            magnitude_mhz=magnitude if np.isfinite(magnitude) else None,
            frequency_se_mhz=se if np.isfinite(se) else None,
            signed_mhz=None if reason else float(np.sign(detune) * magnitude),
            sign_resolution_mhz=resolution,
            reason=reason,
            quality_by_herald={label: model.quality for label, model in models.items()},
        ),
        models,
    )


def _summarize_bias(
    bias: float, branches: tuple[IdleZZActualBranch, ...], ignored_shots: int
) -> IdleZZBiasAnalysis:
    ground, excited = (branch.primary for branch in branches)
    if (
        ground.signed_mhz is None
        or excited.signed_mhz is None
        or ground.frequency_se_mhz is None
        or excited.frequency_se_mhz is None
    ):
        reason = "; ".join(
            f"Spectator {branch.spectator}: {branch.primary.reason}"
            for branch in branches
            if branch.primary.reason is not None
        )
        return IdleZZBiasAnalysis(bias, branches, None, None, reason, ignored_shots)
    return IdleZZBiasAnalysis(
        bias,
        branches,
        excited.signed_mhz - ground.signed_mhz,
        float(np.hypot(ground.frequency_se_mhz, excited.frequency_se_mhz)),
        None,
        ignored_shots,
    )


def _draw_trace(
    ax: Axes,
    label: str,
    trace: _Trace,
    fit: _Fit,
    delays: NDArray[np.float64],
    *,
    color: str,
) -> None:
    ax.errorbar(
        delays,
        trace.contrast,
        yerr=trace.sem,
        fmt=".",
        color=color,
        label=f"{label} data",
    )
    model = fit.models.get(label)
    if model is not None:
        dense = np.linspace(delays[0], delays[-1], max(201, len(delays)))
        ax.plot(dense, model.evaluate(dense), color=color, label=f"{label} fit")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))


def _fit_note(fit: IdleZZFrequency, *, width: int) -> str:
    lines = []
    if fit.magnitude_mhz is not None:
        name, value = (
            ("|f|", fit.magnitude_mhz)
            if fit.signed_mhz is None
            else ("f", fit.signed_mhz)
        )
        error = "" if fit.frequency_se_mhz is None else f" ± {fit.frequency_se_mhz:.2g}"
        lines.append(f"{name} = {value:.5g}{error} MHz (fit SE)")
    if fit.quality_by_herald:
        for label, quality in fit.quality_by_herald.items():
            if quality.r2 is not None and quality.normalized_residual_rms is not None:
                lines.append(
                    f"{label}: R² {quality.r2:.3f} · residual RMS/span {quality.normalized_residual_rms:.3g}"
                )
    if fit.reason:
        lines.append(fill(fit.reason, width=width))
    return "\n".join(lines)


def _draw_sweep(
    plots: Plots, cfg: IdleZZHeraldedCfg, biases: list[IdleZZBiasAnalysis]
) -> None:
    _, ax = plots.subplots(num="zeta", figsize=(8, 5), layout="constrained")
    values = np.array([b.zeta_mhz for b in biases], dtype=np.float64)
    ax.errorbar(
        [b.dc_value for b in biases],
        values,
        yerr=np.array([b.zeta_se_mhz for b in biases], dtype=np.float64),
        fmt="o-",
    )
    if not np.any(np.isfinite(values)):
        ax.text(
            0.5,
            0.5,
            "ζ unavailable: see branch diagnostics",
            transform=ax.transAxes,
            ha="center",
        )
    unit = "native unit" if cfg.dc_unit == "native" else cfg.dc_unit
    ax.set(
        xlabel=f"Coupler DC bias · {cfg.dc_device} ({unit})",
        ylabel="ζ = f(e) − f(g) (MHz)",
        title=f"Heralded idle ZZ · {cfg.target.capitalize()} · signed shift",
    )


def analyze_idle_zz(
    cfg: IdleZZHeraldedCfg, result: IdleZZHeraldedResult, plots: Plots
) -> IdleZZHeraldedAnalysis:
    """Fit fixed herald classes at saved biases and draw data/fit diagnostics.

    Use actual g/e spectator rows and sorted saved µs delays. Invalid saved
    axes/counts raise ValueError. Missing data or failed fits keep observations
    and reasons but do not create substitute estimates. No writeback occurs.
    """
    order = np.argsort(result.delays)
    aligned = _aligned_traces(cfg, result, order)
    delays = result.delays[order]
    main = tuple(aligned)[:2]
    comparisons = ("gg", *tuple(aligned)[2:])
    biases = []
    for index, bias in enumerate(result.dc_values):
        suffix = "" if cfg.dc_device is None else f" · Bias point {index + 1}"
        title = f"Heralded idle ZZ · {cfg.target.capitalize()}{suffix}"
        fig, _ = plots.subplots(
            num=f"ramsey_{index}",
            nrows=2,
            ncols=2,
            figsize=(12, 6),
            gridspec_kw={"height_ratios": (3, 1.4)},
            layout="constrained",
        )
        diag, _ = plots.subplots(
            num=f"diagnostics_{index}",
            nrows=2,
            ncols=3,
            figsize=(14, 8),
            layout="constrained",
        )
        fig.suptitle(f"{title} · Shared-frequency Ramsey")
        diag.suptitle(f"{title} · Independent diagnostics (not used for ζ)")
        diag.supxlabel("Ramsey free-evolution delay (µs)")
        diag.supylabel("Aligned contrast P(e) − P(g)")
        branches = []
        for state, spectator in enumerate(("g", "e")):
            traces = {
                label: _trace(counts[index, state], cfg)
                for label, counts in aligned.items()
            }
            main_traces = {label: traces[label] for label in main}
            joint = _fit(main_traces, delays, cfg.detune)
            diagnostic = {
                label: _fit({label: traces[label]}, delays, cfg.detune)
                for label in comparisons
            }
            branches.append(
                IdleZZActualBranch(
                    spectator,
                    joint.frequency,
                    diagnostic["gg"].frequency,
                    {label: diagnostic[label].frequency for label in comparisons[1:]},
                )
            )
            _draw_primary(
                (fig.axes[state], fig.axes[2 + state]), main_traces, joint, delays, spectator
            )
            _draw_diagnostics(diag.axes[3 * state : 3 * state + 3], traces, diagnostic, delays, spectator)
        raw = result.sums[index]
        ignored_shots = int(
            np.nansum(raw[..., 2, :, :, :]) + np.nansum(raw[..., :2, 2, :, :])
        )
        answer = _summarize_bias(float(bias), tuple(branches), ignored_shots)
        biases.append(answer)
        zeta, error = answer.zeta_mhz, answer.zeta_se_mhz
        fig.supxlabel(
            f"ζ = f(e) − f(g) = {zeta:.5g} ± {error:.2g} MHz (fit SE)"
            if zeta is not None and error is not None
            else "ζ unavailable: see actual spectator branch diagnostics"
        )
    if len(biases) > 1:
        _draw_sweep(plots, cfg, biases)
    return IdleZZHeraldedAnalysis({f"ramsey_{i}": bias for i, bias in enumerate(biases)})


def _draw_primary(
    panels: tuple[Axes, Axes],
    traces: dict[str, _Trace],
    joint: _Fit,
    delays: NDArray[np.float64],
    spectator: str,
) -> None:
    ax, notes = panels
    ax.set(
        title=f"Actual spectator {spectator}",
        xlabel="Ramsey free-evolution delay (µs)",
        ylabel="Aligned contrast P(e) − P(g)",
    )
    for index, (label, trace) in enumerate(traces.items()):
        _draw_trace(ax, label, trace, joint, delays, color=f"C{index}")
    ax.legend(loc="best", ncols=2)
    notes.set_axis_off()
    notes.text(
        0,
        0.95,
        _fit_note(joint.frequency, width=60),
        transform=notes.transAxes,
        va="top",
    )


def _draw_diagnostics(
    panels: list[Axes],
    traces: dict[str, _Trace],
    diagnostic: dict[str, _Fit],
    delays: NDArray[np.float64],
    spectator: str,
) -> None:
    for column, (ax, (label, fit)) in enumerate(zip(panels, diagnostic.items(), strict=True)):
        name = "gg-only" if label == "gg" else f"Herald {label}"
        _draw_trace(ax, label, traces[label], fit, delays, color=f"C{column}")
        ax.set_title(
            f"{name} · actual spectator {spectator}\n{_fit_note(fit.frequency, width=38)}"
        )
        ax.legend(loc="best")
