"""Saved-source local-Z quadratures and thermal ensemble phase diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import FitQuality, compute_fit_quality
from zcu_tools.plotting.plots import Plots

from .._phase import QuadraturePhase, count_phase
from .._sums import herald_probabilities, sigma, terminal_counts

if TYPE_CHECKING:
    from .core import LocalZCfg, LocalZResult


@dataclass(frozen=True)
class LocalZTargetFit:
    """One target's complete-row OLS model, phase_deg = intercept + slope*n.

    unwrapped_deg/fit_curve have Nn entries, or None when any phase is missing.
    slope_deg is logical Bloch accrual per CZ; correction_deg is its negative,
    never rewrapped. intercept_deg is the preparation/projection offset.
    slope_error_deg is residual-scaled covariance 1σ, not a shot-noise model.
    fit_quality holds residual/relative uncertainty diagnostics without a gate.
    Unavailable fits retain None values and reason, never a substituted zero.
    """

    unwrapped_deg: NDArray[np.float64] | None = None
    fit_curve: NDArray[np.float64] | None = None
    intercept_deg: float | None = None
    slope_deg: float | None = None
    correction_deg: float | None = None
    slope_error_deg: float | None = None
    fit_quality: FitQuality | None = None
    reason: str | None = None


@dataclass(frozen=True)
class LocalZAnalysis:
    """F1/F2 classified phase diagnostics and complete-row OLS fits.

    phases is (2,Nn); quadratures is (2,2,Nn) sigma, and acquired_shots is
    the projection's terminal total. fits is F1 then F2; gate_counts is the n axis.
    herald_matrix is the full herald distribution or None for ordinary acquisition.
    pass_fraction is herald (g,g) acceptance, not ground-state purity.
    """

    phases: tuple[tuple[QuadraturePhase, ...], ...]
    quadratures: NDArray[np.float64]
    acquired_shots: NDArray[np.float64]
    gate_counts: tuple[int, ...]
    fits: tuple[LocalZTargetFit, ...]
    herald_matrix: NDArray[np.float64] | None = None
    pass_fraction: NDArray[np.float64] | None = None


def analyze_local_z(
    cfg: LocalZCfg, data: LocalZResult, *, plots: Plots, heralded: bool = False
) -> LocalZAnalysis:
    """Derive classified X/Y sigma and phase using saved cfg and count axes.

    Each projection uses all acquired counts, including other in its denominator.
    Missing/empty projections and zero contrast retain a reason, not a phase.
    Heralded analysis selects herald (g,g) terminal counts and plots all nine
    herald fractions. No IQ projection, shot reclassification or purity correction
    is applied. Complete rows use adjacent unwrap and OLS; correction remains minus slope.
    """
    counts = _source_quadratures(cfg, data, heralded=heralded)
    acquired = counts.sum(axis=(-2, -1))
    quadratures = np.stack((sigma(counts[0], "f1"), sigma(counts[1], "f2")))
    targets = tuple(
        tuple(
            count_phase(float(quadratures[t, 0, n]), float(quadratures[t, 1, n]))
            for n in range(len(cfg.gate_counts))
        )
        for t in range(2)
    )
    fits = tuple(
        _fit_target(np.array(cfg.gate_counts, dtype=np.float64), points)
        for points in targets
    )
    answer = LocalZAnalysis(
        targets,
        quadratures,
        acquired,
        tuple(cfg.gate_counts),
        fits,
        herald_probabilities(data.sums) if heralded else None,
        herald_probabilities(data.sums)[..., 0, 0] if heralded else None,
    )
    _plot_analysis(cfg, answer, plots=plots)
    return answer


def _source_quadratures(
    cfg: LocalZCfg, data: LocalZResult, *, heralded: bool
) -> NDArray[np.float64]:
    """Validate saved scan axes and select terminal counts without averaging."""
    for actual, expected in (
        (data.targets, np.arange(2)),
        (data.quadratures, np.arange(2)),
        (data.gate_counts, np.array(cfg.gate_counts)),
    ):
        if not np.array_equal(actual, expected):
            raise ValueError("LocalZ count axes must match saved cfg")
    shape = (2, 2, len(cfg.gate_counts)) + ((3, 3, 3, 3) if heralded else (3, 3))
    if data.sums.shape != shape:
        raise ValueError(f"Expected summed count shape {shape}, got {data.sums.shape}")
    return (terminal_counts(data.sums, ("g", "g")) if heralded else data.sums)


def _fit_target(
    n: NDArray[np.float64], points: tuple[QuadraturePhase, ...]
) -> LocalZTargetFit:
    if any(p.logical_phase_deg is None for p in points):
        reasons = sorted(
            {
                p.reason or "Phase unavailable"
                for p in points
                if p.logical_phase_deg is None
            }
        )
        return LocalZTargetFit(reason="Incomplete phase row: " + "; ".join(reasons))
    y = np.rad2deg(
        np.unwrap(
            np.deg2rad(
                np.array([p.logical_phase_deg for p in points], dtype=np.float64)
            )
        )
    )
    centered = n - n.mean()
    sxx = float(centered @ centered)
    slope = float(centered @ (y - y.mean()) / sxx)
    intercept = float(y.mean() - slope * n.mean())
    curve = intercept + slope * n
    variance = float(np.sum((y - curve) ** 2) / (len(n) - 2))
    covariance = variance * np.array(
        [
            [1 / len(n) + n.mean() ** 2 / sxx, -n.mean() / sxx],
            [-n.mean() / sxx, 1 / sxx],
        ],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite([slope, intercept]))
        or not np.all(np.isfinite(covariance))
        or not np.all(np.isfinite(curve))
    ):
        return LocalZTargetFit(reason="Nonfinite linear fit or covariance uncertainty")
    quality = compute_fit_quality(
        y, curve, {"intercept_deg": intercept, "slope_deg": slope}, covariance
    )
    return LocalZTargetFit(
        y, curve, intercept, slope, -slope, float(np.sqrt(covariance[1, 1])), quality
    )


def _plot_analysis(cfg: LocalZCfg, answer: LocalZAnalysis, *, plots: Plots) -> None:
    fig, _ = plots.subplots(
        "phase", nrows=3, ncols=1, figsize=(10, 9), layout="constrained"
    )
    fig.suptitle("Both targets · independent classified X/Y acquisitions")
    for t, role in enumerate((cfg.roles.f1, cfg.roles.f2)):
        fig.axes[0].plot(
            answer.gate_counts,
            [p.contrast for p in answer.phases[t]],
            ".-",
            label=f"F{t + 1} {role}",
        )
        fig.axes[1].plot(
            answer.gate_counts,
            [
                np.nan if p.logical_phase_deg is None else p.logical_phase_deg
                for p in answer.phases[t]
            ],
            ".-",
            label=f"F{t + 1} {role}",
        )
        fit = answer.fits[t]
        if fit.unwrapped_deg is not None and fit.fit_curve is not None:
            fig.axes[2].plot(
                answer.gate_counts,
                fit.unwrapped_deg,
                ".",
                label=f"F{t + 1} {role} unwrapped",
            )
            fig.axes[2].plot(
                answer.gate_counts,
                fit.fit_curve,
                "-",
                label=f"F{t + 1} OLS; correction={fit.correction_deg:.5g} ± {fit.slope_error_deg:.2g} deg (1σ)",
            )
    fig.axes[2].set_title(
        "\n".join(
            f"F{t + 1}: {fit.reason}" for t, fit in enumerate(answer.fits) if fit.reason
        )
        or "Complete-row OLS with intercept; correction = −slope"
    )
    for ax, label in zip(
        fig.axes,
        (
            "Classified quadrature contrast",
            "Logical phase (deg, wrapped)",
            "Logical phase (deg, unwrapped)",
        ),
        strict=True,
    ):
        ax.set(xlabel="CZ gate count", ylabel=label)
        ax.set_xlim(answer.gate_counts[0], answer.gate_counts[-1])
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    fig, axes = plots.subplots("quadratures", figsize=(10, 5))
    for t in range(2):
        for q, projection in enumerate(("X", "Y")):
            axes.plot(
                answer.gate_counts,
                answer.quadratures[t, q],
                ".-",
                label=f"F{t + 1} {projection}",
            )
    axes.set(xlabel="CZ gate count", ylabel="Target sigma")
    axes.legend()

    if answer.herald_matrix is not None:
        fig, axes = plots.subplots("herald", figsize=(12, 5))
        matrix = answer.herald_matrix.reshape(-1, 9)
        image = axes.imshow(
            np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap="Blues", aspect="auto"
        )
        axes.set(
            xticks=range(9),
            xticklabels=[
                f"{a},{b}" for a in ("g", "e", "other") for b in ("g", "e", "other")
            ],
            xlabel="Herald class (F1, F2)",
            ylabel="Scan row (target, projection, n)",
            title="Herald distribution; (g,g) is pass fraction, not ground-state purity",
        )
        fig.colorbar(image, ax=axes, label="Herald probability")
        fig.tight_layout()
