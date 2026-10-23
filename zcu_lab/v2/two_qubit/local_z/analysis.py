"""Saved-source local-Z quadratures and thermal ensemble phase diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import FitQuality, compute_fit_quality
from zcu_tools.experiment.records import RunRecord
from zcu_tools.plotting.plots import Plots

from .._phase import QuadraturePhase, quadrature_phase

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
    """Both targets' saved-source phase diagnostics, ordered F1 then F2.

    phases has shape (2,Nn), each entry retaining contrast, nullable logical
    phase and reason. mean_iq is (2,2,Nn,Ntones), with NaN for incomplete
    projections; acquired_shots is (2,2,Nn). gate_counts identifies the n axis.
    fits contains F1 then F2 complete-row linear fits. No measured phase is
    inferred from a missing independent projection.
    """

    phases: tuple[tuple[QuadraturePhase, ...], ...]
    mean_iq: NDArray[np.complex128]
    acquired_shots: NDArray[np.int64]
    gate_counts: tuple[int, ...]
    fits: tuple[LocalZTargetFit, ...]


def analyze_local_z(
    source: RunRecord[LocalZCfg, LocalZResult], *, plots: Plots
) -> LocalZAnalysis:
    """Analyze complete X/Y round pairs using only saved cfg and raw IQ.

    Missing saved cfg raises ValueError. Missing centers or projection rounds
    retain a reason instead of a phase. Invalid axes, partially missing rounds
    and nonfinite acquired IQ raise ValueError. No thermal shots are excluded.
    Each complete target row uses adjacent principal-value unwrap and an OLS
    line with intercept; correction is minus slope, not rewrapped. Interpretation
    assumes adjacent true phase increments have magnitude below 180 degrees.
    Residual-scaled covariance and fit_quality diagnose, not gate, the model.
    """
    cfg, data = source.cfg, source.result
    if cfg is None:
        raise ValueError(
            "Local-Z analysis requires saved source cfg, not current Form/library"
        )
    mean, counts, complete = _source_quadratures(cfg, data)
    targets: list[tuple[QuadraturePhase, ...]] = []
    for t, (tone, centers) in enumerate(
        (
            (cfg.roles.f1_tone, cfg.f1_centers),
            (cfg.roles.f2_tone, cfg.f2_centers),
        )
    ):
        points: list[QuadraturePhase] = []
        for n in range(len(cfg.gate_counts)):
            if not np.all(complete[t, :, n]):
                point = QuadraturePhase(
                    np.nan, None, "Missing complete X/Y projection rounds"
                )
            elif centers is None:
                point = QuadraturePhase(np.nan, None, "Missing saved target GE centers")
            else:
                point = quadrature_phase(
                    data.signals[t, 0, n, :, tone],
                    data.signals[t, 1, n, :, tone],
                    centers,
                )
            points.append(point)
        targets.append(tuple(points))
    fits = tuple(
        _fit_target(np.array(cfg.gate_counts, dtype=np.float64), points)
        for points in targets
    )
    answer = LocalZAnalysis(tuple(targets), mean, counts, tuple(cfg.gate_counts), fits)
    _plot_analysis(cfg, answer, plots=plots)
    return answer


def _source_quadratures(
    cfg: LocalZCfg,
    data: LocalZResult,
) -> tuple[NDArray[np.complex128], NDArray[np.int64], NDArray[np.bool_]]:
    nn, nt, ns = (
        len(cfg.gate_counts),
        len(cfg.modules.readout.tones),
        cfg.rounds * cfg.reps,
    )
    for actual, expected in (
        (data.targets, np.arange(2)),
        (data.quadratures, np.arange(2)),
        (data.gate_counts, np.array(cfg.gate_counts)),
        (data.shot_indices, np.arange(ns)),
        (data.tone_indices, np.arange(nt)),
    ):
        if not np.array_equal(actual, expected):
            raise ValueError("Local-Z raw axes must match saved cfg")
    if data.signals.shape != (2, 2, nn, ns, nt):
        raise ValueError(
            "Expected raw shape (target,quadrature,n,rounds*reps,full tones)"
        )
    rounds = data.signals.reshape(2, 2, nn, cfg.rounds, cfg.reps, nt)
    acquired = np.all(np.isfinite(rounds), axis=(-2, -1))
    missing = np.all(
        np.isnan(rounds) & ~np.isinf(rounds.real) & ~np.isinf(rounds.imag),
        axis=(-2, -1),
    )
    if not np.all(acquired | missing):
        raise ValueError("Nonfinite acquired IQ or partially missing round")
    complete = np.all(acquired, axis=-1)
    counts = acquired.sum(axis=-1, dtype=np.int64) * cfg.reps
    mean = np.full((2, 2, nn, nt), complex(np.nan, np.nan), dtype=np.complex128)
    for t, q, n in np.argwhere(complete):
        mean[t, q, n] = data.signals[t, q, n].mean(axis=0)
    return mean, counts, complete


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
    fig.suptitle("B00 thermal ensemble · both targets · independent X/Y acquisitions")
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
            "GE-normalized contrast",
            "Logical phase (deg, wrapped)",
            "Logical phase (deg, unwrapped)",
        ),
        strict=True,
    ):
        ax.set(xlabel="CZ gate count", ylabel=label)
        ax.set_xlim(answer.gate_counts[0], answer.gate_counts[-1])
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    for tone, readout in enumerate(cfg.modules.readout.tones):
        fig, _ = plots.subplots(
            f"quadratures_tone_{tone}",
            nrows=2,
            ncols=2,
            figsize=(11, 7),
            layout="constrained",
        )
        fig.suptitle(f"Tone {tone} {readout.label} · B00 thermal ensemble · mean IQ")
        for t in range(2):
            for q in range(2):
                ax = fig.axes[2 * t + q]
                iq = answer.mean_iq[t, q, :, tone]
                ax.plot(answer.gate_counts, iq.real, ".-", label="I")
                ax.plot(answer.gate_counts, iq.imag, ".--", label="Q")
                ax.set(
                    xlabel="CZ gate count",
                    ylabel="Mean IQ (ADC units/sample)",
                    title=f"F{t + 1} target · {('X', 'Y')[q]} projection pulse",
                )
                ax.legend()
