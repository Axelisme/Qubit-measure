"""Joint-return probabilities, duration fitting and count-derived plots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import compute_fit_quality
from zcu_tools.analysis.fitting.quality import FitQuality
from zcu_tools.analysis.fitting.rabi import fit_rabi
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots

from ._rabi_scan import DurationScanCfg, RabiScanResult
from ._sums import herald_probabilities, joint_probabilities, terminal_counts


@dataclass(frozen=True)
class ReturnData:
    """Count-derived terminal probabilities and fit eligibility.

    populations is (Nc,Nd,9), flattened F1/F2 g/e/other order, with P00 first.
    complete marks finite nonempty selected counts, including partial rounds.
    acquired_shots is their total (Nc,Nd), NaN for unmeasured points.
    herald_matrix is (Nc,Nd,3,3) or None for ordinary acquisition.
    Its (g,g) cell is pass fraction, not ground-state purity.
    """

    populations: NDArray[np.float64]
    complete: NDArray[np.bool_]
    acquired_shots: NDArray[np.float64]
    herald_matrix: NDArray[np.float64] | None


def joint_return(
    cfg: DurationScanCfg,
    data: RabiScanResult,
    coordinates: NDArray[np.float64],
    *,
    heralded: bool = False,
) -> ReturnData:
    """Normalize saved 9-class counts, selecting herald (g,g) when enabled.

    Saved coordinate/duration axes and sums shape must match cfg. P00 divides
    the (g,g) count by all nine terminal counts, including other. Zero totals
    yield NaN without warnings and cannot be fitted. Completed partial rounds
    contribute their actual counts; missing points remain NaN.
    """
    durations = sweep2array(cfg.duration)
    nc, nd = len(coordinates), len(durations)
    class_shape = (3, 3, 3, 3) if heralded else (3, 3)
    if (
        data.coordinates.shape != coordinates.shape
        or not np.allclose(
            data.coordinates, coordinates, rtol=8 * np.finfo(float).eps, atol=0
        )
        or data.durations.shape != durations.shape
        or not np.allclose(
            data.durations, durations, rtol=8 * np.finfo(float).eps, atol=0
        )
        or data.sums.shape != (nc, nd, *class_shape)
    ):
        raise ValueError("Rabi count axes/shape must match saved cfg and class axes")
    counts = terminal_counts(data.sums, heralded=heralded)
    totals = counts.sum(axis=(-2, -1))
    complete = np.all(np.isfinite(counts), axis=(-2, -1)) & (totals > 0)
    return ReturnData(
        joint_probabilities(counts).reshape(nc, nd, 9),
        complete,
        totals,
        herald_probabilities(data.sums) if heralded else None,
    )


def plot_return_data(
    data: RabiScanResult,
    answer: ReturnData,
    *,
    coordinate_label: str,
    map_name: str,
    plots: Plots,
) -> None:
    """Draw P00 map, nine terminal return curves and optional herald fractions.

    All curves derive from saved counts. Heralded terminal plots use only
    herald (g,g); the separate herald maps retain all nine acceptance classes.
    map_name names the caller's map, not a model or gate-selection policy.
    """
    fig, ax = plots.subplots(map_name, figsize=(9, 6), layout="constrained")
    image = ax.pcolormesh(
        data.durations, data.coordinates, answer.populations[..., 0],
        shading="nearest", vmin=0, vmax=1,
    )
    fig.colorbar(image, ax=ax, label="Observed joint P00 after double Xπ")
    ax.set(
        xlabel="Drive duration (us)", ylabel=coordinate_label,
        title="Nominal B11 return proxy · not gate fidelity",
    )
    labels = tuple(f"({a},{b})" for a in ("g", "e", "other") for b in ("g", "e", "other"))
    fig, _ = plots.subplots(
        "return", nrows=3, ncols=3, figsize=(13, 10), squeeze=False, layout="constrained"
    )
    for outcome, ax in enumerate(fig.axes):
        for coordinate, values in zip(data.coordinates, answer.populations, strict=True):
            ax.plot(data.durations, values[:, outcome], ".-", label=f"{coordinate:g}")
        ax.set(
            xlabel="Drive duration (us)", ylabel=f"Observed P{labels[outcome]}",
            title=f"Terminal {labels[outcome]} · other retained in denominator",
        )
        if len(data.coordinates) <= 25:
            ax.legend(title=coordinate_label, fontsize="x-small", ncols=2)
    if answer.herald_matrix is not None:
        fig, _ = plots.subplots(
            "herald", nrows=3, ncols=3, figsize=(13, 10), squeeze=False, layout="constrained"
        )
        for outcome, ax in enumerate(fig.axes):
            image = ax.pcolormesh(
                data.durations, data.coordinates,
                answer.herald_matrix[..., outcome // 3, outcome % 3],
                shading="nearest", vmin=0, vmax=1,
            )
            fig.colorbar(image, ax=ax, label="Herald probability")
            ax.set(
                xlabel="Drive duration (us)", ylabel=coordinate_label,
                title=f"Herald {labels[outcome]}" + (" · pass fraction, not purity" if outcome == 0 else ""),
            )


@dataclass(frozen=True)
class RabiRate:
    """One complete duration-slice cosine fit in MHz, or an explicit failure.

    rate is None when unresolved; reason explains why. uncertainty is the finite
    frequency standard error, not an acceptance threshold. fitted contains model
    values at all saved durations, or None on failure. quality reports the actual
    fit observations/parameters/covariance, including rejected low-rate fits.
    """

    rate: float | None
    uncertainty: float | None
    reason: str | None
    fitted: NDArray[np.float64] | None
    quality: FitQuality | None


@dataclass(frozen=True)
class RabiAnalysis:
    """Same-shot joint return data and one duration-slice fit per coordinate."""

    return_data: ReturnData
    rates: tuple[RabiRate, ...]


def fit_duration_rates(
    data: RabiScanResult, answer: ReturnData
) -> tuple[RabiRate, ...]:
    """Fit complete regular duration slices with the existing undamped cosine.

    A fitted rate below 1/actual saved duration span is unresolved. No automatic
    Nyquist, contrast/sigma, frequency/sigma, chi-square or fidelity gate is used.
    Numerical failure, zero contrast or nonfinite uncertainty produces a reason,
    not a zero rate or optimizer fallback. No incomplete trace is fitted.
    """
    return tuple(
        _fit_duration_rate(data.durations, values[:, 0], complete=bool(complete.all()))
        for values, complete in zip(answer.populations, answer.complete, strict=True)
    )


def _duration_fit_reason(
    durations: NDArray[np.float64], values: NDArray[np.float64], *, complete: bool
) -> str | None:
    if not complete:
        return "Incomplete duration trace"
    if len(durations) <= 4 or np.ptp(durations) == 0:
        return "Need more than four distinct duration samples"
    if not np.allclose(np.diff(durations), durations[1] - durations[0]):
        return "Cosine fit requires regularly spaced durations"
    if np.ptp(values) == 0:
        return "Zero return contrast: Rabi rate undefined"
    return None


def _fit_duration_rate(
    durations: NDArray[np.float64], values: NDArray[np.float64], *, complete: bool
) -> RabiRate:
    reason = _duration_fit_reason(durations, values, complete=complete)
    if reason:
        return RabiRate(None, None, reason, None, None)
    try:
        _, _, _, _, frequency, error, fitted, (parameters, covariance) = fit_rabi(
            durations, values, decay=False
        )
    except (
        ValueError,
        RuntimeError,
        ZeroDivisionError,
        FloatingPointError,
        OverflowError,
    ) as exc:
        return RabiRate(None, None, f"Cosine fit failed: {exc}", None, None)
    quality = compute_fit_quality(
        values,
        fitted,
        dict(
            zip(
                ("offset", "contrast", "frequency", "phase_deg"),
                parameters,
                strict=True,
            )
        ),
        covariance,
    )
    if (
        not np.all(np.isfinite(parameters))
        or not np.all(np.isfinite(covariance))
        or not np.all(np.isfinite(fitted))
        or np.any(np.diag(covariance) < 0)
        or not np.isfinite(error)
    ):
        return RabiRate(
            None, None, "Nonfinite fit or unavailable uncertainty", None, quality
        )
    if parameters[1] == 0 or frequency == 0:
        return RabiRate(
            None,
            None,
            "Degenerate cosine: contrast or frequency is zero",
            None,
            quality,
        )
    if abs(frequency) < 1 / np.ptp(durations):
        return RabiRate(
            None, None, "Rate below 1/duration span: unresolved", fitted, quality
        )
    return RabiRate(abs(frequency), error, None, fitted, quality)


def plot_duration_rates(
    data: RabiScanResult,
    rates: tuple[RabiRate, ...],
    *,
    coordinate_label: str,
    plots: Plots,
) -> None:
    """Plot valid rates with uncertainty and measured/fitted duration traces."""
    _, ax = plots.subplots("rabi_rate", figsize=(10, 6), layout="constrained")
    for coordinate, rate in zip(data.coordinates, rates, strict=True):
        if rate.rate is not None:
            ax.errorbar(
                coordinate, rate.rate, yerr=rate.uncertainty, fmt="o", color="C0"
            )
    reasons = [
        f"{coordinate:g}: {rate.reason}"
        for coordinate, rate in zip(data.coordinates, rates, strict=True)
        if rate.reason
    ]
    if reasons:
        ax.text(
            0.02,
            0.98,
            "\n".join(reasons),
            va="top",
            transform=ax.transAxes,
            fontsize="small",
        )
    ax.set(
        xlabel=coordinate_label,
        ylabel="Rabi rate (MHz)",
        title="Undamped cosine model · rates below 1/duration span excluded",
    )
    _, ax = plots.subplots("rabi_traces", figsize=(10, 6), layout="constrained")
    for coordinate, rate in zip(data.coordinates, rates, strict=True):
        if rate.fitted is not None:
            ax.plot(data.durations, rate.fitted, label=f"{coordinate:g}")
    ax.set(
        xlabel="Drive duration (us)",
        ylabel="Fitted joint P00",
        title="Duration cosine fits · inspect model against measured return curves",
    )
    if ax.lines and len(data.coordinates) <= 25:
        ax.legend(title=coordinate_label, fontsize="x-small", ncols=2)
