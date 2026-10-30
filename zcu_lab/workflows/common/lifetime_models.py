"""Record both lifetime models without choosing the value used for feedback."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import BaseModel
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.analysis.fitting import (
    FitQuality,
    compute_fit_quality,
    decay_time_bounds,
    fit_dual_decay,
    fit_gauss_decay,
    fit_gauss_decay_fringe,
)
from zcu_tools.plotting.plots import Plots

type LifetimeModelName = Literal[
    "exponential", "dual", "exponential_fringe", "gaussian", "gaussian_fringe"
]


class LifetimeModelRecord(BaseModel):
    """One model's diagnostic estimates, never accepted feedback values.

    model names the envelope/fringe model. lifetimes_us/errors_us contain one
    component, or longer then shorter for dual. Nonfinite estimates become null;
    finite discarded estimates remain available for inspection. residual_rms is
    in projected signal units. quality retains normalized residual and optimizer
    diagnostics. bic is n*log(RSS/n) + k*log(n), with k free parameters and the
    same observed samples for both models; zero RSS uses float64 tiny. Missing
    metrics are null. status is recorded, discarded, or failed; reason explains
    discard/failure. primary identifies the existing acceptance model, not a
    BIC winner. A discarded comparison never replaces or rejects the primary.
    """

    model: LifetimeModelName
    primary: bool
    lifetimes_us: tuple[float | None, ...] = ()
    errors_us: tuple[float | None, ...] = ()
    residual_rms: float | None = None
    quality: FitQuality | None = None
    bic: float | None = None
    status: Literal["recorded", "discarded", "failed"] = "recorded"
    reason: str | None = None


@dataclass(frozen=True)
class LifetimeCurve:
    """One numerical model before record sanitization.

    model names the envelope/fringe. lifetimes_us/errors_us hold its components
    in us; fitted is aligned to the caller's times/observations. quality comes
    from its real optimizer parameters/covariance. free_parameters excludes
    deliberately fixed center/phase parameters for BIC. fixed_phase is the
    fringe phase in degrees held fixed by the primary, or None for free phase;
    the alternative uses the same choice.
    """

    model: LifetimeModelName
    lifetimes_us: tuple[float, ...]
    errors_us: tuple[float, ...]
    fitted: NDArray[np.float64]
    quality: FitQuality
    free_parameters: int
    fixed_phase: float | None = None


@dataclass(frozen=True)
class LifetimeComparison:
    """Two serializable model records and available curves on the same delays.

    records keeps primary first, secondary second, including failed attempts.
    curves contains only fitted models; a discarded finite curve is still drawn.
    """

    records: tuple[LifetimeModelRecord, ...]
    curves: tuple[LifetimeCurve, ...]


def _discard_reason(curve: LifetimeCurve, limit_us: float) -> str | None:
    reasons: list[str] = []
    for index, (value, error) in enumerate(
        zip(curve.lifetimes_us, curve.errors_us, strict=True)
    ):
        if not np.isfinite(value) or value <= 0:
            reasons.append(
                f"component {index + 1}: lifetime must be finite and positive"
            )
        if not np.isfinite(error) or error < 0 or error > value:
            reasons.append(
                f"component {index + 1}: error must be finite, nonnegative and <= lifetime"
            )
        if value > limit_us:
            reasons.append(
                f"component {index + 1}: lifetime {value:.6g} us exceeds window limit {limit_us:.6g} us"
            )
    reasons.extend(f"{issue.path}: {issue.reason}" for issue in curve.quality.invalid)
    return "; ".join(reasons) or None


def _record(
    curve: LifetimeCurve,
    observed: NDArray[np.float64],
    *,
    primary: bool,
    limit_us: float,
) -> LifetimeModelRecord:
    residual = observed - curve.fitted
    rss = float(np.sum(residual**2))
    rms, bic = None, None
    if np.isfinite(rss):
        rms = float(np.sqrt(rss / len(observed)))
        bic = float(
            len(observed) * np.log(max(rss / len(observed), np.finfo(float).tiny))
            + curve.free_parameters * np.log(len(observed))
        )
    reason = _discard_reason(curve, limit_us)
    return LifetimeModelRecord(
        model=curve.model,
        primary=primary,
        lifetimes_us=tuple(
            float(x) if np.isfinite(x) else None for x in curve.lifetimes_us
        ),
        errors_us=tuple(float(x) if np.isfinite(x) else None for x in curve.errors_us),
        residual_rms=rms,
        quality=curve.quality,
        bic=bic,
        status="discarded" if reason else "recorded",
        reason=reason,
    )


def _secondary(
    times: NDArray[np.float64], observed: NDArray[np.float64], primary: LifetimeCurve
) -> LifetimeCurve:
    if primary.model == "exponential":
        t, err, tb, errb, fitted, (params, covariance) = fit_dual_decay(times, observed)
        names = ("y0", "yscale1", "decay_time1", "yscale2", "decay_time2")
        model, values, errors, free = "dual", (t, tb), (err, errb), 5
        fixed = ()
    elif primary.model == "exponential_fringe":
        t, err, _, _, fitted, (params, covariance) = fit_gauss_decay_fringe(
            times,
            observed,
            fixedparams=(None, None, None, primary.fixed_phase, None)
            if primary.fixed_phase is not None
            else None,
        )
        names = ("y0", "yscale", "freq", "phase", "decay_time")
        model, values, errors, free = (
            "gaussian_fringe",
            (t,),
            (err,),
            primary.free_parameters,
        )
        fixed = ("phase",) if primary.fixed_phase is not None else ()
    else:
        raise ValueError("Primary must be exponential or exponential_fringe")
    quality = compute_fit_quality(
        observed,
        fitted,
        dict(zip(names, map(float, params), strict=True)),
        covariance,
        decay_bounds=decay_time_bounds(times),
        fixed_parameters=fixed,
    )
    return LifetimeCurve(model, values, errors, fitted, quality, free)


def gaussian_comparison(
    times: NDArray[np.float64], observed: NDArray[np.float64]
) -> LifetimeCurve:
    """Fit a zero-centered, non-fringe Gaussian envelope; return its curve/quality.

    times are the primary model's us delays, observed its projected signal.
    Fix Gaussian center to zero via fit_gauss_decay's fixedparams. T is sqrt(2)
    times sigma, so the envelope is baseline + A*exp(-(t/T)**2). Numeric fit
    failures propagate to the comparison caller; no acceptance occurs.
    """
    t, err, fitted, (params, covariance) = fit_gauss_decay(
        times, observed, fixedparams=(None, None, 0.0, None)
    )
    quality = compute_fit_quality(
        observed,
        fitted,
        dict(zip(("y0", "yscale", "center", "sigma"), map(float, params), strict=True)),
        covariance,
        fixed_parameters=("center",),
    )
    return LifetimeCurve("gaussian", (t,), (err,), fitted, quality, 3)


def compare_lifetime_models(
    times: NDArray[np.float64],
    observed: NDArray[np.float64],
    primary: LifetimeCurve,
    *,
    max_window_factor: float,
    t1: bool = False,
) -> LifetimeComparison:
    """Keep the supplied primary and fit the alternative on identical samples.

    times are acquired us delays without shifting the origin, observed the
    primary projection. t1 selects dual decay; otherwise choose Gaussian decay
    or Gaussian fringe to match the primary. max_window_factor is positive and
    finite; diagnostic lifetimes above factor*max(times) are discarded. Numeric
    secondary failures, including numerical seed/bound ValueErrors on the
    already fitted trace, become a failed record, never a primary failure.
    Invalid max_window_factor raises ValueError. No acceptance or feedback is changed.
    """
    if not np.isfinite(max_window_factor) or max_window_factor <= 0:
        raise ValueError("max_window_factor must be finite and positive")
    limit = max_window_factor * float(np.max(times))
    first = _record(primary, observed, primary=True, limit_us=limit)
    model: LifetimeModelName = (
        "dual"
        if t1
        else "gaussian_fringe"
        if primary.model == "exponential_fringe"
        else "gaussian"
    )
    try:
        second = (
            _secondary(times, observed, primary)
            if t1 or primary.model == "exponential_fringe"
            else gaussian_comparison(times, observed)
        )
    except (
        AnalysisFailed,
        ValueError,
        RuntimeError,
        ArithmeticError,
        np.linalg.LinAlgError,
    ) as exc:
        return LifetimeComparison(
            (
                first,
                LifetimeModelRecord(
                    model=model, primary=False, status="failed", reason=str(exc)
                ),
            ),
            (primary,),
        )
    return LifetimeComparison(
        (first, _record(second, observed, primary=False, limit_us=limit)),
        (primary, second),
    )


def show_lifetime_comparison(
    plots: Plots,
    name: str,
    times: NDArray[np.float64],
    observed: NDArray[np.float64],
    comparison: LifetimeComparison,
) -> None:
    """Replace a workflow model plot with observations and both diagnostic fits.

    plots owns the reusable named figure; times/observed are the common fit
    samples in us/projected units. Does not save PNG or modify accepted trends.
    """
    if name not in plots:
        plots.subplots(name)

    def draw(figure: Figure) -> None:
        ax = figure.axes[0]
        ax.clear()
        ax.plot(times, observed, ".", label="observed")
        draw_lifetime_comparison(ax, times, comparison)
        ax.set(xlabel="Delay (us)", ylabel="Projected signal")
        ax.legend()

    plots.draw(name, draw)


def draw_lifetime_comparison(
    ax: Axes, times: NDArray[np.float64], comparison: LifetimeComparison
) -> None:
    """Overlay each available curve with model, us estimates and discard status.

    ax already contains observations at times; caller owns legend and saving.
    Failed alternatives are labeled without inventing a fitted curve.
    """
    for record in comparison.records:
        curve = next((c for c in comparison.curves if c.model == record.model), None)
        label = f"{record.model} ({record.status})"
        if curve is not None:
            values = ", ".join(
                f"{t:.3g}±{e:.2g}"
                for t, e in zip(curve.lifetimes_us, curve.errors_us, strict=True)
            )
            ax.plot(
                times,
                curve.fitted,
                "-" if record.primary else "--",
                label=f"{label}: {values} us",
            )
        else:
            ax.plot([], [], "--", label=label)
