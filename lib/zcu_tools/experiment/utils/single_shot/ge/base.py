from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from scipy.special import erfc
from typing_extensions import (
    TypedDict,  # closed/extra_items (PEP 728) not in stdlib 3.13
)

from zcu_tools.analysis.fitting import FitQuality, compute_fit_quality
from zcu_tools.analysis.fitting.singleshot import (
    calc_population_pdf,
    fit_singleshot,
    fit_singleshot_p0,
    gauss_func,
)
from zcu_tools.plotting.plots import Plots

from .density import plot_ge_iq_density


class GENumericalError(RuntimeError):
    """GE fitting or classification failed numerically; __cause__ is the original error.

    Callers may isolate this failure per dataset. Plotting errors are not
    represented by this type.
    """


def rotate(
    I_orig: NDArray[np.float64], Q_orig: NDArray[np.float64], theta: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    I_new = I_orig * np.cos(theta) - Q_orig * np.sin(theta)
    Q_new = I_orig * np.sin(theta) + Q_orig * np.cos(theta)
    return I_new, Q_new


def scatter_ge_plot(
    ax: Axes,
    Ige: tuple[NDArray[np.float64], NDArray[np.float64]],
    Qge: tuple[NDArray[np.float64], NDArray[np.float64]],
    title: str | None = None,
    max_points: int = 10000,
) -> None:
    Ig, Ie = Ige
    Qg, Qe = Qge

    g_center = np.median(Ig)
    e_center = np.median(Ie)

    rand_gen = np.random.default_rng(42)
    downsample_idx = rand_gen.choice(
        len(Ig), size=min(max_points, len(Ig)), replace=False
    )
    Ig, Qg = np.take(Ig, downsample_idx), np.take(Qg, downsample_idx)
    Ie, Qe = np.take(Ie, downsample_idx), np.take(Qe, downsample_idx)

    # sort Ig, Qg by Ig from high to low
    sort_g_idx = np.argsort(Ig)
    sort_e_idx = np.argsort(Ie)
    if g_center < e_center:
        sort_g_idx = sort_g_idx[::-1]
    else:
        sort_e_idx = sort_e_idx[::-1]
    Ig, Qg = Ig.take(sort_g_idx), Qg.take(sort_g_idx)
    Ie, Qe = Ie.take(sort_e_idx), Qe.take(sort_e_idx)

    # Mix the data to avoid one color covering the other
    I_all = np.concatenate((Ig, Ie))
    Q_all = np.concatenate((Qg, Qe))
    c_all = np.array(["b"] * len(Ig) + ["r"] * len(Ie))

    p = np.arange(len(I_all)).reshape(2, -1).T.flatten()
    ax.scatter(
        I_all.take(p),
        Q_all.take(p),
        c=c_all.take(p),
        marker=".",
        edgecolor="None",
        alpha=0.25,
    )
    # Plot empty data for legend
    ax.scatter([], [], label="g", color="b")
    ax.scatter([], [], label="e", color="r")
    ax.set_xlabel("I [ADC levels]")
    ax.set_ylabel("Q [ADC levels]")
    ax.legend(loc="upper right")
    if title is not None:
        ax.set_title(title, fontsize=14)
    ax.axis("equal")


def hist(
    Ig: NDArray[np.float64],
    Ie: NDArray[np.float64],
    numbins: int | str = "auto",
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    I_tot = np.concatenate((Ie, Ig))
    xlims = (np.min(I_tot).item(), np.max(I_tot).item())
    bins = np.histogram_bin_edges(I_tot, bins=numbins)
    ng, *_ = np.histogram(Ig, bins=bins, range=xlims)
    ne, *_ = np.histogram(Ie, bins=bins, range=xlims)
    ng = ng / np.sum(ng)
    ne = ne / np.sum(ne)
    if ax is not None:
        xs = 0.5 * (bins[1:] + bins[:-1])
        for weights, color, label in ((ng, "b", "g"), (ne, "r", "e")):
            ax.hist(
                xs,
                bins=bins.tolist(),
                range=xlims,
                alpha=0.5,
                color=color,
                weights=weights,
                label=label,
            )
        ax.set_ylabel("Counts", fontsize=14)
        ax.set_xlabel("I [ADC levels]", fontsize=14)
        ax.legend(loc="upper right")
        if title is not None:
            ax.set_title(title, fontsize=14)
    return ng, ne, bins


def fidelity_func(tp: float, tn: float, fp: float, fn: float) -> float:
    # this method calculates fidelity as (Ngg+Nee)/N = Ngg/N + Nee/N=(0.5N-Nge)/N + (0.5N-Neg)/N = 1-(Nge+Neg)/N
    if (tp + fn) > (tn + fp):
        return (tp + fn) / (tp + tn + fp + fn)
    return (tn + fp) / (tp + tn + fp + fn)


def calc_fidelity(
    ng: NDArray[np.float64], ne: NDArray[np.float64], bins: NDArray[np.float64]
) -> tuple[float, float]:
    """Return empirical preparation-assignment fidelity and its threshold edge.

    ng/ne are normalized histogram masses, shape (N,), for equally weighted
    preparations. bins contains their common increasing N+1 edges in the same
    units as the projected IQ. Select the edge maximizing cumulative contrast.
    Score both possible label orientations and return the larger correct fraction;
    swapping preparation rows or reflecting the axis preserves that score.
    The threshold is bins[split]: histogram entries [:split] lie below it,
    entries [split:] above.
    This score includes preparation mixing and is not pure-state readout fidelity.
    """
    cum_ng, cum_ne = np.cumsum(ng), np.cumsum(ne)
    contrast = np.abs(2 * (cum_ng - cum_ne) / (ng.sum() + ne.sum()))
    tind = contrast.argmax()

    split = tind + 1
    tp, fp = ng[split:].sum(), ne[split:].sum()
    tn, fn = ng[:split].sum(), ne[:split].sum()
    fid = fidelity_func(tp, tn, fp, fn)

    return fid, float(bins[split])


def calc_ideal_fidelity(sg: float, se: float, s: float) -> float:
    # The distance between the centers of the two Gaussians
    distance = np.abs(sg - se)

    if np.isclose(s, 0):
        return 1.0 if not np.isclose(distance, 0) else 0.5

    p_error = 0.5 * erfc(distance / (2 * np.sqrt(2) * s))

    tp = p_error
    fn = p_error
    tn = 1 - p_error
    fp = 1 - p_error

    return fidelity_func(tp=tp, tn=tn, fp=fp, fn=fn)


class GE_FitResult(TypedDict, closed=True):
    """GE calibration and diagnostics.

    length_ratio is the dimensionless joint readout length / T1. It is None
    when the joint covariance cannot estimate its relative standard error or
    that error exceeds GEFitOptions.length_ratio_max_relative_error. A user-fixed
    length_ratio is returned unchanged. ge_params and length_ratio_g/e retain
    their fitted values regardless of identifiability.
    """

    length_ratio: float | None
    ge_params: tuple[float, float, float, float, float, float, float]
    p0_gg: float
    p0_ge: float
    p0_eg: float
    p0_ee: float
    s: float
    length_ratio_g: float
    length_ratio_e: float
    theta: float
    threshold: float
    g_center: complex
    e_center: complex
    fit_quality: dict[str, FitQuality]


@pydantic_dataclass(frozen=True)
class GEFitOptions:
    """Shared paired-histogram fit and display controls.

    numbins defaults to NumPy's automatic histogram bins. length_ratio,
    init_p0_g, init_p0_e and avg_p optionally fix the joint model's readout/T1
    ratio, preparation populations and average population; None fits each value.
    logscale defaults to linear PDF axes. align_t1 defaults to sharing the joint
    readout/T1 ratio in the two population refinements.
    length_ratio_max_relative_error is a dimensionless, finite, positive
    threshold on the joint relative standard error sigma/|p|. A missing error
    estimate or sigma/|p| above the threshold makes the reported length_ratio
    None; a fixed length_ratio bypasses this check. The default 1.0 corresponds
    to fit_tools' sigma(ln p) > 1 weak-parameter rule. Nonpositive or nonfinite
    thresholds raise ValidationError at construction.
    raw_iq_diagnostic shows raw IQ, fitted SNR with first-order standard error,
    and joint fit quality. False retains the default rotated IQ figure. This
    option affects only plotted fits.
    """

    numbins: int | str = "auto"
    length_ratio: float | None = None
    init_p0_g: float | None = None
    init_p0_e: float | None = None
    avg_p: float | None = None
    logscale: bool = False
    align_t1: bool = True
    length_ratio_max_relative_error: float = Field(
        default=1.0, gt=0.0, allow_inf_nan=False
    )
    raw_iq_diagnostic: bool = False


def fitting_ge_and_plot(
    signals: NDArray[np.complex128],
    classify_func: Callable[
        [
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        dict[str, Any],
    ],
    *,
    options: GEFitOptions | None = None,
    plots: Plots | None = None,
    figure_name: str = "fit",
) -> tuple[float, NDArray[np.float64], GE_FitResult]:
    """Fit paired raw IQ (2,N) and publish diagnostics under figure_name.

    IQ uses ADC units per sample. Existing histogram/population and quality
    policy is unchanged by the name, which defaults to fit. plots owns the
    figure; caller finishes/releases the operation. With plots=None, return the
    same calibration and quality without creating diagnostics. With plots supplied,
    empty or conflicting names raise from Plots. Rotation/classification, histogram fitting, population /
    center validation and fidelity calculations translate ValueError,
    RuntimeError, ArithmeticError and numpy.linalg.LinAlgError into
    GENumericalError with the original cause. Plotting is outside that boundary;
    its errors propagate unchanged. The input must have paired rows (2,N).

    The returned length_ratio is the dimensionless joint readout length / T1.
    Fixed options.length_ratio is returned unchanged. For a free fit, an
    unavailable joint relative standard error or sigma/|p| greater than
    options.length_ratio_max_relative_error returns None and labels the plot
    "not identifiable". Other fitted parameters and quality diagnostics are
    unchanged by this reporting threshold.
    """
    options = GEFitOptions() if options is None else options
    (Ig, Ie), (Qg, Qe) = signals.real, signals.imag

    try:
        # Calculate the angle of rotation
        theta = classify_func(Ig, Qg, Ie, Qe)["theta"]
        theta = (theta + np.pi / 2) % np.pi - np.pi / 2

        # Rotate the IQ data
        (Ig, Qg), (Ie, Qe) = rotate(Ig, Qg, theta), rotate(Ie, Qe, theta)

        g_pdfs, e_pdfs, bins = hist(Ig, Ie, options.numbins)

        xs = 0.5 * (bins[:-1] + bins[1:])

        ge_params, joint_covariance = fit_singleshot(
            xs,
            g_pdfs,
            e_pdfs,
            fixedparams=[
                None,
                None,
                None,
                options.init_p0_g,
                options.init_p0_e,
                options.avg_p,
                options.length_ratio,
            ],
        )
        sg, se, s, joint_p0_gg, joint_p0_ge, p_avg, length_ratio = ge_params
        (p0_gg, p0_ge, l_ratio_g), ground_covariance = fit_singleshot_p0(
            xs,
            g_pdfs,
            joint_p0_gg,
            joint_p0_ge,
            ge_params=ge_params,
            fit_length_ratio=not options.align_t1,
        )
        (p0_eg, p0_ee, l_ratio_e), excited_covariance = fit_singleshot_p0(
            xs,
            e_pdfs,
            joint_p0_ge,
            joint_p0_gg,
            ge_params=ge_params,
            fit_length_ratio=not options.align_t1,
        )
        populations = np.array([[p0_gg, p0_ge], [p0_eg, p0_ee]])
        if (
            not np.isfinite(populations).all()
            or np.any(populations < 0)
            or np.any(populations.sum(axis=1) > 1.0 + 1e-12)
        ):
            raise ValueError(
                "GE fit returned invalid populations; calibration is unavailable"
            )

        fit_g_pdfs = calc_population_pdf(xs, sg, se, s, p0_gg, p0_ge, p_avg, l_ratio_g)
        fit_e_pdfs = calc_population_pdf(xs, sg, se, s, p0_eg, p0_ee, p_avg, l_ratio_e)
        # Joint parameters describe the original paired histogram model, not the
        # independently refined populations used for the final plotted PDFs.
        joint_g_pdfs = calc_population_pdf(xs, *ge_params)
        joint_e_pdfs = calc_population_pdf(
            xs, sg, se, s, joint_p0_ge, joint_p0_gg, p_avg, length_ratio
        )
        fit_quality = {
            "joint": compute_fit_quality(
                np.concatenate([g_pdfs, e_pdfs]),
                np.concatenate([joint_g_pdfs, joint_e_pdfs]),
                dict(
                    zip(
                        ("sg", "se", "s", "p0_g", "p0_e", "p_avg", "length_ratio"),
                        ge_params,
                        strict=True,
                    )
                ),
                joint_covariance,
            ),
            "ground": compute_fit_quality(
                g_pdfs,
                fit_g_pdfs,
                {"p0_g": p0_gg, "p0_e": p0_ge, "length_ratio": l_ratio_g},
                ground_covariance,
            ),
            "excited": compute_fit_quality(
                e_pdfs,
                fit_e_pdfs,
                {"p0_g": p0_eg, "p0_e": p0_ee, "length_ratio": l_ratio_e},
                excited_covariance,
            ),
        }
        identifiable_length_ratio = _identifiable_length_ratio(
            length_ratio, fit_quality["joint"], options
        )
        rotated_g_center = _estimate_rotated_center(Ig, Qg, sg, s)
        rotated_e_center = _estimate_rotated_center(Ie, Qe, se, s)

        if (
            not np.isfinite([rotated_g_center, rotated_e_center, s]).all()
            or s <= 0
            or np.isclose(rotated_g_center, rotated_e_center)
        ):
            raise ValueError(
                "GE fit returned invalid centers or width; calibration is unavailable"
            )
        fid, threshold = calc_fidelity(g_pdfs, e_pdfs, bins)
        ideal_fid = calc_ideal_fidelity(sg, se, s)

    except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
        raise GENumericalError(str(exc)) from exc

    centers = (
        rotated_g_center * np.exp(-1j * theta),
        rotated_e_center * np.exp(-1j * theta),
    )
    if plots is not None:
        fig, raw_axes = plots.subplots(
            num=figure_name, nrows=2, ncols=2, figsize=(8, 8)
        )
        # tolerated(cast): Plots keeps a native 2x2 object array, not typed Axes rows.
        axs = cast("tuple[tuple[Axes, Axes], tuple[Axes, Axes]]", raw_axes)
        annotated = options.raw_iq_diagnostic
        iq, ground = axs[0]
        combined, excited = axs[1]
        if annotated:
            _plot_raw_ge_iq(iq, signals, centers)
        else:
            scatter_ge_plot(iq, (Ig, Ie), (Qg, Qe), "Rotated")
            _mark_rotated_centers(iq, (rotated_g_center, rotated_e_center))
            iq.set_xlim(np.min(bins), np.max(bins))
            iq.axvline(threshold, color="0.2", linestyle="--")

        _plot_ge_histograms(
            (combined, ground, excited),
            bins,
            ge_params,
            (
                _PopulationHistogram(g_pdfs, fit_g_pdfs, p0_gg, p0_ge, True),
                _PopulationHistogram(e_pdfs, fit_e_pdfs, p0_eg, p0_ee, False),
            ),
            annotated=annotated,
        )

        _format_ge_fit_axes(
            (combined, ground, excited),
            threshold,
            g_pdfs,
            e_pdfs,
            logscale=options.logscale,
        )

        length_title = (
            f"{length_ratio:.1f}"
            if options.align_t1
            else f"{l_ratio_g:.1f} / {l_ratio_e:.1f}"
        )
        title = _diagnostic_title(
            length_title if identifiable_length_ratio is not None else None,
            fit_quality["joint"] if annotated else None,
            (sg, se, s),
            joint_covariance,
        )
        _layout_ge_diagnostic(
            fig,
            (iq, combined, ground, excited),
            title,
            (fid, ideal_fid),
            annotated=annotated,
        )

    return (
        fid,
        populations,
        GE_FitResult(
            length_ratio=identifiable_length_ratio,
            ge_params=ge_params,
            p0_gg=p0_gg,
            p0_ge=p0_ge,
            p0_eg=p0_eg,
            p0_ee=p0_ee,
            s=s,
            length_ratio_g=l_ratio_g,
            length_ratio_e=l_ratio_e,
            theta=theta,
            threshold=threshold,
            g_center=centers[0],
            e_center=centers[1],
            fit_quality=fit_quality,
        ),
    )


def _plot_raw_ge_iq(
    ax: Axes, signals: NDArray[np.complex128], centers: tuple[complex, complex]
) -> None:
    plot_ge_iq_density(ax, signals[0], signals[1])
    ax.scatter([], [], marker="s", color="b", label="g prep")
    ax.scatter([], [], marker="s", color="r", label="e prep")
    ax.scatter(
        [z.real for z in centers],
        [z.imag for z in centers],
        color=["navy", "darkred"],
        marker="x",
        s=26,
        linewidths=1.5,
        label="Fitted centers",
    )
    ax.set(xlabel="I (a.u.)", ylabel="Q (a.u.)")
    ax.set_title("Raw IQ with fitted g/e centers", fontsize=10, pad=23)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        ncols=3,
        fontsize=10,
        handlelength=1,
        handletextpad=0.35,
        columnspacing=0.8,
        borderpad=0.25,
        frameon=False,
    )
    ax.grid(False)


def _mark_rotated_centers(ax: Axes, centers: tuple[complex, complex]) -> None:
    for center, face, color in zip(centers, ("b", "r"), ("r", "b"), strict=True):
        ax.plot(
            center.real,
            center.imag,
            markerfacecolor=face,
            color=color,
            linestyle=":",
            marker="o",
            markersize=5,
        )


def _plot_ge_histograms(
    axes: tuple[Axes, Axes, Axes],
    bins: NDArray[np.float64],
    ge_params: tuple[float, float, float, float, float, float, float],
    histograms: tuple[_PopulationHistogram, _PopulationHistogram],
    *,
    annotated: bool,
) -> None:
    combined, ground, excited = axes
    xs = 0.5 * (bins[:-1] + bins[1:])
    for panel, histogram, color, label in zip(
        (ground, excited), histograms, ("b", "r"), ("g", "e"), strict=True
    ):
        for ax in (combined, panel):
            ax.hist(
                xs,
                bins=bins.tolist(),
                weights=histogram.observed,
                color=color,
                label=label if ax is combined else None,
                alpha=0.5,
                edgecolor="none" if annotated else None,
                linewidth=0 if annotated else None,
            )
        _plot_population_histogram(
            panel, xs, ge_params, histogram, show_ideal=ge_params[-1] != 0.0
        )
    combined.legend(
        loc="lower center" if annotated else "upper right",
        bbox_to_anchor=(0.5, 1) if annotated else None,
        ncols=2 if annotated else 1,
        fontsize=10 if annotated else None,
        frameon=False if annotated else None,
    )
    combined.plot(xs, histograms[0].fitted, "b-", label="g")
    combined.plot(xs, histograms[1].fitted, "r-", label="e")
    combined.set_ylabel(
        "Fraction per bin" if annotated else "Counts", fontsize=10 if annotated else 14
    )


def _layout_ge_diagnostic(
    fig: Figure,
    axes: tuple[Axes, Axes, Axes, Axes],
    title: str,
    scores: tuple[float, float],
    *,
    annotated: bool,
) -> None:
    iq, combined, ground, excited = axes
    prepared, ideal = scores
    if annotated:
        fig.suptitle(title, fontsize=10)
        combined.set_title(
            f"Prepared assignment: {prepared:.1%}\nIdeal model: {ideal:.1%}",
            fontsize=10,
            pad=23,
        )
        for ax in axes:
            ax.tick_params(labelsize=10)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=2))
        iq.xaxis.label.set_fontsize(10)
        iq.yaxis.label.set_fontsize(10)
        for histogram in (combined, ground, excited):
            histogram.set_xlabel("Projected I (a.u.)", fontsize=10)
        for panel, prep in ((ground, "g"), (excited, "e")):
            panel.set_title(f"{prep} prep · {panel.get_title()}", fontsize=10, pad=40)
            panel.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 1),
                ncols=3,
                fontsize=10,
                columnspacing=1,
                handlelength=1.5,
                frameon=False,
            )
        fig.set_layout_engine("constrained", hspace=0.08)
    else:
        combined.set_xlabel("I [ADC levels]", fontsize=14)
        combined.set_title(
            f"Prepared assignment: {prepared:.1%}; ideal model: {ideal:.1%}",
            fontsize=12,
        )
        fig.suptitle(title)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.25, wspace=0.15)


def _diagnostic_title(
    length_title: str | None,
    joint_quality: FitQuality | None,
    centers_and_width: tuple[float, float, float],
    joint_covariance: NDArray[np.float64],
) -> str:
    title = (
        "Readout length: not identifiable"
        if length_title is None
        else f"Readout length = {length_title} " + r"$T_1$"
    )
    if joint_quality is None:
        return title

    snr = _fitted_snr_title(*centers_and_width, joint_covariance[:3, :3])
    if joint_quality.r2 is not None:
        quality = f"R² {joint_quality.r2:.3g}"
    elif joint_quality.normalized_residual_rms is not None:
        quality = f"normalized RMS {joint_quality.normalized_residual_rms:.3g}"
    else:
        quality = "R² unavailable"
    return f"{title}\n{snr} · Joint {quality}"


def _fitted_snr_title(
    sg: float, se: float, width: float, covariance: NDArray[np.float64]
) -> str:
    separation = se - sg
    ratio = abs(separation) / width
    if not math.isfinite(ratio):
        return "Fitted SNR = unavailable"

    title = f"Fitted SNR = {ratio:.3g} ± "
    if (
        separation == 0.0
        or not np.isfinite(covariance).all()
        or np.any(covariance.diagonal() < 0.0)
    ):
        return title + "unavailable"

    # Propagate the full joint covariance in physical (sg, se, width) order.
    slope = math.copysign(1.0, separation) / width
    gradient = np.array([-slope, slope, -ratio / width])
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        variance = float(gradient @ covariance @ gradient)
    if not math.isfinite(variance) or variance <= 0.0:
        return title + "unavailable"
    return title + f"{math.sqrt(variance):.3g}"


def _identifiable_length_ratio(
    fitted_ratio: float, joint_quality: FitQuality, options: GEFitOptions
) -> float | None:
    """Preserve fixed ratios; apply the joint relative-error threshold to free fits."""
    if options.length_ratio is not None:
        return options.length_ratio
    relative_error = joint_quality.relative_parameter_errors["length_ratio"]
    if (
        relative_error is None
        or relative_error > options.length_ratio_max_relative_error
    ):
        return None
    return fitted_ratio


def _estimate_rotated_center(
    values: NDArray[np.float64],
    quadratures: NDArray[np.float64],
    center: float,
    width: float,
) -> complex:
    """Estimate the orthogonal median using IQ samples within two fitted widths."""
    mask = (values > center - 2 * width) & (values < center + 2 * width)
    if not mask.any():
        raise ValueError("GE fit centers have no supporting IQ samples")
    return complex(center + 1j * np.median(quadratures[mask]))


@dataclass(frozen=True)
class _PopulationHistogram:
    """One prepared-state histogram and its independently refined populations."""

    observed: NDArray[np.float64]
    fitted: NDArray[np.float64]
    ground_population: float
    excited_population: float
    prepared_ground: bool


def _plot_population_histogram(
    ax: Axes,
    xs: NDArray[np.float64],
    ge_params: tuple[float, float, float, float, float, float, float],
    histogram: _PopulationHistogram,
    *,
    show_ideal: bool,
) -> None:
    """Plot total, Gaussian components and residual for one prepared state."""
    sg, se, width, *_ = ge_params
    pg, pe = histogram.ground_population, histogram.excited_population
    ground = pg * gauss_func(xs, sg, width)
    excited = pe * gauss_func(xs, se, width)
    residual = np.clip(histogram.observed - histogram.fitted, 0.0, None)
    ax.plot(xs, histogram.fitted, "k-", label="total")
    if show_ideal:
        ax.plot(xs, ground + excited, "k--", alpha=0.5, label="ideal total")
    if histogram.prepared_ground:
        ax.plot(xs, ground, "b-", alpha=0.4, label="ground")
        ax.plot(xs, excited, "r--", alpha=0.4, label="excited")
    else:
        ax.plot(xs, excited, "r-", alpha=0.4, label="excited")
        ax.plot(xs, ground, "b--", alpha=0.4, label="ground")
    ax.plot(xs, residual, "g--", alpha=0.5, label="other")
    ax.set_title(f"{pg:.1%} / {pe:.1%} / {max(0.0, 1 - pg - pe):.1%}", fontsize=14)
    ax.legend()


def _format_ge_fit_axes(
    axes: tuple[Axes, Axes, Axes],
    threshold: float,
    g_pdfs: NDArray[np.float64],
    e_pdfs: NDArray[np.float64],
    *,
    logscale: bool,
) -> None:
    """Apply the shared threshold and optional histogram logarithmic limits."""
    for ax in axes:
        ax.axvline(threshold, color="0.2", linestyle="--")

    if logscale:
        y_max, y_min = 1.5 * np.max([g_pdfs, e_pdfs]), np.min([g_pdfs, e_pdfs]) + 1e-4
        for ax in axes:
            ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)
