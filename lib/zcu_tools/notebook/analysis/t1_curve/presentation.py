"""Notebook T1 figures, dense plotting curves and parameter display."""

from __future__ import annotations

import os
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.simulate import flux2value
from zcu_tools.simulate.fluxonium import (
    calculate_eff_t1_vs_flux_fast,
)

from .base import (
    format_parameter,
    omega_mask,
    plot_eff_t1_with_sample,
    plot_Q_vs_omega,
    plot_sample_t1,
    plot_t1_vs_elements,
)
from .fit import (
    T1FitResult,
)
from .models import (
    CURVE_LABELS,
    NOISE_CHANNELS,
    MechanismName,
    PurcellEffectParams,
    T1ChannelAnalysis,
    T1CurveContext,
    T1MechanismProbe,
    T1PreparedData,
)
from .purcell import (
    calculate_purcell_t1_limit,
    combine_t1_limits,
)

_Q_LABELS: dict[MechanismName, str] = {
    "capacitive": r"$Q_{cap}$",
    "quasiparticle": r"$Q_{qp}=1/x_{qp}$",
    "inductive": r"$Q_{ind}$",
}


def plot_purcell_Temp_upper_bound(
    data: T1PreparedData,
    purcell: PurcellEffectParams,
    *,
    Temp: float,
    t_flux_count: int = 1000,
    flux_range: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Calculate and plot a dense Purcell limit at Temp in K over sample T1.

    data supplies prepared normalized flux and ns T1/errors; purcell supplies
    GHz resonator inputs. Use t_flux_count points across flux_range, defaulting
    to the analysis window. Return a new Figure/Axes without showing or saving.
    Fewer than two points or invalid Purcell calculation inputs raise ValueError.
    The dense curve is calculated before creating the figure.
    """
    if t_flux_count < 2:
        raise ValueError("t_flux_count must be at least 2")
    resolved_flux_range = flux_range or data.analysis_flux_range
    t_fluxs = np.linspace(
        resolved_flux_range[0], resolved_flux_range[1], t_flux_count, dtype=np.float64
    )
    purcell_T1s = calculate_purcell_t1_limit(
        data.calibration.context,
        t_fluxs,
        purcell,
        Temp=Temp,
    )

    fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 4))
    sample = data.sample
    ax.errorbar(
        sample.fluxs,
        sample.T1_ns,
        yerr=sample.T1err_ns,
        fmt=".",
        color="tab:blue",
        label="T1 samples",
        zorder=3,
    )
    if len(data.fit.fluxs) != len(sample.fluxs) or not np.array_equal(
        data.fit.fluxs, sample.fluxs
    ):
        ax.scatter(
            data.fit.fluxs,
            data.fit.T1_ns,
            color="tab:blue",
            s=18,
            label="fit points",
            zorder=4,
        )
    ax.plot(
        t_fluxs,
        purcell_T1s,
        color="tab:red",
        linestyle="-.",
        linewidth=2.0,
        label=f"Purcell upper @ {Temp * 1e3:.2f} mK",
        zorder=2,
    )
    ax.set_xlim(*resolved_flux_range)
    ax.set_xlabel(r"Flux quanta ($\Phi_{ext}/\Phi_0$)")
    ax.set_ylabel(r"$T_1$ (ns)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize="small")
    return fig, ax


def plot_t1_curve_data(data: T1PreparedData) -> tuple[Figure, Axes]:
    """Plot data.sample T1/errors in ns against normalized flux.

    Use the calibration alignment for the secondary device coordinate. Return
    a new Figure/Axes without showing, saving or closing it. No positive finite
    sample T1 raises ValueError in the base plotter.
    """
    context = data.calibration.context
    return plot_sample_t1(
        flux2value(data.sample.fluxs, context.flux_half, context.flux_period),
        data.sample.T1_ns,
        data.sample.T1err_ns,
        context.flux_half,
        context.flux_period,
    )


def plot_t1_flux_calibration(data: T1PreparedData) -> tuple[Figure, Axes]:
    """Plot raw, f01-corrected and aligned sample flux at f01 heights in MHz.

    data supplies prepared row provenance and the dimensionless analysis
    window. Return a new Figure/Axes with correction categories and branch
    segments, without showing or saving. Empty sample data raises ValueError.
    """
    sample = data.sample
    if len(sample.fluxs) == 0:
        raise ValueError("No T1 samples are available for flux calibration plotting")

    raw_fluxs = np.asarray(sample.raw_fluxs, dtype=np.float64)
    corrected_fluxs = raw_fluxs + np.asarray(sample.flux_corrections, dtype=np.float64)
    aligned_fluxs = np.asarray(sample.fluxs, dtype=np.float64)
    corrected_mask = sample.f01_correction_applied
    sources = np.asarray(sample.flux_sources, dtype=object)
    reasons = np.asarray(sample.correction_skipped_reason, dtype=object)
    not_corrected = ~corrected_mask
    explicit_mask = not_corrected & (sources == "explicit")
    missing_mask = not_corrected & (reasons == "missing_f01") & ~sample.f01_measured
    disabled_mask = not_corrected & (reasons == "disabled")
    rejected_mask = not_corrected & ~explicit_mask & ~missing_mask & ~disabled_mask
    f01_mhz = sample.f01_mhz
    fig_height = float(np.clip(1.8 + 0.18 * len(aligned_fluxs), 3.0, 8.0))
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    lower, upper = data.analysis_flux_range
    ax.axvspan(lower, upper, color="tab:green", alpha=0.08, label="analysis window")

    for f01_freq_mhz, raw_flux, corrected_flux, aligned_flux, is_corrected in zip(
        f01_mhz,
        raw_fluxs,
        corrected_fluxs,
        aligned_fluxs,
        corrected_mask,
        strict=True,
    ):
        line_color = "tab:blue" if is_corrected else "tab:gray"
        # Stage 1: raw -> f01-corrected (pre-alignment).
        ax.plot(
            [raw_flux, corrected_flux],
            [f01_freq_mhz, f01_freq_mhz],
            color=line_color,
            alpha=0.55,
            linewidth=1.0,
        )
        # Stage 2: f01-corrected -> integer-aligned (final analysis flux).
        ax.plot(
            [corrected_flux, aligned_flux],
            [f01_freq_mhz, f01_freq_mhz],
            color="tab:green",
            alpha=0.55,
            linewidth=1.0,
            linestyle=":",
        )

    ax.scatter(
        raw_fluxs,
        f01_mhz,
        marker="o",
        facecolors="none",
        edgecolors="tab:gray",
        s=34,
        label="raw flux",
        zorder=3,
    )
    if np.any(corrected_mask):
        ax.scatter(
            corrected_fluxs[corrected_mask],
            f01_mhz[corrected_mask],
            marker="o",
            color="tab:blue",
            s=20,
            label="f01 corrected flux",
            zorder=4,
        )
    if np.any(explicit_mask):
        ax.scatter(
            corrected_fluxs[explicit_mask],
            f01_mhz[explicit_mask],
            marker="o",
            color="tab:purple",
            s=20,
            label="explicit flux (uncorrected)",
            zorder=4,
        )
    if np.any(missing_mask):
        ax.scatter(
            corrected_fluxs[missing_mask],
            f01_mhz[missing_mask],
            marker="s",
            color="tab:cyan",
            s=24,
            label="missing f01 (model freq)",
            zorder=4,
        )
    if np.any(disabled_mask):
        ax.scatter(
            corrected_fluxs[disabled_mask],
            f01_mhz[disabled_mask],
            marker="^",
            color="tab:brown",
            s=30,
            label="correction disabled",
            zorder=4,
        )
    if np.any(rejected_mask):
        ax.scatter(
            corrected_fluxs[rejected_mask],
            f01_mhz[rejected_mask],
            marker="x",
            color="tab:orange",
            s=38,
            label="kept raw (rejected)",
            zorder=4,
        )
    ax.scatter(
        aligned_fluxs,
        f01_mhz,
        marker="D",
        facecolors="none",
        edgecolors="tab:green",
        s=28,
        label="integer aligned flux",
        zorder=5,
    )

    ax.set_xlabel(r"Flux quanta ($\Phi_{ext}/\Phi_0$)")
    ax.set_ylabel("f01 frequency (MHz)")
    ax.set_title("f01 flux correction and branch alignment")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="best", fontsize="small")
    return fig, ax


def plot_t1_mechanism_probe(probe: T1MechanismProbe) -> tuple[Figure, Axes]:
    """Plot dimensionless pointwise Q/errors and fitted Q against rad/ns omega.

    probe supplies the mechanism labels and fitted points. Return a new
    Figure/Axes on log scales without showing or saving. Plotting errors
    propagate; this stage does not refit Q or temperature.
    """
    fig, ax = plot_Q_vs_omega(
        probe.data.fit.omegas,
        probe.q_values,
        probe.q_errors,
        Qname=_Q_LABELS[probe.mechanism],
    )
    ax.plot(
        probe.fit_omegas,
        probe.fit_q_values,
        label=f"{CURVE_LABELS[probe.mechanism]} probe",
    )
    _set_log_y_limits_to_three_sigma(ax, probe.q_values)
    ax.legend(fontsize="small")
    return fig, ax


def plot_t1_mechanism_dipole(probe: T1MechanismProbe) -> tuple[Figure, Axes]:
    """Plot the probe ns T1/errors against its rate-prefactor dipoles in ns^-1.

    Use intrinsic T1 when Purcell subtraction was enabled and restrict to the
    probe rad/ns omega window and valid positive points. Return a new
    Figure/Axes without showing or saving; no valid points raises ValueError.
    """
    T1_for_q = (
        probe.data.fit.T1_ns
        if probe.purcell_correction is None
        else probe.purcell_correction.intrinsic_T1_ns
    )
    T1err_for_q = (
        probe.data.fit.T1err_ns
        if probe.purcell_correction is None
        else probe.purcell_correction.intrinsic_T1err_ns
    )
    valid = omega_mask(probe.data.fit.omegas, probe.omega_range)
    if probe.purcell_correction is not None:
        valid &= probe.purcell_correction.valid_mask
    valid &= (
        np.isfinite(probe.dipoles)
        & (probe.dipoles > 0.0)
        & np.isfinite(T1_for_q)
        & (T1_for_q > 0.0)
        & np.isfinite(probe.q_values)
        & (probe.q_values > 0.0)
    )
    if not np.any(valid):
        raise ValueError(f"No finite positive {probe.mechanism} dipole points to plot")

    Q_name = (
        r"$x_{qp}$"
        if probe.mechanism == "quasiparticle"
        else _Q_LABELS[probe.mechanism]
    )
    product2val: Callable[[float], float] = (
        (lambda product: 1.0 / product)
        if probe.mechanism == "quasiparticle"
        else (lambda product: product)
    )
    fig, ax = plot_t1_vs_elements(
        probe.dipoles[valid],
        T1_for_q[valid],
        T1err_for_q[valid],
        Q_name=Q_name,
        product2val=product2val,
    )
    return fig, ax


def plot_t1_mechanism_limit(
    probe: T1MechanismProbe,
    *,
    t_flux_count: int = 1000,
    flux_range: tuple[float, float] | None = None,
    purcell: PurcellEffectParams | None = None,
) -> tuple[Figure, Axes]:
    """Calculate dense single-mechanism ns T1 curves and plot their bounds.

    probe supplies temperature in K and dimensionless Q/x_qp estimates. Use
    t_flux_count points across normalized flux_range (default analysis window).
    purcell overrides probe.purcell when supplied; its GHz parameters add a
    relaxation rate to each dense curve. Return a new Figure/Axes without
    showing or saving. Numerical/plotting failures propagate, including invalid
    Purcell inputs. Calculate mechanism curves, then Purcell, then create axes.
    """
    data = probe.data
    context = data.calibration.context
    resolved_purcell = probe.purcell if purcell is None else purcell
    resolved_flux_range = flux_range or data.analysis_flux_range
    t_fluxs = np.linspace(
        resolved_flux_range[0], resolved_flux_range[1], t_flux_count, dtype=np.float64
    )
    channel_name, option_name = NOISE_CHANNELS[probe.parameter_name]
    values = {
        "lower": probe.parameter_lower,
        "probe": probe.parameter_init,
        "upper": probe.parameter_upper,
    }
    mechanism_curves = {
        f"{CURVE_LABELS[probe.mechanism]} {label}": calculate_eff_t1_vs_flux_fast(
            context.params,
            t_fluxs,
            [(channel_name, {option_name: value})],
            probe.temperature,
        )
        for label, value in values.items()
    }
    probe_curve_name = f"{CURVE_LABELS[probe.mechanism]} probe"
    display_T1s = mechanism_curves[probe_curve_name]
    mechanism_label = CURVE_LABELS[probe.mechanism]
    lower_curve_name = f"{mechanism_label} lower"
    upper_curve_name = f"{mechanism_label} upper"
    component_t1s = {
        "- lower": mechanism_curves[lower_curve_name],
        "- upper": mechanism_curves[upper_curve_name],
    }
    plot_label = mechanism_label
    parameter_lines = [
        format_parameter(probe.parameter_name, probe.parameter_init),
        f"Temp = {probe.temperature * 1e3:.2f} mK",
    ]
    if resolved_purcell is not None:
        purcell_T1s = calculate_purcell_t1_limit(
            context,
            t_fluxs,
            resolved_purcell,
            Temp=probe.temperature,
        )
        combined_curves = {
            key: combine_t1_limits(value, purcell_T1s)
            for key, value in mechanism_curves.items()
        }
        display_T1s = combined_curves[probe_curve_name]
        component_t1s = {
            "- lower": combined_curves[lower_curve_name],
            "- upper": combined_curves[upper_curve_name],
        }
        component_t1s = {**component_t1s, "Purcell": purcell_T1s}
        plot_label = f"{plot_label} + Purcell"
    band_label = f"{mechanism_label} bounds"
    component_bands = {
        band_label: (
            component_t1s["- lower"],
            component_t1s["- upper"],
        )
    }
    fig, ax = plot_eff_t1_with_sample(
        flux2value(data.sample.fluxs, context.flux_half, context.flux_period),
        data.sample.T1_ns,
        data.sample.T1err_ns,
        display_T1s,
        context.flux_half,
        context.flux_period,
        t_fluxs,
        label=plot_label,
        component_t1s=component_t1s,
        component_bands=component_bands,
        parameter_text="\n".join(parameter_lines),
    )
    ax.set_xlim(*resolved_flux_range)
    return fig, ax


def plot_t1_channel_analysis(
    channel_analysis: T1ChannelAnalysis,
    *,
    parameter_text: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot prepared ns T1 samples and the channel analysis dense ns curves.

    channel_analysis supplies normalized flux, fit parameters and component
    order. parameter_text overrides the active-parameter display when supplied.
    Return a new Figure/Axes without recalculating, showing or saving curves.
    Base plotting failures, including no positive sample T1, propagate.
    """
    combined_fit = channel_analysis.combined_fit
    data = combined_fit.data
    context = data.calibration.context
    if parameter_text is None:
        parameter_text = t1_parameter_text(combined_fit.fit_result)
    fig, ax = plot_eff_t1_with_sample(
        flux2value(data.sample.fluxs, context.flux_half, context.flux_period),
        data.sample.T1_ns,
        data.sample.T1err_ns,
        channel_analysis.curves.display_T1_ns,
        context.flux_half,
        context.flux_period,
        channel_analysis.curves.fluxs,
        label="effective T1",
        component_t1s=channel_analysis.curves.component_T1s_ns,
        parameter_text=parameter_text,
    )
    ax.set_xlim(*channel_analysis.flux_range)
    return fig, ax


# tolerated: Notebook save calls keep figure/context/name positional and rendering controls optional.
def save_t1_curve_figure(  # noqa: PLR0913
    fig: Figure,
    context: T1CurveContext,
    filename: str,
    *,
    show: bool = True,
    close: bool = True,
    bbox_inches: str | None = None,
    dpi: int = 160,
) -> str:
    """Save fig to context.image_dir/filename and return that path.

    dpi and bbox_inches pass to Matplotlib savefig. After saving, show the
    figure when show=True, then close it when close=True. Save/show failures
    propagate without continuing the later side effects; directories are not
    created here.
    """
    path = os.path.join(context.image_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return path


def t1_parameter_text(
    fit_result: T1FitResult,
    *,
    extra_lines: tuple[str, ...] = (),
) -> str:
    """Render active dimensionless noise parameters and Temp in mK as lines.

    fit_result supplies Q_cap/x_qp/Q_ind and Temp in K. Omit inactive None
    parameters, append extra_lines unchanged and return one newline-joined
    string. This neither validates fit success nor displays uncertainty.
    """
    lines: list[str] = []
    params = fit_result.params
    if params.Q_cap is not None:
        lines.append(format_parameter("Q_cap", params.Q_cap))
    if params.x_qp is not None:
        lines.append(format_parameter("x_qp", params.x_qp))
    if params.Q_ind is not None:
        lines.append(format_parameter("Q_ind", params.Q_ind))
    lines.append(f"Temp = {params.Temp * 1e3:.2f} mK")
    lines.extend(extra_lines)
    return "\n".join(lines)


def _set_log_y_limits_to_three_sigma(
    ax: Axes,
    values: NDArray[np.float64],
) -> None:
    positive = np.asarray(values, dtype=np.float64)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size < 2:
        return
    log_values = np.log(positive)
    spread = float(np.std(log_values))
    if not np.isfinite(spread) or spread <= 0.0:
        return
    center = float(np.mean(log_values))
    ax.set_ylim(
        float(np.exp(center - 3.0 * spread)), float(np.exp(center + 3.0 * spread))
    )
