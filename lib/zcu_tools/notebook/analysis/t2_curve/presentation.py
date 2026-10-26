"""Render, save and display staged T2 curve analysis results."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base import (
    plot_flux_noise_sensitivity,
    plot_t2_channel_curves,
    plot_t2e_vs_flux,
    plot_thermal_photon_t2_limit,
    t2_parameter_text,
)
from .models import (
    DisplayFn,
    T2ChannelAnalysis,
    T2CurveAnalysisConfig,
    T2CurveAnalysisResult,
    T2CurveContext,
    T2DephasingAnalysis,
    T2FluxNoiseProbe,
    T2PhotonShotNoiseProbe,
)


def plot_t2_dephasing_data(data: T2DephasingAnalysis) -> tuple[Figure, Axes]:
    return plot_t2e_vs_flux(
        data.sample.fluxs,
        data.sample.T2e_us,
        data.sample.T1_us,
        fit_fluxs=data.fit.fluxs,
        fit_T2e_us=data.fit.T2e_us,
        fit_T2e_err_us=data.fit.T2e_err_us,
    )


def plot_t2_flux_calibration(data: T2DephasingAnalysis) -> tuple[Figure, Axes]:
    """Plot author flux and integer-aligned flux without inferring prior correction.

    data supplies prepared rows and a normalized analysis window. Return a
    new Figure/Axes without showing or saving. Empty data raises ValueError.
    Vertical positions are sample positions in the prepared window.
    """
    window = data.window
    if len(window.fluxs) == 0:
        raise ValueError("No T2 samples are available for flux calibration plotting")

    raw_fluxs = window.raw_fluxs
    aligned_fluxs = window.fluxs
    y_positions = np.arange(len(aligned_fluxs), dtype=np.float64)
    fig_height = float(np.clip(1.8 + 0.18 * len(aligned_fluxs), 3.0, 8.0))
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    lower, upper = data.analysis_flux_range
    ax.axvspan(lower, upper, color="tab:green", alpha=0.08, label="analysis window")
    for y, raw, aligned in zip(y_positions, raw_fluxs, aligned_fluxs, strict=True):
        ax.plot([raw, aligned], [y, y], color="tab:green", alpha=0.55, linestyle=":")
    ax.scatter(
        raw_fluxs,
        y_positions,
        marker="o",
        facecolors="none",
        edgecolors="tab:gray",
        label="author flux",
    )
    ax.scatter(
        aligned_fluxs,
        y_positions,
        marker="D",
        facecolors="none",
        edgecolors="tab:green",
        label="integer aligned flux",
    )
    ax.set_xlabel(r"Flux quanta ($\Phi_{ext}/\Phi_0$)")
    ax.set_ylabel("sample index")
    ax.set_title("Author flux and integer branch alignment")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="best", fontsize="small")
    return fig, ax


def plot_flux_noise_probe(probe: T2FluxNoiseProbe) -> tuple[Figure, Axes]:
    return plot_flux_noise_sensitivity(
        probe.domega_dflux,
        probe.residual_gamma_phi_per_us,
        A_phi=probe.A_phi_fit,
        gamma_phi_err=probe.data.fit.gamma_phi_err_per_us,
        title="Flux-noise-only dephasing probe",
    )


def plot_photon_shot_noise_probe(
    probe: T2PhotonShotNoiseProbe,
    *,
    n_th_range: tuple[float, float] = (1e-5, 1e-1),
    n_th_count: int = 400,
) -> tuple[Figure, Axes]:
    n_th_axis = np.logspace(
        np.log10(n_th_range[0]), np.log10(n_th_range[1]), n_th_count
    )
    return plot_thermal_photon_t2_limit(
        n_th_axis,
        T1_us=probe.thermal.half_T1_us,
        T2e_us=probe.thermal.half_T2e_us,
        kappa_over_2pi_mhz=probe.readout_kappa_over_2pi_mhz,
        chi_over_2pi_mhz=probe.thermal.half_chi_over_2pi_mhz,
        equivalent_n_th=probe.thermal.half_equivalent_n_th,
    )


def plot_nth_limit_vs_flux(probe: T2PhotonShotNoiseProbe) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(
        probe.pointwise_table["flux"],
        probe.pointwise_table["n_th"],
        "o",
        label="pointwise equivalent n_th",
    )
    ax.axhline(
        probe.n_th_init,
        color="tab:red",
        label=f"probe n_th={probe.n_th_init:.2e}",
    )
    ax.axhline(
        probe.n_th_fit,
        color="tab:gray",
        linestyle="--",
        alpha=0.7,
        label=f"photon-only fit={probe.n_th_fit:.2e}",
    )
    ax.set_xlabel(r"Flux $\Phi_\mathrm{ext}/\Phi_0$")
    ax.set_ylabel(r"Equivalent $n_{th}$")
    ax.set_title("Photon-shot-noise-only n_th probe")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_t2_channel_analysis(
    channel_analysis: T2ChannelAnalysis,
    *,
    parameter_text: str | None = None,
) -> tuple[Figure, Axes]:
    data = channel_analysis.combined_fit.data
    if parameter_text is None:
        parameter_text = t2_parameter_text(channel_analysis.combined_fit.fit_result)
    return plot_t2_channel_curves(
        data.sample.fluxs,
        data.sample.T2e_us,
        data.sample.T1_us,
        channel_analysis.curves,
        fit_fluxs=channel_analysis.combined_fit.fit_fluxs,
        fit_T2e_us=channel_analysis.combined_fit.fit_T2e_us,
        fit_T2e_err_us=channel_analysis.combined_fit.fit_T2e_err_us,
        parameter_text=parameter_text,
        xlim=channel_analysis.flux_range,
    )


# tolerated: Notebook save API separates destination, show/close lifecycle and render options.
def save_t2_curve_figure(  # noqa: PLR0913
    fig: Figure,
    context: T2CurveContext,
    filename: str,
    *,
    show: bool = True,
    close: bool = True,
    bbox_inches: str | None = None,
    dpi: int = 160,
) -> str:
    path = os.path.join(context.image_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return path


def save_t2_standard_figures(
    flux_probe: T2FluxNoiseProbe,
    photon_probe: T2PhotonShotNoiseProbe | None,
    channel_analysis: T2ChannelAnalysis,
    *,
    config: T2CurveAnalysisConfig,
) -> dict[str, str]:
    """Render standard figures, omitting photon figures when photon_probe is None.

    Config controls the thermal-photon axis and saving/showing. Create all
    available figures, then save (160 dpi), show and close each
    figure in that order. Return paths only for figures actually saved under
    the records' context image_dir. Plot/save/display failures propagate.
    """
    data = channel_analysis.combined_fit.data
    context = data.calibration.context
    figures = {
        "flux_calibration": (
            plot_t2_flux_calibration(data)[0],
            "flux_calibration.png",
            None,
        ),
        "t2e_vs_flux": (plot_t2_dephasing_data(data)[0], "T2e_vs_flux.png", None),
        "flux_noise_probe": (
            plot_flux_noise_probe(flux_probe)[0],
            "Gamma_phi_vs_flux_sensitivity.png",
            None,
        ),
        "channel_overlay": (
            plot_t2_channel_analysis(channel_analysis)[0],
            "T2e_flux_noise_fit.png",
            "tight",
        ),
    }
    if photon_probe is not None:
        figures["photon_shot_noise_probe"] = (
            plot_photon_shot_noise_probe(
                photon_probe,
                n_th_range=config.thermal_n_th_range,
                n_th_count=config.thermal_n_th_count,
            )[0],
            "T2e_thermal_photon_limit.png",
            None,
        )
        figures["n_th_vs_flux"] = (
            plot_nth_limit_vs_flux(photon_probe)[0],
            "n_th_vs_flux.png",
            None,
        )
    paths: dict[str, str] = {}
    for name, (fig, filename, bbox_inches) in figures.items():
        path = os.path.join(context.image_dir, filename)
        if config.save_figures:
            fig.savefig(path, dpi=160, bbox_inches=bbox_inches)
            paths[name] = path
        if config.show_figures:
            plt.show()
        plt.close(fig)
    return paths


def emit_t2_curve_outputs(
    result: T2CurveAnalysisResult,
    *,
    display_fn: DisplayFn | None,
    verbose: bool,
    tables: bool,
) -> None:
    """Print fit metadata and display result tables in the standard order.

    Print only when verbose. Display tables only when tables is true and
    display_fn is provided; omit an empty half-flux preview. The callback
    accepts each DataFrame as one object. Omit unavailable photon probe tables
    and print the disabled-channel reason. Output/callback failures propagate.
    """
    if verbose:
        print("params =", result.context.params, "GHz")
        print("bare_rf =", result.context.bare_rf, "GHz")
        print("g =", result.context.g, "GHz")
        if not result.context.photon_shot_noise_enabled:
            print("photon shot noise disabled: missing dispersive inputs")
        print(
            "readout kappa/2pi =",
            result.combined_fit.readout_kappa_over_2pi_mhz,
            "MHz",
        )
        print("fit success =", result.combined_fit.fit_result.success)
        print("fit message =", result.combined_fit.fit_result.message)
        print("fixed =", result.combined_fit.fit_result.fixed)
        print("free =", result.combined_fit.fit_result.free)
        print(f"Images saved under: {result.context.image_dir}")

    if tables and display_fn is not None:
        display_fn(result.calibration.summary_table)
        display_fn(result.data.branch_coverage)
        if not result.data.half_preview.empty:
            display_fn(result.data.half_preview)
        display_fn(result.flux_probe.summary_table)
        if result.photon_probe is not None:
            display_fn(result.photon_probe.summary_table)
            display_fn(result.photon_probe.thermal_limit_table)
        display_fn(result.combined_fit.params_table)
        display_fn(result.summary_table)
