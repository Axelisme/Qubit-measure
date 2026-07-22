from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.meta_tool import T1CurveFit
from zcu_tools.notebook.analysis.fit_tools import (
    choose_current_scale_from_f01 as _choose_current_scale_from_f01,
)
from zcu_tools.notebook.analysis.fit_tools import (
    predict_domega_dflux as predict_domega_dflux,
)
from zcu_tools.notebook.analysis.fit_tools import (
    predict_f01_mhz as predict_f01_mhz,
)
from zcu_tools.simulate.fluxonium import (
    calculate_dispersive_vs_flux_fast,
    calculate_eff_t1_vs_flux_fast,
)

from .fit import (
    T2FitResult,
    flux_noise_gamma_phi_per_us,
    thermal_photon_gamma_phi_per_us,
    thermal_photon_t2_limit_us,
)


@dataclass(frozen=True, slots=True)
class T2ChannelCurves:
    fluxs: NDArray[np.float64]
    T1_us: NDArray[np.float64]
    T2_relax_us: NDArray[np.float64]
    Tphi_flux_us: NDArray[np.float64]
    Tphi_photon_us: NDArray[np.float64]
    T2_effective_us: NDArray[np.float64]
    gamma_phi_flux: NDArray[np.float64]
    gamma_phi_photon: NDArray[np.float64]
    t1_label: str


def choose_current_scale(
    raw_values: NDArray[np.float64],
    measured_freqs_mhz: NDArray[np.float64],
    *,
    params: tuple[float, float, float],
    flux_half: float,
    flux_period: float,
    candidates: tuple[float, ...] = (1.0, 1000.0),
) -> tuple[float, pd.DataFrame]:
    return _choose_current_scale_from_f01(
        raw_values,
        measured_freqs_mhz,
        params=params,
        flux_half=flux_half,
        flux_period=flux_period,
        candidates=candidates,
    )


def dispersive_chi01_over_2pi_mhz(
    params: tuple[float, float, float],
    fluxs: NDArray[np.float64],
    bare_rf: float,
    g: float,
    *,
    res_dim: int = 5,
    qub_dim: int = 15,
) -> NDArray[np.float64]:
    rf_0_ghz, rf_1_ghz = calculate_dispersive_vs_flux_fast(
        params,
        np.asarray(fluxs, dtype=np.float64),
        bare_rf,
        g,
        res_dim=res_dim,
        qub_dim=qub_dim,
        return_dim=2,
    )
    return np.asarray(1e3 * np.abs(rf_1_ghz - rf_0_ghz), dtype=np.float64)


def make_thermal_limit_table(
    n_th_values: NDArray[np.float64],
    *,
    T1_us: float,
    kappa_over_2pi_mhz: float,
    chi_over_2pi_mhz: float,
) -> pd.DataFrame:
    n_th_arr = np.asarray(n_th_values, dtype=np.float64)
    n_th_arr = np.unique(np.sort(n_th_arr[np.isfinite(n_th_arr)]))
    return pd.DataFrame(
        {
            "n_th": n_th_arr,
            "Gamma_phi_th (1/us)": thermal_photon_gamma_phi_per_us(
                n_th_arr,
                kappa_over_2pi_mhz=kappa_over_2pi_mhz,
                chi_over_2pi_mhz=chi_over_2pi_mhz,
            ),
            "T2_limit (us)": thermal_photon_t2_limit_us(
                n_th_arr,
                T1_us=T1_us,
                kappa_over_2pi_mhz=kappa_over_2pi_mhz,
                chi_over_2pi_mhz=chi_over_2pi_mhz,
            ),
        }
    )


def calculate_t2_channel_curves(
    t_fluxs: NDArray[np.float64],
    *,
    params: tuple[float, float, float],
    fit_result: T2FitResult,
    kappa_over_2pi_mhz: float,
    chi_over_2pi_mhz: NDArray[np.float64],
    domega_dflux: NDArray[np.float64],
    t1_curve_fit: T1CurveFit | None = None,
    fit_fluxs: NDArray[np.float64] | None = None,
    fit_T1_us: NDArray[np.float64] | None = None,
) -> T2ChannelCurves:
    fluxs = np.asarray(t_fluxs, dtype=np.float64)
    chi_arr = np.asarray(chi_over_2pi_mhz, dtype=np.float64)
    domega_arr = np.asarray(domega_dflux, dtype=np.float64)
    _validate_same_shape("chi_over_2pi_mhz", chi_arr, fluxs)
    _validate_same_shape("domega_dflux", domega_arr, fluxs)

    A_phi = fit_result.params.A_phi
    n_th = fit_result.params.n_th
    if t1_curve_fit is not None:
        t1_noise_channels = t1_noise_channels_from_curve_fit(t1_curve_fit)
        T1_us = 1e-3 * calculate_eff_t1_vs_flux_fast(
            params,
            fluxs,
            t1_noise_channels,
            t1_curve_fit.params.Temp,
        )
        t1_label = "T1-curve fit"
    else:
        if fit_fluxs is None or fit_T1_us is None:
            raise ValueError(
                "fit_fluxs and fit_T1_us are required when t1_curve_fit is missing"
            )
        order = np.argsort(fit_fluxs)
        T1_us = np.interp(fluxs, fit_fluxs[order], fit_T1_us[order])
        t1_label = "interpolated measured T1"

    gamma_phi_flux = (
        flux_noise_gamma_phi_per_us(A_phi, domega_arr)
        if A_phi is not None
        else np.zeros_like(fluxs, dtype=np.float64)
    )
    gamma_phi_photon = (
        np.asarray(
            thermal_photon_gamma_phi_per_us(
                n_th,
                kappa_over_2pi_mhz=kappa_over_2pi_mhz,
                chi_over_2pi_mhz=chi_arr,
            ),
            dtype=np.float64,
        )
        if n_th is not None
        else np.zeros_like(fluxs, dtype=np.float64)
    )
    T2_relax_us = 2.0 * T1_us
    Tphi_flux_us = _safe_inverse(gamma_phi_flux)
    Tphi_photon_us = _safe_inverse(gamma_phi_photon)
    T2_effective_us = 1.0 / (1.0 / (2.0 * T1_us) + gamma_phi_flux + gamma_phi_photon)

    return T2ChannelCurves(
        fluxs=fluxs,
        T1_us=T1_us,
        T2_relax_us=T2_relax_us,
        Tphi_flux_us=Tphi_flux_us,
        Tphi_photon_us=Tphi_photon_us,
        T2_effective_us=T2_effective_us,
        gamma_phi_flux=gamma_phi_flux,
        gamma_phi_photon=gamma_phi_photon,
        t1_label=t1_label,
    )


def t1_noise_channels_from_curve_fit(
    t1_curve_fit: T1CurveFit,
) -> list[tuple[str, dict[str, float]]]:
    channels: list[tuple[str, dict[str, float]]] = []
    if t1_curve_fit.params.Q_cap is not None:
        channels.append(("t1_capacitive", {"Q_cap": t1_curve_fit.params.Q_cap}))
    if t1_curve_fit.params.x_qp is not None:
        channels.append(
            ("t1_quasiparticle_tunneling", {"x_qp": t1_curve_fit.params.x_qp})
        )
    if t1_curve_fit.params.Q_ind is not None:
        channels.append(("t1_inductive", {"Q_ind": t1_curve_fit.params.Q_ind}))
    return channels


def t2_parameter_text(
    fit_result: T2FitResult,
    *,
    extra_lines: Sequence[str] = (),
) -> str:
    lines: list[str] = []
    if fit_result.params.A_phi is not None:
        lines.append(
            rf"$A_\Phi$ = {fit_result.params.A_phi * 1e6:.3f} "
            rf"$\mu\Phi_0/\sqrt{{Hz}}$"
        )
    if fit_result.params.n_th is not None:
        lines.append(rf"$n_{{th}}$ = {fit_result.params.n_th:.3e}")
    lines.append(f"reduced chi2 = {fit_result.reduced_chi2:.3g}")
    lines.extend(extra_lines)
    return "\n".join(lines)


def plot_t2e_vs_flux(
    sample_fluxs: NDArray[np.float64],
    sample_T2e_us: NDArray[np.float64],
    sample_T1_us: NDArray[np.float64],
    *,
    fit_fluxs: NDArray[np.float64] | None = None,
    fit_T2e_us: NDArray[np.float64] | None = None,
    fit_T2e_err_us: NDArray[np.float64] | None = None,
    title: str = "Echo T2 vs T1 ceiling",
) -> tuple[Figure, Axes]:
    sample_fluxs = np.asarray(sample_fluxs, dtype=np.float64)
    sample_T2e_us = np.asarray(sample_T2e_us, dtype=np.float64)
    sample_T1_us = np.asarray(sample_T1_us, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(sample_fluxs, sample_T2e_us, "o", color="tab:blue", label="T2 echo sample")
    if fit_fluxs is not None and fit_T2e_us is not None:
        ax.errorbar(
            fit_fluxs,
            fit_T2e_us,
            yerr=fit_T2e_err_us,
            fmt="none",
            ecolor="tab:blue",
            alpha=0.7,
            capsize=3,
        )
    ax.scatter(
        sample_fluxs,
        2.0 * sample_T1_us,
        s=24,
        color="tab:green",
        label="2*T1 sample",
    )
    ax.set_xlabel(r"Flux $\Phi_\mathrm{ext}/\Phi_0$")
    ax.set_ylabel("Time (us)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_thermal_photon_t2_limit(
    n_th_axis: NDArray[np.float64],
    *,
    T1_us: float,
    T2e_us: float,
    kappa_over_2pi_mhz: float,
    chi_over_2pi_mhz: float,
    equivalent_n_th: float,
    title: str = "Half-flux thermal photon shot-noise ceiling",
) -> tuple[Figure, Axes]:
    n_th_axis = np.asarray(n_th_axis, dtype=np.float64)
    T2_limit_axis_us = thermal_photon_t2_limit_us(
        n_th_axis,
        T1_us=T1_us,
        kappa_over_2pi_mhz=kappa_over_2pi_mhz,
        chi_over_2pi_mhz=chi_over_2pi_mhz,
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.semilogx(n_th_axis, T2_limit_axis_us, label="thermal photon + T1 limit")
    ax.axhline(2.0 * T1_us, color="tab:green", linestyle="--", label="2*T1")
    ax.axhline(T2e_us, color="tab:orange", linestyle=":", label="measured T2 echo")
    ax.axvline(
        equivalent_n_th,
        color="tab:red",
        linestyle=":",
        label=rf"equiv. $n_{{th}}={equivalent_n_th:.2e}$",
    )
    ax.set_xlabel(r"Residual readout thermal photons $n_{th}$")
    ax.set_ylabel("T2 limit (us)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_flux_noise_sensitivity(
    domega_dflux: NDArray[np.float64],
    gamma_phi_residual: NDArray[np.float64],
    *,
    A_phi: float,
    gamma_phi_err: NDArray[np.float64] | None = None,
    title: str = "Joint-fit photon-subtracted echo dephasing vs flux sensitivity",
) -> tuple[Figure, Axes]:
    sensitivity = np.sqrt(np.log(2.0)) * np.abs(
        np.asarray(domega_dflux, dtype=np.float64)
    )
    line_x = np.linspace(0.0, 1.05 * np.nanmax(sensitivity), 200)
    line_y = float(A_phi) * line_x

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.errorbar(
        sensitivity,
        gamma_phi_residual,
        yerr=gamma_phi_err,
        fmt="o",
        capsize=3,
        label="sample dephasing after fitted photon subtraction",
    )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.plot(
        line_x,
        line_y,
        label=rf"$A_\Phi$ = {float(A_phi) * 1e6:.2f} $\mu\Phi_0/\sqrt{{Hz}}$",
    )
    ax.set_xlabel(r"$\sqrt{\ln 2}\,|\partial\omega_{01}/\partial\Phi|$ (1/us/Phi0)")
    ax.set_ylabel(r"$\Gamma_\phi - \Gamma_\mathrm{photon,fit}$ (1/us)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_t2_channel_curves(
    sample_fluxs: NDArray[np.float64],
    sample_T2e_us: NDArray[np.float64],
    sample_T1_us: NDArray[np.float64],
    curves: T2ChannelCurves,
    *,
    fit_fluxs: NDArray[np.float64] | None = None,
    fit_T2e_us: NDArray[np.float64] | None = None,
    fit_T2e_err_us: NDArray[np.float64] | None = None,
    parameter_text: str | None = None,
    xlim: tuple[float, float] | None = None,
    title: str = "Echo T2 channel limits",
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    ax.plot(sample_fluxs, sample_T2e_us, "o", color="tab:blue", label="T2 echo sample")
    if fit_fluxs is not None and fit_T2e_us is not None:
        ax.errorbar(
            fit_fluxs,
            fit_T2e_us,
            yerr=fit_T2e_err_us,
            fmt="none",
            ecolor="tab:blue",
            alpha=0.7,
            capsize=3,
        )
    ax.scatter(
        sample_fluxs,
        2.0 * sample_T1_us,
        s=24,
        color="tab:green",
        alpha=0.65,
        label="2*T1 sample",
    )
    ax.plot(
        curves.fluxs,
        curves.T2_relax_us,
        "--",
        color="tab:green",
        label=f"2*T1 ({curves.t1_label})",
    )
    ax.plot(
        curves.fluxs,
        curves.Tphi_flux_us,
        ":",
        color="tab:blue",
        label="pure 1/f flux noise",
    )
    ax.plot(
        curves.fluxs,
        curves.Tphi_photon_us,
        "-.",
        color="tab:red",
        label="photon shot noise",
    )
    ax.plot(
        curves.fluxs,
        curves.T2_effective_us,
        "-",
        color="black",
        linewidth=2.0,
        label="effective: 2*T1 + flux + photons",
    )
    ax.set_xlabel(r"Flux $\Phi_\mathrm{ext}/\Phi_0$")
    ax.set_ylabel("Time (us)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    if parameter_text:
        ax.text(
            1.02,
            0.98,
            parameter_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="0.5",
                alpha=0.85,
            ),
            clip_on=False,
        )
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_yscale("log")
    ax.set_ylim(_t2_channel_ylim(sample_T2e_us, sample_T1_us, curves))
    return fig, ax


def _safe_inverse(values: NDArray[np.float64]) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    return np.divide(1.0, arr, out=np.full_like(arr, np.inf), where=arr > 0.0)


def _validate_same_shape(
    name: str,
    values: NDArray[np.float64],
    reference: NDArray[np.float64],
) -> None:
    if values.shape != reference.shape:
        raise ValueError(f"{name} must have the same shape as t_fluxs")


def _t2_channel_ylim(
    sample_T2e_us: NDArray[np.float64],
    sample_T1_us: NDArray[np.float64],
    curves: T2ChannelCurves,
) -> tuple[float, float]:
    ceiling_candidates = [
        float(np.nanmax(sample_T2e_us)),
        float(np.nanmax(2.0 * np.asarray(sample_T1_us, dtype=np.float64))),
    ]
    for values in (
        curves.T2_relax_us,
        curves.Tphi_flux_us,
        curves.Tphi_photon_us,
        curves.T2_effective_us,
    ):
        finite_positive = values[np.isfinite(values) & (values > 0.0)]
        if finite_positive.size:
            ceiling_candidates.append(float(np.nanpercentile(finite_positive, 99.0)))
    return (
        max(0.5, 0.8 * float(np.nanmin(sample_T2e_us))),
        1.2 * float(np.nanmax(ceiling_candidates)),
    )
