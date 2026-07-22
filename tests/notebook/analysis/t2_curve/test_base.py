from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from zcu_tools.notebook.analysis.t2_curve import (
    T2FitParams,
    calculate_t2_channel_curves,
    fit_t2_noise_params,
    make_thermal_limit_table,
    plot_flux_noise_sensitivity,
    plot_t2_channel_curves,
    plot_t2e_vs_flux,
    plot_thermal_photon_t2_limit,
    t2_parameter_text,
    thermal_photon_gamma_phi_per_us,
)

_KAPPA_OVER_2PI_MHZ = 14.754


def test_make_thermal_limit_table_sorts_and_filters_n_th_values() -> None:
    table = make_thermal_limit_table(
        np.asarray([1e-3, np.nan, 0.0, 1e-4, 1e-3], dtype=np.float64),
        T1_us=60.0,
        kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
        chi_over_2pi_mhz=2.9,
    )

    assert table["n_th"].to_list() == [0.0, 1e-4, 1e-3]
    assert table["T2_limit (us)"].iloc[0] == pytest.approx(120.0)
    assert table["T2_limit (us)"].is_monotonic_decreasing


def test_calculate_t2_channel_curves_uses_interpolated_t1_fallback() -> None:
    fit_result = _fit_result()
    fluxs = np.linspace(0.49, 0.53, 20, dtype=np.float64)
    domega_dflux = np.linspace(1_000.0, 12_000.0, len(fluxs), dtype=np.float64)
    chi = np.linspace(2.8, 3.4, len(fluxs), dtype=np.float64)
    fit_fluxs = np.asarray([0.49, 0.53], dtype=np.float64)
    fit_T1_us = np.asarray([50.0, 70.0], dtype=np.float64)

    curves = calculate_t2_channel_curves(
        fluxs,
        params=(3.4, 0.9, 0.6),
        fit_result=fit_result,
        kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
        chi_over_2pi_mhz=chi,
        domega_dflux=domega_dflux,
        fit_fluxs=fit_fluxs,
        fit_T1_us=fit_T1_us,
    )

    assert curves.t1_label == "interpolated measured T1"
    assert curves.T2_effective_us.shape == fluxs.shape
    assert curves.T2_relax_us[0] == pytest.approx(100.0)
    assert np.all(curves.T2_effective_us < curves.T2_relax_us)


def test_calculate_t2_channel_curves_allows_partial_mechanisms() -> None:
    T1s = np.asarray([50.0, 55.0, 60.0], dtype=np.float64)
    domega_dflux = np.asarray([1_000.0, 6_000.0, 12_000.0], dtype=np.float64)
    chi = np.asarray([2.8, 3.0, 3.2], dtype=np.float64)
    gamma_phi = 2.0e-6 * np.sqrt(np.log(2.0)) * np.abs(domega_dflux)
    T2s = 1.0 / (1.0 / (2.0 * T1s) + gamma_phi)
    fit_result = fit_t2_noise_params(
        T1s,
        T2s,
        domega_dflux,
        chi,
        kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
        init=T2FitParams(A_phi=1e-6),
    )

    curves = calculate_t2_channel_curves(
        np.asarray([0.49, 0.50, 0.51], dtype=np.float64),
        params=(3.4, 0.9, 0.6),
        fit_result=fit_result,
        kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
        chi_over_2pi_mhz=chi,
        domega_dflux=domega_dflux,
        fit_fluxs=np.asarray([0.49, 0.51], dtype=np.float64),
        fit_T1_us=np.asarray([50.0, 60.0], dtype=np.float64),
    )

    assert np.all(curves.gamma_phi_photon == 0.0)
    assert np.all(np.isinf(curves.Tphi_photon_us))
    assert np.all(np.isfinite(curves.T2_effective_us))


def test_plot_helpers_return_figures() -> None:
    fit_result = _fit_result()
    sample_fluxs = np.asarray([0.5, 0.51, 0.52], dtype=np.float64)
    sample_T2e_us = np.asarray([30.0, 8.0, 4.0], dtype=np.float64)
    sample_T1_us = np.asarray([60.0, 70.0, 80.0], dtype=np.float64)
    sample_err = np.asarray([2.0, 0.5, 0.4], dtype=np.float64)
    domega_dflux = np.asarray([100.0, 6_000.0, 12_000.0], dtype=np.float64)
    residual = np.asarray([0.001, 0.02, 0.04], dtype=np.float64)

    curves = calculate_t2_channel_curves(
        sample_fluxs,
        params=(3.4, 0.9, 0.6),
        fit_result=fit_result,
        kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
        chi_over_2pi_mhz=np.asarray([2.8, 3.0, 3.2], dtype=np.float64),
        domega_dflux=domega_dflux,
        fit_fluxs=sample_fluxs,
        fit_T1_us=sample_T1_us,
    )

    figs = [
        plot_t2e_vs_flux(
            sample_fluxs,
            sample_T2e_us,
            sample_T1_us,
            fit_fluxs=sample_fluxs,
            fit_T2e_us=sample_T2e_us,
            fit_T2e_err_us=sample_err,
        )[0],
        plot_thermal_photon_t2_limit(
            np.logspace(-5, -3, 10),
            T1_us=60.0,
            T2e_us=30.0,
            kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
            chi_over_2pi_mhz=2.9,
            equivalent_n_th=1e-4,
        )[0],
        plot_flux_noise_sensitivity(
            domega_dflux,
            residual,
            A_phi=fit_result.params.A_phi or 0.0,
            gamma_phi_err=sample_err / 100.0,
        )[0],
        plot_t2_channel_curves(
            sample_fluxs,
            sample_T2e_us,
            sample_T1_us,
            curves,
            fit_fluxs=sample_fluxs,
            fit_T2e_us=sample_T2e_us,
            fit_T2e_err_us=sample_err,
            parameter_text=t2_parameter_text(fit_result),
            xlim=(0.49, 0.53),
        )[0],
    ]

    try:
        assert all(fig.axes for fig in figs)
    finally:
        for fig in figs:
            plt.close(fig)


def _fit_result():
    T1s = np.asarray([50.0, 55.0, 60.0], dtype=np.float64)
    domega_dflux = np.asarray([1_000.0, 6_000.0, 12_000.0], dtype=np.float64)
    chi = np.asarray([2.8, 3.0, 3.2], dtype=np.float64)
    gamma_phi = 2.0e-6 * np.sqrt(np.log(2.0)) * np.abs(domega_dflux)
    gamma_phi += np.asarray(
        thermal_photon_gamma_phi_per_us(
            2e-3,
            kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
            chi_over_2pi_mhz=chi,
        ),
        dtype=np.float64,
    )
    T2s = 1.0 / (1.0 / (2.0 * T1s) + gamma_phi)
    return fit_t2_noise_params(
        T1s,
        T2s,
        domega_dflux,
        chi,
        kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
        init=T2FitParams(A_phi=1e-6, n_th=1e-3),
    )
