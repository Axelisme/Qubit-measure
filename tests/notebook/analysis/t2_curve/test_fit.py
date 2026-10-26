from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import zcu_tools.notebook.analysis.t2_curve.t2_curve_fit as public_fit_mod
from numpy.typing import NDArray
from zcu_tools.notebook.analysis.fit_tools import find_weak_parameters
from zcu_tools.notebook.analysis.t2_curve import (
    FluxResidualWeighting,
    MeasurementErrorPolicy,
    T2FitParams,
    fit_t2_noise_params,
    flux_noise_gamma_phi_per_us,
    thermal_photon_gamma_phi_per_us,
)

_KAPPA_OVER_2PI_MHZ = 14.754


def test_t2_curve_fit_module_exports_public_api() -> None:
    assert public_fit_mod.T2FitParams is T2FitParams
    assert public_fit_mod.fit_t2_noise_params is fit_t2_noise_params


def test_fit_t2_noise_params_recovers_joint_parameters() -> None:
    true_params = T2FitParams(A_phi=2.4e-6, n_th=6.0e-3)
    T1s, T2s, domega_dflux, chi = _synthetic_samples(true_params)

    result = fit_t2_noise_params(
        T1s,
        T2s,
        domega_dflux,
        chi,
        kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
        init=T2FitParams(A_phi=1.0e-6, n_th=2.0e-3),
        bounds={"A_phi": (1e-8, 1e-4), "n_th": (1e-5, 1e-1)},
    )

    assert result.success
    assert result.fixed == ()
    assert result.free == ("A_phi", "n_th")
    assert result.params.A_phi == pytest.approx(true_params.A_phi, rel=1e-6)
    assert result.params.n_th == pytest.approx(true_params.n_th, rel=1e-6)
    np.testing.assert_allclose(result.model_T2s, T2s, rtol=1e-7)
    np.testing.assert_allclose(result.residuals, 0.0, atol=1e-8)


def test_joint_fit_reports_log_covariance_in_free_order() -> None:
    T1s, T2s, sensitivity, chi = _synthetic_samples(
        T2FitParams(A_phi=2.4e-6, n_th=6e-3)
    )
    result = fit_t2_noise_params(
        T1s,
        T2s * (1 + 0.005 * np.sin(np.arange(T2s.size))),
        sensitivity,
        chi,
        kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
        init=T2FitParams(A_phi=2e-6, n_th=5e-3),
    )

    info = result.correlation
    assert info is not None
    assert info.parameter_names == result.free
    np.testing.assert_allclose(info.covariance, info.covariance.T)
    np.testing.assert_allclose(np.diag(info.correlation), 1)
    np.testing.assert_allclose(
        np.sqrt(np.diag(info.covariance)) * [result.params.A_phi, result.params.n_th],
        np.asarray([result.stderr.A_phi, result.stderr.n_th], dtype=np.float64),
    )
    assert info.rank == 2
    assert np.isfinite(info.condition_number)


def test_well_constrained_free_parameter_is_not_weak() -> None:
    T1s, T2s, sensitivity, chi = _synthetic_samples(T2FitParams(A_phi=2.4e-6))
    result = fit_t2_noise_params(
        T1s,
        T2s * (1 + 0.001 * np.sin(np.arange(T2s.size))),
        sensitivity,
        chi,
        kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
        init=T2FitParams(A_phi=2e-6),
    )
    assert result.correlation is not None
    assert result.correlation.parameter_names == ("A_phi",)
    assert find_weak_parameters(result.correlation) == ()


def test_negligible_photon_channel_is_flagged_by_log_uncertainty() -> None:
    T1s, T2s, sensitivity, chi = _synthetic_samples(
        T2FitParams(A_phi=2.4e-6, n_th=1e-5)
    )
    result = fit_t2_noise_params(
        T1s,
        T2s * (1 + 0.02 * np.sin(np.arange(T2s.size))),
        sensitivity,
        chi,
        kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
        init=T2FitParams(A_phi=2e-6, n_th=1e-5),
        bounds={"n_th": (1e-8, 0.1)},
    )
    weak = find_weak_parameters(result.correlation)
    photon = next(entry for entry in weak if entry.parameter == "n_th")
    assert any("sigma(ln n_th)" in reason for reason in photon.reasons)


def test_nearly_degenerate_channels_report_strong_correlation() -> None:
    sensitivity = np.linspace(9_900.0, 10_100.0, 30)
    chi = np.full(30, 3.0)
    T1s = np.full(30, 60.0)
    gamma = flux_noise_gamma_phi_per_us(2.4e-6, sensitivity)
    gamma += thermal_photon_gamma_phi_per_us(
        0.006,
        kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
        chi_over_2pi_mhz=chi,
    )
    result = fit_t2_noise_params(
        T1s,
        1 / (1 / (2 * T1s) + gamma),
        sensitivity,
        chi,
        kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
        init=T2FitParams(A_phi=2e-6, n_th=0.005),
    )
    assert result.correlation is not None
    assert abs(result.correlation.correlation[0, 1]) > 0.95
    assert [entry.parameter for entry in find_weak_parameters(result.correlation)] == [
        "A_phi",
        "n_th",
    ]


def test_fit_t2_noise_params_keeps_fixed_values() -> None:
    true_params = T2FitParams(A_phi=2.4e-6, n_th=6.0e-3)
    T1s, T2s, domega_dflux, chi = _synthetic_samples(true_params)

    result = fit_t2_noise_params(
        T1s,
        T2s,
        domega_dflux,
        chi,
        kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
        init=T2FitParams(A_phi=1.0e-6, n_th=true_params.n_th),
        fixed=("n_th",),
    )

    assert result.fixed == ("n_th",)
    assert result.free == ("A_phi",)
    assert result.params.n_th == true_params.n_th
    assert result.stderr.n_th == 0.0
    assert result.correlation is not None
    assert result.correlation.parameter_names == ("A_phi",)
    assert result.params.A_phi == pytest.approx(true_params.A_phi, rel=1e-6)


def test_fit_t2_noise_params_allows_zero_model_dephasing() -> None:
    T1s = np.asarray([10.0, 10.0], dtype=np.float64)
    gamma_phi_obs = np.asarray([0.01, 0.02], dtype=np.float64)
    T2s = 1.0 / (1.0 / (2.0 * T1s) + gamma_phi_obs)
    domega_dflux = np.asarray([0.0, 10_000.0], dtype=np.float64)
    chi = np.ones_like(T1s)

    result = fit_t2_noise_params(
        T1s,
        T2s,
        domega_dflux,
        chi,
        kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
        init=T2FitParams(A_phi=2.0e-6),
        fixed=("A_phi",),
    )

    assert result.success
    assert result.correlation is None
    assert result.gamma_phi_flux[0] == 0.0
    assert np.all(np.isfinite(result.residuals))
    assert np.max(np.abs(result.residuals)) < 1.0


def test_fit_t2_noise_params_applies_error_and_flux_weight_policies() -> None:
    true_params = T2FitParams(A_phi=2.4e-6, n_th=6.0e-3)
    T1s, T2s, domega_dflux, chi = _synthetic_samples(true_params)
    fluxs = np.r_[np.full(12, 0.5), np.linspace(0.51, 0.53, len(T1s) - 12)]
    T2errs = np.full_like(T2s, 0.2)
    T2errs[0] = np.nan

    result = fit_t2_noise_params(
        T1s,
        T2s,
        domega_dflux,
        chi,
        kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
        init=T2FitParams(A_phi=1.0e-6, n_th=2.0e-3),
        bounds={"A_phi": (1e-8, 1e-4), "n_th": (1e-5, 1e-1)},
        T2errs=T2errs,
        fluxs=fluxs,
        T2_error_policy=MeasurementErrorPolicy(nan_policy="bin_median"),
        flux_weighting=FluxResidualWeighting(mode="equal_flux_bin", bin_width=0.005),
    )

    assert result.success
    assert result.T2_error_resolution is not None
    assert result.T2_error_resolution.nan_mask[0]
    assert result.T2_error_resolution.effective_errors[0] == pytest.approx(0.2)
    assert result.flux_weights.effective_observation_count < len(T2s)
    assert result.params.A_phi == pytest.approx(true_params.A_phi, rel=1e-6)
    assert result.params.n_th == pytest.approx(true_params.n_th, rel=1e-6)


def test_fit_t2_noise_params_fills_all_nan_t1_errors_with_fallback() -> None:
    true_params = T2FitParams(A_phi=2.4e-6, n_th=6.0e-3)
    T1s, T2s, domega_dflux, chi = _synthetic_samples(true_params)
    T1errs = np.full_like(T1s, np.nan)
    T2errs = np.full_like(T2s, 0.2)

    result = fit_t2_noise_params(
        T1s,
        T2s,
        domega_dflux,
        chi,
        kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
        init=T2FitParams(A_phi=1.0e-6, n_th=2.0e-3),
        bounds={"A_phi": (1e-8, 1e-4), "n_th": (1e-5, 1e-1)},
        T1errs=T1errs,
        T2errs=T2errs,
        T1_error_policy=MeasurementErrorPolicy(
            nan_policy="bin_median",
            fallback_error=1.0,
        ),
    )

    assert result.success
    assert result.T1_error_resolution is not None
    np.testing.assert_allclose(result.T1_error_resolution.effective_errors, 1.0)
    np.testing.assert_array_equal(
        result.T1_error_resolution.fallback_fill_mask,
        np.ones_like(T1s, dtype=bool),
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bounds": {"bad": (1.0, 2.0)}}, "unknown bound"),
        ({"fixed": ("bad",)}, "unknown fixed"),
        ({"fixed": ("n_th",)}, "inactive"),
        ({"bounds": {"n_th": (1e-5, 1e-1)}}, "inactive"),
    ],
)
def test_fit_t2_noise_params_validates_whitelist_metadata(
    kwargs: dict[str, Any], match: str
) -> None:
    T1s, T2s, domega_dflux, chi = _synthetic_samples(T2FitParams(A_phi=2.4e-6))

    with pytest.raises(ValueError, match=match):
        fit_t2_noise_params(
            T1s,
            T2s,
            domega_dflux,
            chi,
            kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
            init=T2FitParams(A_phi=1.0e-6),
            **kwargs,
        )


@pytest.mark.parametrize(
    ("T1s", "T2s", "T1errs", "T2errs", "match"),
    [
        (np.array([10.0]), np.array([1.0, 2.0]), None, None, "same shape"),
        (np.array([10.0]), np.array([-1.0]), None, None, "finite and positive"),
        (np.array([1.0]), np.array([3.0]), None, None, "pure-dephasing"),
        (np.array([10.0]), np.array([1.0]), np.array([0.0]), None, "positive finite"),
        (
            np.array([10.0]),
            np.array([1.0]),
            None,
            np.array([np.inf]),
            "positive finite",
        ),
    ],
)
def test_fit_t2_noise_params_validates_data(
    T1s: NDArray[np.float64],
    T2s: NDArray[np.float64],
    T1errs: NDArray[np.float64] | None,
    T2errs: NDArray[np.float64] | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        fit_t2_noise_params(
            T1s,
            T2s,
            np.ones_like(T1s),
            np.ones_like(T1s),
            kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
            init=T2FitParams(A_phi=1e-6, n_th=1e-3),
            T1errs=T1errs,
            T2errs=T2errs,
        )


def _synthetic_samples(
    params: T2FitParams,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    fluxs = np.linspace(0.49, 0.53, 30, dtype=np.float64)
    T1s = 60.0 + 20.0 * (fluxs - 0.49)
    domega_dflux = np.linspace(1_000.0, 18_000.0, len(fluxs), dtype=np.float64)
    chi = 2.8 + 25.0 * (fluxs - 0.49)
    gamma_phi = np.zeros_like(fluxs)
    if params.A_phi is not None:
        gamma_phi += flux_noise_gamma_phi_per_us(params.A_phi, domega_dflux)
    if params.n_th is not None:
        gamma_phi += np.asarray(
            thermal_photon_gamma_phi_per_us(
                params.n_th,
                kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
                chi_over_2pi_mhz=chi,
            ),
            dtype=np.float64,
        )
    T2s = 1.0 / (1.0 / (2.0 * T1s) + gamma_phi)
    return T1s, T2s, domega_dflux, chi
