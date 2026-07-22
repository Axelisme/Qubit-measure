from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import zcu_tools.notebook.analysis.t2_curve.workflow as workflow
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from zcu_tools.notebook.analysis.fit_tools import F01FluxCorrectionResult
from zcu_tools.notebook.analysis.t2_curve import (
    FluxResidualWeighting,
    MeasurementErrorPolicy,
    T2CurveAnalysisConfig,
    T2CurveContext,
    T2CurveData,
    T2DephasingAnalysis,
    T2FitParams,
    T2FluxCalibration,
    T2WindowData,
    analyze_flux_noise_limit,
    analyze_photon_shot_noise_limit,
    fit_t2_curve,
    flux_noise_gamma_phi_per_us,
    make_t2_fit_bounds,
    make_t2_fit_init,
    mechanisms_to_fixed_params,
    prepare_t2_dephasing_data,
    thermal_photon_gamma_phi_per_us,
)

_KAPPA_OVER_2PI_MHZ = 14.754


def test_mechanism_probes_feed_combined_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    true_A_phi = 2.4e-6
    true_n_th = 3.0e-3
    data, domega_dflux, chi = _synthetic_dephasing_data(true_A_phi, true_n_th)

    monkeypatch.setattr(
        workflow, "predict_domega_dflux", lambda *_args, **_kwargs: domega_dflux
    )
    monkeypatch.setattr(
        workflow,
        "dispersive_chi01_over_2pi_mhz",
        lambda *_args, **_kwargs: chi,
    )

    flux_probe = analyze_flux_noise_limit(
        data,
        readout_kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
        assumed_n_th=true_n_th,
    )
    photon_probe = analyze_photon_shot_noise_limit(
        data,
        readout_kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
        assumed_A_phi=true_A_phi,
    )
    init = make_t2_fit_init(
        active_mechanisms=("flux_noise", "photon_shot_noise"),
        flux_probe=flux_probe,
        photon_probe=photon_probe,
    )
    combined = fit_t2_curve(
        data,
        readout_kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
        init=init,
        bounds=make_t2_fit_bounds(init),
        fixed=mechanisms_to_fixed_params(()),
    )

    assert flux_probe.A_phi_fit == pytest.approx(true_A_phi, rel=1e-6)
    assert photon_probe.n_th_fit == pytest.approx(true_n_th, rel=1e-6)
    assert combined.fit_result.params.A_phi == pytest.approx(true_A_phi, rel=1e-6)
    assert combined.fit_result.params.n_th == pytest.approx(true_n_th, rel=1e-6)


def test_photon_probe_uses_pointwise_minimum_for_combined_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, domega_dflux, chi = _synthetic_dephasing_data(A_phi=2.4e-6, n_th=3.0e-3)

    monkeypatch.setattr(
        workflow, "predict_domega_dflux", lambda *_args, **_kwargs: domega_dflux
    )
    monkeypatch.setattr(
        workflow,
        "dispersive_chi01_over_2pi_mhz",
        lambda *_args, **_kwargs: chi,
    )

    photon_probe = analyze_photon_shot_noise_limit(
        data,
        readout_kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
        assumed_A_phi=0.0,
    )
    init = make_t2_fit_init(
        active_mechanisms=("photon_shot_noise",),
        photon_probe=photon_probe,
    )

    pointwise_min = float(np.nanmin(photon_probe.pointwise_table["n_th"]))
    assert photon_probe.n_th_init == pytest.approx(pointwise_min)
    assert photon_probe.n_th_fit > photon_probe.n_th_init
    assert init.n_th == pytest.approx(photon_probe.n_th_init)


def test_flux_noise_probe_ignores_near_zero_sensitivity_for_pointwise_upper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, domega_dflux, chi = _synthetic_dephasing_data(A_phi=2.4e-6, n_th=0.0)
    domega_dflux[0] = 0.0

    monkeypatch.setattr(
        workflow, "predict_domega_dflux", lambda *_args, **_kwargs: domega_dflux
    )
    monkeypatch.setattr(
        workflow,
        "dispersive_chi01_over_2pi_mhz",
        lambda *_args, **_kwargs: chi,
    )

    flux_probe = analyze_flux_noise_limit(
        data,
        readout_kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
        min_sensitivity_fraction=1e-3,
    )

    assert np.isnan(flux_probe.pointwise_table["A_phi (Phi0/sqrtHz)"].iloc[0])
    assert flux_probe.A_phi_upper < 1e-3


def test_fit_t2_curve_fills_nan_errors_in_fit_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, domega_dflux, chi = _synthetic_dephasing_data(A_phi=2.4e-6, n_th=3.0e-3)
    data.fit.T2e_err_us[10] = np.nan

    monkeypatch.setattr(
        workflow,
        "predict_domega_dflux",
        lambda _params, fluxs, **_kwargs: np.interp(
            fluxs, data.fit.fluxs, domega_dflux
        ),
    )
    monkeypatch.setattr(
        workflow,
        "dispersive_chi01_over_2pi_mhz",
        lambda _params, fluxs, *_args, **_kwargs: np.interp(fluxs, data.fit.fluxs, chi),
    )

    combined = fit_t2_curve(
        data,
        readout_kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
        init=T2FitParams(A_phi=2.0e-6, n_th=1.0e-3),
        T2_error_policy=MeasurementErrorPolicy(nan_policy="bin_median"),
        flux_weighting=FluxResidualWeighting(mode="equal_flux_bin", bin_width=0.004),
    )

    assert len(combined.fit_fluxs) == len(data.fit.fluxs)
    assert combined.fit_result.T2_error_resolution is not None
    assert combined.fit_result.T2_error_resolution.nan_mask[10]
    assert np.isfinite(combined.fit_result.T2_error_resolution.effective_errors[10])
    assert combined.fit_result.flux_weights.effective_observation_count < len(
        data.fit.fluxs
    )


def test_make_t2_fit_init_can_select_partial_mechanisms() -> None:
    init = make_t2_fit_init(
        active_mechanisms=("flux_noise",),
        A_phi=2.0e-6,
        n_th=1.0e-3,
    )

    assert init == T2FitParams(A_phi=2.0e-6, n_th=None)
    assert mechanisms_to_fixed_params(("flux_noise",)) == ("A_phi",)
    assert mechanisms_to_fixed_params(("photon_shot_noise",)) == ("n_th",)


def test_t2_curve_analysis_config_defaults_match_weighted_fit_contract() -> None:
    config = T2CurveAnalysisConfig(result_dir="/tmp/result")

    assert config.use_weighted_points_only is False
    assert config.T1_error_policy.fallback_error == pytest.approx(1.0)
    assert config.loss == "linear"


def test_prepare_t2_dephasing_data_default_keeps_nan_error_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fluxs = np.array([0.5, 0.51], dtype=np.float64)
    values = fluxs - 0.5
    samples = pd.DataFrame(
        {
            "calibrated mA": values,
            "Freq (MHz)": [350.0, 360.0],
            "T1 (us)": [60.0, 61.0],
            "T1err (us)": [0.5, 0.5],
            "T2e (us)": [30.0, 25.0],
            "T2e err (us)": [np.nan, 0.2],
        }
    )
    calibration = _synthetic_calibration(samples)

    def _identity_correction(
        dev_values: NDArray[np.float64],
        _f01_freqs: NDArray[np.float64],
        *_args: object,
        **_kwargs: object,
    ) -> F01FluxCorrectionResult:
        return F01FluxCorrectionResult(
            raw_fluxs=fluxs,
            corrected_fluxs=fluxs,
            corrected_dev_values=dev_values,
            candidate_biases=np.zeros_like(dev_values),
            candidate_flux_corrections=np.zeros_like(dev_values),
            accepted=np.ones_like(dev_values, dtype=bool),
        )

    monkeypatch.setattr(workflow, "correct_flux_from_f01", _identity_correction)

    data = prepare_t2_dephasing_data(calibration, analysis_flux_range=(0.49, 0.52))

    assert len(data.fit.T2e_us) == 2
    assert np.isnan(data.fit.T2e_err_us[0])
    np.testing.assert_allclose(data.window.raw_fluxs, fluxs)
    np.testing.assert_allclose(data.window.flux_corrections, [0.0, 0.0])


def test_plot_t2_flux_calibration_shows_before_after_positions() -> None:
    data, _, _ = _synthetic_dephasing_data(2.4e-6, 3.0e-3)
    window = T2WindowData(
        current_raw=data.window.current_raw[:3],
        values=data.window.values[:3],
        raw_fluxs=np.array([0.488, 0.500, 0.514], dtype=np.float64),
        fluxs=np.array([0.490, 0.500, 0.510], dtype=np.float64),
        f01_mhz=data.window.f01_mhz[:3],
        T2e_us=data.window.T2e_us[:3],
        T2e_err_us=data.window.T2e_err_us[:3],
        T1_us=data.window.T1_us[:3],
        T1_err_us=data.window.T1_err_us[:3],
        f01_correction_accepted=np.array([True, False, True]),
        flux_corrections=np.array([0.002, 0.000, -0.004], dtype=np.float64),
        kept_rows=3,
        source_rows=3,
    )
    data = T2DephasingAnalysis(
        calibration=data.calibration,
        analysis_flux_range=data.analysis_flux_range,
        max_abs_flux_correction=data.max_abs_flux_correction,
        max_rel_t2e_err=data.max_rel_t2e_err,
        use_weighted_points_only=data.use_weighted_points_only,
        window=window,
        sample=data.sample,
        fit=data.fit,
        branch_coverage=data.branch_coverage,
        half_preview=data.half_preview,
        summary_table=data.summary_table,
    )

    fig, ax = workflow.plot_t2_flux_calibration(data)

    labels = {collection.get_label() for collection in ax.collections}
    assert "raw flux" in labels
    assert "f01 corrected flux" in labels
    assert "kept raw flux" in labels
    assert len(ax.lines) == len(window.fluxs)
    plt.close(fig)


def _synthetic_dephasing_data(
    A_phi: float,
    n_th: float,
) -> tuple[T2DephasingAnalysis, NDArray[np.float64], NDArray[np.float64]]:
    fluxs = np.linspace(0.49, 0.53, 30, dtype=np.float64)
    values = fluxs.copy()
    T1_us = 60.0 + 5.0 * (fluxs - 0.49)
    T1_err_us = np.full_like(fluxs, 0.5)
    T2e_err_us = np.full_like(fluxs, 0.2)
    domega_dflux = np.linspace(1_000.0, 18_000.0, len(fluxs), dtype=np.float64)
    chi = 2.8 + 25.0 * (fluxs - 0.49)
    gamma_phi = flux_noise_gamma_phi_per_us(A_phi, domega_dflux)
    gamma_phi += np.asarray(
        thermal_photon_gamma_phi_per_us(
            n_th,
            kappa_over_2pi_mhz=_KAPPA_OVER_2PI_MHZ,
            chi_over_2pi_mhz=chi,
        ),
        dtype=np.float64,
    )
    T2e_us = 1.0 / (1.0 / (2.0 * T1_us) + gamma_phi)
    gamma_phi_err = np.sqrt(
        (T2e_err_us / T2e_us**2) ** 2 + (0.5 * T1_err_us / T1_us**2) ** 2
    )
    curve = T2CurveData(
        values=values,
        fluxs=fluxs,
        f01_mhz=np.full_like(fluxs, 350.0),
        T1_us=T1_us,
        T1_err_us=T1_err_us,
        T2e_us=T2e_us,
        T2e_err_us=T2e_err_us,
        gamma_phi_per_us=gamma_phi,
        gamma_phi_err_per_us=gamma_phi_err,
        Tphi_us=1.0 / gamma_phi,
    )
    context = T2CurveContext(
        result_dir="/tmp/result",
        image_dir="/tmp/result/t2_curve",
        samples_filename="samples.csv",
        params=(3.4, 0.9, 0.6),
        flux_half=0.5,
        flux_int=0.0,
        flux_period=1.0,
        bare_rf=5.8,
        g=0.07,
        samples_df=pd.DataFrame(),
        t1_curve_fit=None,
        params_table=pd.DataFrame(),
        samples_preview=pd.DataFrame(),
        available_columns=(),
    )
    calibration = T2FluxCalibration(
        context=context,
        current_scale=1.0,
        scale_report=pd.DataFrame(),
        t2e_df=pd.DataFrame(),
        t2r_df=pd.DataFrame(),
        freq_rows=pd.DataFrame(),
        summary_table=pd.DataFrame(),
    )
    window = T2WindowData(
        current_raw=values,
        values=values,
        raw_fluxs=fluxs,
        fluxs=fluxs,
        f01_mhz=np.full_like(fluxs, 350.0),
        T2e_us=T2e_us,
        T2e_err_us=T2e_err_us,
        T1_us=T1_us,
        T1_err_us=T1_err_us,
        f01_correction_accepted=np.ones_like(fluxs, dtype=bool),
        flux_corrections=np.zeros_like(fluxs),
        kept_rows=len(fluxs),
        source_rows=len(fluxs),
    )
    data = T2DephasingAnalysis(
        calibration=calibration,
        analysis_flux_range=(0.49, 0.53),
        max_abs_flux_correction=0.03,
        max_rel_t2e_err=0.5,
        use_weighted_points_only=True,
        window=window,
        sample=curve,
        fit=curve,
        branch_coverage=pd.DataFrame(),
        half_preview=pd.DataFrame(),
        summary_table=pd.DataFrame(),
    )
    return data, domega_dflux, chi


def _synthetic_calibration(samples: pd.DataFrame) -> T2FluxCalibration:
    context = T2CurveContext(
        result_dir="/tmp/result",
        image_dir="/tmp/result/t2_curve",
        samples_filename="samples.csv",
        params=(3.4, 0.9, 0.6),
        flux_half=0.0,
        flux_int=0.5,
        flux_period=1.0,
        bare_rf=5.8,
        g=0.07,
        samples_df=samples,
        t1_curve_fit=None,
        params_table=pd.DataFrame(),
        samples_preview=samples,
        available_columns=tuple(samples.columns),
    )
    return T2FluxCalibration(
        context=context,
        current_scale=1.0,
        scale_report=pd.DataFrame(),
        t2e_df=samples,
        t2r_df=pd.DataFrame(),
        freq_rows=samples,
        summary_table=pd.DataFrame(),
    )
