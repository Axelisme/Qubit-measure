from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest
import zcu_tools.notebook.analysis.t2_curve.workflow as workflow
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from zcu_tools.meta_tool import SampleTableV2Error
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
    T2RowDiagnostics,
    T2WindowData,
    analyze_flux_noise_limit,
    analyze_photon_shot_noise_limit,
    calibrate_t2_flux,
    fit_t2_curve,
    flux_noise_gamma_phi_per_us,
    make_t2_fit_bounds,
    make_t2_fit_init,
    mechanisms_to_fixed_params,
    prepare_t2_dephasing_data,
    run_t2_curve_analysis,
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
    assert config.correct_flux_from_f01_enabled is True


def test_prepare_t2_dephasing_data_default_keeps_nan_error_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fluxs = np.array([0.5, 0.51], dtype=np.float64)
    samples = pd.DataFrame(
        {
            "dev_value": [0.0, 0.01],
            "dev_unit": ["A", "A"],
            "flux_int": [0.5, 0.5],
            "flux_period": [1.0, 1.0],
            "Freq (MHz)": [350.0, 360.0],
            "T1 (us)": [60.0, 61.0],
            "T1err (us)": [0.5, 0.5],
            "T2e (us)": [30.0, 25.0],
            "T2e err (us)": [np.nan, 0.2],
        }
    )
    calibration = _synthetic_calibration(samples)

    def _identity_correction(
        raw_fluxs: NDArray[np.float64],
        _f01_freqs_ghz: NDArray[np.float64],
        *_args: object,
        **_kwargs: object,
    ) -> F01FluxCorrectionResult:
        return F01FluxCorrectionResult(
            raw_fluxs=raw_fluxs,
            corrected_fluxs=raw_fluxs,
            accepted=np.ones_like(raw_fluxs, dtype=bool),
        )

    monkeypatch.setattr(workflow, "correct_flux_from_f01", _identity_correction)

    data = prepare_t2_dephasing_data(calibration, analysis_flux_range=(0.49, 0.52))

    assert len(data.fit.T2e_us) == 2
    assert np.isnan(data.fit.T2e_err_us[0])
    np.testing.assert_allclose(data.window.raw_fluxs, [-0.5, -0.49])
    np.testing.assert_allclose(data.window.fluxs, fluxs)
    np.testing.assert_allclose(data.window.flux_corrections, [0.0, 0.0])
    # No T2r rows resolved: the public row-diagnostics object is present and
    # empty rather than absent.
    assert isinstance(data.t2r_diagnostics, T2RowDiagnostics)
    assert data.t2r_diagnostics.sample_indexes.shape == (0,)
    assert data.t2r_diagnostics.flux_sources == ()


def test_plot_t2_flux_calibration_shows_provenance_categories() -> None:
    data, _, _ = _synthetic_dephasing_data(2.4e-6, 3.0e-3)
    window = T2WindowData(
        fluxs=np.array([0.490, 0.500, 0.510, 0.520, 0.495], dtype=np.float64),
        raw_fluxs=np.array([0.488, 0.500, 0.510, 0.520, 0.495], dtype=np.float64),
        integer_shifts=np.zeros(5, dtype=np.float64),
        flux_sources=("row-frame", "explicit", "row-frame", "row-frame", "row-frame"),
        f01_mhz=data.window.f01_mhz[:5],
        f01_measured=np.array([True, True, False, True, True], dtype=bool),
        f01_correction_applied=np.array([True, False, False, False, False], dtype=bool),
        flux_corrections=np.array([0.002, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        correction_skipped_reason=(
            "",
            "explicit_flux",
            "missing_f01",
            "disabled",
            "rejected",
        ),
        T2e_us=data.window.T2e_us[:5],
        T2e_err_us=data.window.T2e_err_us[:5],
        T1_us=data.window.T1_us[:5],
        T1_err_us=data.window.T1_err_us[:5],
        kept_rows=5,
        source_rows=5,
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
        t2r_diagnostics=_empty_t2r_diagnostics(),
    )

    fig, ax = workflow.plot_t2_flux_calibration(data)

    labels = {collection.get_label() for collection in ax.collections}
    assert "raw flux" in labels
    assert "f01 corrected flux" in labels
    assert "explicit flux (uncorrected)" in labels
    assert "missing f01 (model freq)" in labels
    assert "correction disabled" in labels
    assert "kept raw (rejected)" in labels
    assert "integer aligned flux" in labels
    # One f01-correction segment and one branch-alignment segment per row.
    assert len(ax.lines) == 2 * len(window.fluxs)
    plt.close(fig)


def test_t2e_and_t2r_rows_with_null_freq_reach_analysis() -> None:
    samples = pd.DataFrame(
        {
            "dev_value": [0.0, 0.0, 0.01, 0.0, 0.0, 0.01],
            "dev_unit": ["A", "A", "A", "A", "A", "A"],
            "flux": [0.5, np.nan, np.nan, 0.5, np.nan, np.nan],
            "flux_int": [np.nan, 0.5, np.nan, np.nan, 0.5, np.nan],
            "flux_period": [np.nan, 1.0, np.nan, np.nan, 1.0, np.nan],
            "Freq (MHz)": [np.nan] * 6,
            "T1 (us)": [60.0, 61.0, 62.0, 63.0, 64.0, 65.0],
            "T1err (us)": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "T2e (us)": [30.0, 28.0, 26.0, np.nan, np.nan, np.nan],
            "T2e err (us)": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            "T2r (us)": [np.nan, np.nan, np.nan, 24.0, 22.0, 20.0],
        }
    )
    calibration = _synthetic_calibration(samples)

    cal = calibrate_t2_flux(calibration.context)
    assert len(cal.t2e_df) == 3
    assert len(cal.t2r_df) == 3
    data = prepare_t2_dephasing_data(cal, analysis_flux_range=(0.49, 0.53))

    assert len(data.fit.T2e_us) == 3
    assert not np.any(data.window.f01_measured)
    assert data.window.flux_sources == ("explicit", "row-frame", "fallback-frame")
    assert data.window.correction_skipped_reason == (
        "explicit_flux",
        "missing_f01",
        "missing_f01",
    )
    assert np.all(np.isfinite(data.window.f01_mhz))
    np.testing.assert_allclose(np.sort(data.window.raw_fluxs), [-0.5, -0.49, 0.5])
    np.testing.assert_allclose(np.sort(data.window.fluxs), [0.5, 0.5, 0.51])

    # Row-level T2r diagnostics expose every resolved row in deterministic
    # samples_df positional order: stable position, raw/corrected/aligned
    # coordinates, model-frequency reachability, observed-vs-model source,
    # integer shift, applied flag and exact correction-skip reasons.
    diag = data.t2r_diagnostics
    assert isinstance(diag, T2RowDiagnostics)
    assert diag.sample_indexes.tolist() == [3, 4, 5]
    assert diag.in_window.tolist() == [True, True, True]
    assert diag.flux_sources == ("explicit", "row-frame", "fallback-frame")
    np.testing.assert_allclose(diag.raw_fluxs, [0.5, -0.5, -0.49])
    np.testing.assert_allclose(diag.corrected_fluxs, [0.5, -0.5, -0.49])
    np.testing.assert_allclose(diag.aligned_fluxs, [0.5, 0.5, 0.51])
    assert diag.integer_shifts.tolist() == [0.0, 1.0, 1.0]
    assert not np.any(diag.correction_applied)
    assert diag.correction_skipped_reason == (
        "explicit_flux",
        "missing_f01",
        "missing_f01",
    )
    # Null-observed rows carry a directly inspectable finite model f01 and use
    # it as the f01 source.
    assert not np.any(diag.f01_measured)
    assert np.all(np.isfinite(diag.f01_model_mhz))
    np.testing.assert_allclose(diag.f01_model_mhz, data.window.f01_mhz)
    np.testing.assert_allclose(diag.f01_used_mhz, diag.f01_model_mhz)

    coverage_t2e = data.branch_coverage.loc[
        data.branch_coverage["subset"] == "T2e rows"
    ].iloc[0]
    coverage_t2r = data.branch_coverage.loc[
        data.branch_coverage["subset"] == "T2r rows"
    ].iloc[0]
    assert coverage_t2e["n_in_flux_window"] == 3
    assert coverage_t2r["n_in_flux_window"] == 3
    assert coverage_t2e["window_shifted_rows"] == 2
    assert coverage_t2r["window_shifted_rows"] == 2

    # Per-subset provenance must distinguish explicit / row-frame / fallback.
    for coverage in (coverage_t2e, coverage_t2r):
        assert coverage["n_explicit"] == 1
        assert coverage["n_row_frame"] == 1
        assert coverage["n_fallback_frame"] == 1

    # Null Freq (MHz) rows keep raw coordinates, align into the window, and
    # every one of them reaches a finite model f01 at its aligned flux.
    for coverage in (coverage_t2e, coverage_t2r):
        assert coverage["n_with_f01"] == 0
        assert coverage["n_model_f01"] == 3
        assert coverage["raw_min_flux"] == pytest.approx(-0.5)
        assert coverage["raw_max_flux"] == pytest.approx(0.5)
        assert coverage["aligned_min_flux"] == pytest.approx(0.5)
        assert coverage["aligned_max_flux"] == pytest.approx(0.51)

    # Correction-skip reasons are exact per subset, not a collapsed boolean.
    for coverage in (coverage_t2e, coverage_t2r):
        assert coverage["skipped"] == 3
        assert coverage["skipped_explicit_flux"] == 1
        assert coverage["skipped_missing_f01"] == 2
        assert coverage["skipped_rejected"] == 0
        assert coverage["skipped_disabled"] == 0

    # Coverage and summary aggregates derive from the row-level object.
    assert coverage_t2r["n_in_flux_window"] == int(np.count_nonzero(diag.in_window))
    assert coverage_t2r["n_model_f01"] == int(
        np.count_nonzero(~diag.f01_measured & np.isfinite(diag.f01_model_mhz))
    )
    assert coverage_t2r["skipped_explicit_flux"] == sum(
        1 for reason in diag.correction_skipped_reason if reason == "explicit_flux"
    )
    assert coverage_t2r["window_shifted_rows"] == int(
        np.count_nonzero(diag.integer_shifts[diag.in_window] != 0.0)
    )

    # Dephasing summary reports T2r provenance, observed/model f01 and exact
    # skip reasons instead of T2e-only diagnostics.
    summary = data.summary_table
    assert (
        summary.loc[summary["metric"] == "window T2r rows", "value"].iloc[0]
        == f"{int(np.count_nonzero(diag.in_window))}/{len(diag.raw_fluxs)}"
    )
    assert (
        summary.loc[
            summary["metric"] == "T2r flux sources (explicit/row/fallback)",
            "value",
        ].iloc[0]
        == "1/1/1"
    )
    assert (
        summary.loc[
            summary["metric"] == "T2r f01 source (observed/model)", "value"
        ].iloc[0]
        == "0/3"
    )
    assert (
        summary.loc[
            summary["metric"] == "T2r f01 correction skipped reasons", "value"
        ].iloc[0]
        == "explicit_flux=1 missing_f01=2"
    )
    assert (
        summary.loc[summary["metric"] == "T2r integer shifts", "value"].iloc[0]
        == "k=0:1 k=1:2"
    )


def test_duplicate_index_labels_keep_positional_row_identity() -> None:
    # Duplicate DataFrame labels must not multiply row identity: T2r is finite
    # only at position 0, and the T2e row lives at the other duplicate-label
    # position (1); position 2 is a fallback row. Label-based membership
    # reconstruction would turn the one-row T2r selection into two rows.
    samples = pd.DataFrame(
        {
            "dev_value": [0.0, 0.0, 0.01],
            "dev_unit": ["A", "A", "A"],
            "flux": [0.5, np.nan, np.nan],
            "flux_int": [np.nan, 0.5, np.nan],
            "flux_period": [np.nan, 1.0, np.nan],
            "Freq (MHz)": [np.nan] * 3,
            "T1 (us)": [60.0, 61.0, 62.0],
            "T1err (us)": [0.5, 0.5, 0.5],
            "T2e (us)": [np.nan, 28.0, np.nan],
            "T2e err (us)": [0.2, 0.2, 0.2],
            "T2r (us)": [24.0, np.nan, np.nan],
        },
        index=[7, 7, 8],
    )
    cal = calibrate_t2_flux(_synthetic_calibration(samples).context)
    assert cal.t2e_mask.tolist() == [False, True, False]
    assert cal.t2r_mask.tolist() == [True, False, False]
    data = prepare_t2_dephasing_data(cal, analysis_flux_range=(0.49, 0.53))

    # Exactly one T2r row, at position 0; the duplicate-label position 1 is
    # never attributed to T2r.
    diag = data.t2r_diagnostics
    assert diag.sample_indexes.tolist() == [0]
    assert len(diag.raw_fluxs) == 1
    assert diag.in_window.tolist() == [True]
    assert diag.flux_sources == ("explicit",)
    np.testing.assert_allclose(diag.raw_fluxs, [0.5])
    np.testing.assert_allclose(diag.corrected_fluxs, [0.5])
    np.testing.assert_allclose(diag.aligned_fluxs, [0.5])
    assert diag.integer_shifts.tolist() == [0.0]
    assert not np.any(diag.f01_measured)
    assert np.all(np.isfinite(diag.f01_model_mhz))
    np.testing.assert_allclose(diag.f01_used_mhz, diag.f01_model_mhz)
    assert not np.any(diag.correction_applied)
    assert diag.correction_skipped_reason == ("explicit_flux",)

    coverage_t2r = data.branch_coverage.loc[
        data.branch_coverage["subset"] == "T2r rows"
    ].iloc[0]
    assert coverage_t2r["n_in_flux_window"] == 1
    assert coverage_t2r["n_explicit"] == 1
    assert coverage_t2r["n_row_frame"] == 0
    assert coverage_t2r["skipped"] == 1
    assert coverage_t2r["skipped_explicit_flux"] == 1
    summary = data.summary_table
    assert summary.loc[summary["metric"] == "window T2r rows", "value"].iloc[0] == "1/1"

    # The duplicate-label T2e row (position 1) is selected exactly once for
    # branch coverage and half preview.
    coverage_t2e = data.branch_coverage.loc[
        data.branch_coverage["subset"] == "T2e rows"
    ].iloc[0]
    assert coverage_t2e["n_in_flux_window"] == 1
    assert coverage_t2e["n_row_frame"] == 1
    assert coverage_t2e["skipped_missing_f01"] == 1
    assert len(data.half_preview) == 1


def test_t2_calibration_reports_resolution_provenance() -> None:
    samples = pd.DataFrame(
        {
            "dev_value": [0.0, 0.0, 0.01],
            "dev_unit": ["A", "A", "A"],
            "flux": [0.5, np.nan, np.nan],
            "flux_int": [np.nan, 0.5, np.nan],
            "flux_period": [np.nan, 1.0, np.nan],
            "Freq (MHz)": [350.0, 351.0, 352.0],
            "T1 (us)": [60.0, 61.0, 62.0],
            "T1err (us)": [0.5, 0.5, 0.5],
            "T2e (us)": [30.0, 28.0, 26.0],
            "T2e err (us)": [0.2, 0.2, 0.2],
        }
    )
    cal = calibrate_t2_flux(_synthetic_calibration(samples).context)

    assert cal.resolution.sources == (
        "explicit",
        "row-frame",
        "fallback-frame",
    )
    np.testing.assert_allclose(cal.resolution.values, [0.5, -0.5, -0.49])
    assert not hasattr(cal, "current_scale")


def test_t2_unresolved_flux_rows_fail_fast() -> None:
    samples = pd.DataFrame(
        {
            "dev_value": [0.0, 0.01],
            "dev_unit": ["V", "V"],
            "Freq (MHz)": [np.nan, np.nan],
            "T1 (us)": [60.0, 61.0],
            "T1err (us)": [0.5, 0.5],
            "T2e (us)": [30.0, 28.0],
            "T2e err (us)": [0.2, 0.2],
        }
    )
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

    with pytest.raises(SampleTableV2Error, match="unable to resolve flux"):
        calibrate_t2_flux(context, fallback_frame_unit="A")


def test_t2_integer_equivalent_branches_align_to_same_flux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = pd.DataFrame(
        {
            "dev_value": [0.0, 0.0],
            "dev_unit": ["A", "A"],
            "flux": [0.5, np.nan],
            "flux_int": [np.nan, 0.5],
            "flux_period": [np.nan, 1.0],
            "Freq (MHz)": [350.0, 351.0],
            "T1 (us)": [60.0, 61.0],
            "T1err (us)": [0.5, 0.5],
            "T2e (us)": [30.0, 28.0],
            "T2e err (us)": [0.2, 0.2],
        }
    )
    calibration = _synthetic_calibration(samples)

    def _identity_correction(
        raw_fluxs: NDArray[np.float64],
        _f01_freqs_ghz: NDArray[np.float64],
        *_args: object,
        **_kwargs: object,
    ) -> F01FluxCorrectionResult:
        return F01FluxCorrectionResult(
            raw_fluxs=raw_fluxs,
            corrected_fluxs=raw_fluxs,
            accepted=np.ones_like(raw_fluxs, dtype=bool),
        )

    monkeypatch.setattr(workflow, "correct_flux_from_f01", _identity_correction)

    data = prepare_t2_dephasing_data(calibration, analysis_flux_range=(0.49, 0.53))

    np.testing.assert_allclose(data.window.fluxs, [0.5, 0.5])
    np.testing.assert_allclose(data.window.raw_fluxs, [0.5, -0.5])
    np.testing.assert_allclose(data.window.integer_shifts, [0.0, 1.0])

    # Coverage window/half classification must use the aligned analysis flux, so
    # the shifted derived row lands at half flux with the explicit 0.5 row.
    coverage = data.branch_coverage.loc[
        data.branch_coverage["subset"] == "T2e rows"
    ].iloc[0]
    assert coverage["n_in_flux_window"] == 2
    assert coverage["corr_min_flux"] == pytest.approx(-0.5)
    assert coverage["corr_max_flux"] == pytest.approx(0.5)
    assert coverage["aligned_min_flux"] == pytest.approx(0.5)
    assert coverage["aligned_max_flux"] == pytest.approx(0.5)
    assert coverage["window_below_half_flux"] == 0
    assert coverage["window_at_half_flux"] == 2
    assert coverage["window_above_half_flux"] == 0
    assert coverage["window_shifted_rows"] == 1
    assert coverage["window_shift_min"] == pytest.approx(0.0)
    assert coverage["window_shift_max"] == pytest.approx(1.0)

    # Half preview must keep both aligned rows (explicit 0.5 and shifted -0.5).
    assert len(data.half_preview) == 2
    np.testing.assert_allclose(
        data.half_preview["aligned flux"].to_numpy(dtype=np.float64),
        [0.5, 0.5],
    )
    np.testing.assert_allclose(
        data.half_preview["integer shift"].to_numpy(dtype=np.float64),
        [0.0, 1.0],
    )
    assert set(data.half_preview.columns) >= {
        "raw flux",
        "f01-corrected flux",
        "aligned flux",
        "in window",
        "integer shift",
    }

    # Dephasing summary reports the branch-shift counts.
    shifts_row = data.summary_table.loc[
        data.summary_table["metric"] == "integer shifts"
    ]
    assert shifts_row["value"].iloc[0] == "k=0:1 k=1:1"


def test_t2_explicit_rows_are_not_auto_corrected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = pd.DataFrame(
        {
            "dev_value": [0.0, 0.0],
            "dev_unit": ["A", "A"],
            "flux": [0.5, np.nan],
            "flux_int": [np.nan, 0.5],
            "flux_period": [np.nan, 1.0],
            "Freq (MHz)": [350.0, 351.0],
            "T1 (us)": [60.0, 61.0],
            "T1err (us)": [0.5, 0.5],
            "T2e (us)": [30.0, 28.0],
            "T2e err (us)": [0.2, 0.2],
        }
    )
    calibration = _synthetic_calibration(samples)

    def _fake_correct(
        raw_fluxs: NDArray[np.float64],
        _f01_freqs_ghz: NDArray[np.float64],
        *_args: object,
        **_kwargs: object,
    ) -> F01FluxCorrectionResult:
        return F01FluxCorrectionResult(
            raw_fluxs=raw_fluxs,
            corrected_fluxs=raw_fluxs + 0.002,
            accepted=np.ones_like(raw_fluxs, dtype=bool),
        )

    monkeypatch.setattr(workflow, "correct_flux_from_f01", _fake_correct)

    data = prepare_t2_dephasing_data(calibration, analysis_flux_range=(0.49, 0.53))

    assert data.window.flux_sources == ("explicit", "row-frame")
    np.testing.assert_array_equal(data.window.f01_correction_applied, [False, True])
    np.testing.assert_allclose(data.window.raw_fluxs, [0.5, -0.5])
    np.testing.assert_allclose(data.window.flux_corrections, [0.0, 0.002])
    np.testing.assert_allclose(data.window.fluxs, [0.5, 0.502])
    assert data.window.correction_skipped_reason == ("explicit_flux", "")


def test_t2_correction_disabled_preserves_raw_flux_and_reports_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = pd.DataFrame(
        {
            "dev_value": [0.0, 0.0],
            "dev_unit": ["A", "A"],
            "flux": [0.5, np.nan],
            "flux_int": [np.nan, 0.5],
            "flux_period": [np.nan, 1.0],
            "Freq (MHz)": [350.0, 351.0],
            "T1 (us)": [60.0, 61.0],
            "T1err (us)": [0.5, 0.5],
            "T2e (us)": [30.0, 28.0],
            "T2e err (us)": [0.2, 0.2],
        }
    )
    calibration = _synthetic_calibration(samples)

    def _must_not_run(
        *_args: object,
        **_kwargs: object,
    ) -> F01FluxCorrectionResult:
        raise AssertionError("f01 correction must not run when disabled")

    monkeypatch.setattr(workflow, "correct_flux_from_f01", _must_not_run)

    data = prepare_t2_dephasing_data(
        calibration,
        analysis_flux_range=(0.49, 0.53),
        correct_flux_from_f01_enabled=False,
    )

    assert not np.any(data.window.f01_correction_applied)
    np.testing.assert_allclose(data.window.flux_corrections, [0.0, 0.0])
    np.testing.assert_allclose(data.window.raw_fluxs, [0.5, -0.5])
    np.testing.assert_allclose(data.window.fluxs, [0.5, 0.5])
    np.testing.assert_allclose(data.window.integer_shifts, [0.0, 1.0])
    assert data.window.correction_skipped_reason == ("disabled", "disabled")
    assert (
        data.summary_table.loc[
            data.summary_table["metric"] == "f01 correction skipped reasons",
            "value",
        ].iloc[0]
        == "disabled=2"
    )
    coverage = data.branch_coverage.loc[
        data.branch_coverage["subset"] == "T2e rows"
    ].iloc[0]
    assert coverage["accepted"] == 0
    assert coverage["skipped"] == 2


def test_run_t2_curve_analysis_threads_correction_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    config = T2CurveAnalysisConfig(
        result_dir="/tmp/result",
        correct_flux_from_f01_enabled=False,
        progress=False,
        verbose=False,
        display_tables=False,
        save_figures=False,
        show_figures=False,
        fit_bounds={"A_phi": (0.0, 1e-3), "n_th": (0.0, 1.0)},
    )

    def _fake_prepare(calibration: object, **kwargs: object) -> object:
        captured["prepare_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(workflow, "load_t2_curve_context", lambda **_kwargs: object())
    monkeypatch.setattr(
        workflow, "calibrate_t2_flux", lambda _context, **_kwargs: object()
    )
    monkeypatch.setattr(workflow, "prepare_t2_dephasing_data", _fake_prepare)
    monkeypatch.setattr(
        workflow,
        "analyze_flux_noise_limit",
        lambda _data, **_kwargs: object(),
    )
    monkeypatch.setattr(
        workflow,
        "analyze_photon_shot_noise_limit",
        lambda _data, **_kwargs: object(),
    )
    monkeypatch.setattr(workflow, "make_t2_fit_init", lambda **_kwargs: object())
    monkeypatch.setattr(workflow, "fit_t2_curve", lambda _data, **_kwargs: object())
    monkeypatch.setattr(
        workflow,
        "build_t2_channel_curves",
        lambda _combined_fit, **_kwargs: object(),
    )
    monkeypatch.setattr(workflow, "collect_t2_curve_result", lambda **_kwargs: object())

    run_t2_curve_analysis(config)

    prepare_kwargs = cast(dict[str, object], captured["prepare_kwargs"])
    assert prepare_kwargs["correct_flux_from_f01_enabled"] is False


def test_t2_workflow_has_no_current_scale_vocabulary() -> None:
    import zcu_tools.notebook.analysis.fit_tools as fit_tools

    assert not hasattr(workflow, "choose_current_scale")
    assert not hasattr(workflow, "choose_current_scale_from_f01")
    assert not hasattr(fit_tools, "choose_current_scale_from_f01")
    assert "calibrated mA" not in workflow._REQUIRED_COLUMNS


def _synthetic_dephasing_data(
    A_phi: float,
    n_th: float,
) -> tuple[T2DephasingAnalysis, NDArray[np.float64], NDArray[np.float64]]:
    fluxs = np.linspace(0.49, 0.53, 30, dtype=np.float64)
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
        resolution=workflow.SampleFluxResolution(
            values=np.asarray(fluxs, dtype=np.float64),
            sources=("row-frame",) * len(fluxs),
        ),
        fallback_frame=workflow.SampleFluxFrame("A", 0.5, 1.0),
        t2e_df=pd.DataFrame(),
        t2r_df=pd.DataFrame(),
        t2e_mask=np.zeros(0, dtype=bool),
        t2r_mask=np.zeros(0, dtype=bool),
        freq_rows=pd.DataFrame(),
        summary_table=pd.DataFrame(),
    )
    window = T2WindowData(
        fluxs=fluxs,
        raw_fluxs=fluxs,
        integer_shifts=np.zeros_like(fluxs),
        flux_sources=("row-frame",) * len(fluxs),
        f01_mhz=np.full_like(fluxs, 350.0),
        f01_measured=np.ones_like(fluxs, dtype=bool),
        f01_correction_applied=np.ones_like(fluxs, dtype=bool),
        flux_corrections=np.zeros_like(fluxs),
        correction_skipped_reason=("",) * len(fluxs),
        T2e_us=T2e_us,
        T2e_err_us=T2e_err_us,
        T1_us=T1_us,
        T1_err_us=T1_err_us,
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
        t2r_diagnostics=_empty_t2r_diagnostics(),
    )
    return data, domega_dflux, chi


def _empty_t2r_diagnostics() -> T2RowDiagnostics:
    return T2RowDiagnostics(
        sample_indexes=np.empty(0, dtype=np.int64),
        in_window=np.empty(0, dtype=bool),
        flux_sources=(),
        f01_observed_mhz=np.empty(0, dtype=np.float64),
        f01_model_mhz=np.empty(0, dtype=np.float64),
        f01_used_mhz=np.empty(0, dtype=np.float64),
        f01_measured=np.empty(0, dtype=bool),
        raw_fluxs=np.empty(0, dtype=np.float64),
        corrected_fluxs=np.empty(0, dtype=np.float64),
        aligned_fluxs=np.empty(0, dtype=np.float64),
        integer_shifts=np.empty(0, dtype=np.float64),
        correction_applied=np.empty(0, dtype=bool),
        correction_skipped_reason=(),
    )


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
    resolution = workflow.resolve_sample_flux(
        samples, fallback_frame=workflow.SampleFluxFrame("A", 0.5, 1.0)
    )
    return T2FluxCalibration(
        context=context,
        resolution=resolution,
        fallback_frame=workflow.SampleFluxFrame("A", 0.5, 1.0),
        t2e_df=samples,
        t2r_df=pd.DataFrame(),
        t2e_mask=np.ones(len(samples), dtype=bool),
        t2r_mask=np.zeros(len(samples), dtype=bool),
        freq_rows=samples,
        summary_table=pd.DataFrame(),
    )
