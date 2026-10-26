from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from zcu_tools.notebook.analysis.t2_curve import (
    FluxResidualWeighting,
    MeasurementErrorPolicy,
    T2CurveContext,
    T2CurveData,
    T2DephasingAnalysis,
    T2FitParams,
    T2FluxCalibration,
    T2RowDiagnostics,
    T2WindowData,
    calibrate_t2_flux,
    fit_t2_curve,
    flux_noise_gamma_phi_per_us,
    make_t2_fit_init,
    mechanisms_to_fixed_params,
    prepare_t2_dephasing_data,
    thermal_photon_gamma_phi_per_us,
)

_KAPPA_OVER_2PI_MHZ = 14.754


def test_fit_t2_curve_fills_nan_errors_in_fit_window() -> None:
    data, _, _ = _synthetic_dephasing_data(A_phi=2.4e-6, n_th=3.0e-3)
    data.fit.T2e_err_us[10] = np.nan

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


def test_prepare_t2_dephasing_data_default_keeps_nan_error_points() -> None:
    fluxs = np.array([0.5, 0.51], dtype=np.float64)
    samples = pd.DataFrame(
        {
            "flux": [-0.5, -0.49],
            "freq_mhz": [350.0, 360.0],
            "t1_us": [60.0, 61.0],
            "t1_err_us": [0.5, 0.5],
            "t2e_us": [30.0, 25.0],
            "t2e_err_us": [np.nan, 0.2],
        }
    )
    calibration = _synthetic_calibration(samples)

    data = prepare_t2_dephasing_data(
        calibration,
        analysis_flux_range=(0.49, 0.52),
    )

    assert len(data.fit.T2e_us) == 2
    assert np.isnan(data.fit.T2e_err_us[0])
    np.testing.assert_allclose(data.window.raw_fluxs, [-0.5, -0.49])
    np.testing.assert_allclose(data.window.fluxs, fluxs)
    # No T2r rows resolved: the public row-diagnostics object is present and
    # empty rather than absent.
    assert isinstance(data.t2r_diagnostics, T2RowDiagnostics)
    assert data.t2r_diagnostics.sample_indexes.shape == (0,)


@pytest.fixture
def null_frequency_analysis() -> T2DephasingAnalysis:
    """Fresh analysis with echo/Ramsey rows spanning integer branches."""
    samples = pd.DataFrame(
        {
            "flux": [0.5, -0.5, -0.49, 0.5, -0.5, -0.49],
            "freq_mhz": [np.nan] * 6,
            "t1_us": [60.0, 61.0, 62.0, 63.0, 64.0, 65.0],
            "t1_err_us": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "t2e_us": [30.0, 28.0, 26.0, np.nan, np.nan, np.nan],
            "t2e_err_us": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            "t2r_us": [np.nan, np.nan, np.nan, 24.0, 22.0, 20.0],
        }
    )
    calibration = _synthetic_calibration(samples)

    cal = calibrate_t2_flux(calibration.context)
    return prepare_t2_dephasing_data(cal, analysis_flux_range=(0.49, 0.53))


def test_null_frequency_rows_reach_the_fit_window(
    null_frequency_analysis: T2DephasingAnalysis,
) -> None:
    data = null_frequency_analysis
    assert len(data.calibration.t2e_df) == 3
    assert len(data.calibration.t2r_df) == 3
    assert len(data.fit.T2e_us) == 3
    assert not np.any(data.window.f01_measured)
    assert np.all(np.isfinite(data.window.f01_mhz))
    np.testing.assert_allclose(np.sort(data.window.raw_fluxs), [-0.5, -0.49, 0.5])
    np.testing.assert_allclose(np.sort(data.window.fluxs), [0.5, 0.5, 0.51])


def test_null_frequency_t2r_diagnostics_preserve_row_identity(
    null_frequency_analysis: T2DephasingAnalysis,
) -> None:
    data = null_frequency_analysis
    # Row-level T2r diagnostics expose every resolved row in deterministic
    # samples_df positional order: stable position, author/aligned coordinates,
    # model-frequency reachability, observed/model source and integer shift.
    diag = data.t2r_diagnostics
    assert isinstance(diag, T2RowDiagnostics)
    assert diag.sample_indexes.tolist() == [3, 4, 5]
    assert diag.in_window.tolist() == [True, True, True]
    np.testing.assert_allclose(diag.raw_fluxs, [0.5, -0.5, -0.49])
    np.testing.assert_allclose(diag.aligned_fluxs, [0.5, 0.5, 0.51])
    assert diag.integer_shifts.tolist() == [0.0, 1.0, 1.0]
    # Null-observed rows carry a directly inspectable finite model f01 and use
    # it as the f01 source.
    assert not np.any(diag.f01_measured)
    assert np.all(np.isfinite(diag.f01_model_mhz))
    np.testing.assert_allclose(diag.f01_model_mhz, data.window.f01_mhz)
    np.testing.assert_allclose(diag.f01_used_mhz, diag.f01_model_mhz)


def test_null_frequency_coverage_and_summary_match_row_diagnostics(
    null_frequency_analysis: T2DephasingAnalysis,
) -> None:
    data = null_frequency_analysis
    diag = data.t2r_diagnostics
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

    # Null freq_mhz rows keep raw coordinates, align into the window, and
    # every one of them reaches a finite model f01 at its aligned flux.
    for coverage in (coverage_t2e, coverage_t2r):
        assert coverage["n_with_f01"] == 0
        assert coverage["n_model_f01"] == 3
        assert coverage["raw_min_flux"] == pytest.approx(-0.5)
        assert coverage["raw_max_flux"] == pytest.approx(0.5)
        assert coverage["aligned_min_flux"] == pytest.approx(0.5)
        assert coverage["aligned_max_flux"] == pytest.approx(0.51)

    # Coverage and summary aggregates derive from the row-level object.
    assert coverage_t2r["n_in_flux_window"] == int(np.count_nonzero(diag.in_window))
    assert coverage_t2r["n_model_f01"] == int(
        np.count_nonzero(~diag.f01_measured & np.isfinite(diag.f01_model_mhz))
    )
    assert coverage_t2r["window_shifted_rows"] == int(
        np.count_nonzero(diag.integer_shifts[diag.in_window] != 0.0)
    )

    # Dephasing summary reports T2r observed/model f01 and integer shifts.
    summary = data.summary_table
    assert (
        summary.loc[summary["metric"] == "window T2r rows", "value"].iloc[0]
        == f"{int(np.count_nonzero(diag.in_window))}/{len(diag.raw_fluxs)}"
    )
    assert (
        summary.loc[
            summary["metric"] == "T2r f01 source (observed/model)", "value"
        ].iloc[0]
        == "0/3"
    )
    assert (
        summary.loc[summary["metric"] == "T2r integer shifts", "value"].iloc[0]
        == "k=0:1 k=1:2"
    )


def test_duplicate_index_labels_keep_positional_row_identity() -> None:
    # Duplicate DataFrame labels must not multiply row identity: T2r is finite
    # only at position 0, and the T2e row lives at the other duplicate-label
    # position (1); position 2 has no finite lifetime. Label-based membership
    # reconstruction would turn the one-row T2r selection into two rows.
    samples = pd.DataFrame(
        {
            "flux": [0.5, -0.5, -0.49],
            "freq_mhz": [np.nan] * 3,
            "t1_us": [60.0, 61.0, 62.0],
            "t1_err_us": [0.5, 0.5, 0.5],
            "t2e_us": [np.nan, 28.0, np.nan],
            "t2e_err_us": [0.2, 0.2, 0.2],
            "t2r_us": [24.0, np.nan, np.nan],
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
    np.testing.assert_allclose(diag.raw_fluxs, [0.5])
    np.testing.assert_allclose(diag.aligned_fluxs, [0.5])
    assert diag.integer_shifts.tolist() == [0.0]
    assert not np.any(diag.f01_measured)
    assert np.all(np.isfinite(diag.f01_model_mhz))
    np.testing.assert_allclose(diag.f01_used_mhz, diag.f01_model_mhz)

    coverage_t2r = data.branch_coverage.loc[
        data.branch_coverage["subset"] == "T2r rows"
    ].iloc[0]
    assert coverage_t2r["n_in_flux_window"] == 1
    summary = data.summary_table
    assert summary.loc[summary["metric"] == "window T2r rows", "value"].iloc[0] == "1/1"

    # The duplicate-label T2e row (position 1) is selected exactly once for
    # branch coverage and half preview.
    coverage_t2e = data.branch_coverage.loc[
        data.branch_coverage["subset"] == "T2e rows"
    ].iloc[0]
    assert coverage_t2e["n_in_flux_window"] == 1
    assert len(data.half_preview) == 1


def test_t2_integer_equivalent_branches_align_to_same_flux() -> None:
    samples = pd.DataFrame(
        {
            "flux": [0.5, -0.5],
            "freq_mhz": [350.0, 351.0],
            "t1_us": [60.0, 61.0],
            "t1_err_us": [0.5, 0.5],
            "t2e_us": [30.0, 28.0],
            "t2e_err_us": [0.2, 0.2],
        }
    )
    calibration = _synthetic_calibration(samples)

    data = prepare_t2_dephasing_data(
        calibration,
        analysis_flux_range=(0.49, 0.53),
    )

    np.testing.assert_allclose(data.window.fluxs, [0.5, 0.5])
    np.testing.assert_allclose(data.window.raw_fluxs, [0.5, -0.5])
    np.testing.assert_allclose(data.window.integer_shifts, [0.0, 1.0])

    # Coverage window/half classification must use the aligned analysis flux, so
    # the shifted author coordinate lands at half flux with the 0.5 row.
    coverage = data.branch_coverage.loc[
        data.branch_coverage["subset"] == "T2e rows"
    ].iloc[0]
    assert coverage["n_in_flux_window"] == 2
    assert coverage["aligned_min_flux"] == pytest.approx(0.5)
    assert coverage["aligned_max_flux"] == pytest.approx(0.5)
    assert coverage["window_below_half_flux"] == 0
    assert coverage["window_at_half_flux"] == 2
    assert coverage["window_above_half_flux"] == 0
    assert coverage["window_shifted_rows"] == 1
    assert coverage["window_shift_min"] == pytest.approx(0.0)
    assert coverage["window_shift_max"] == pytest.approx(1.0)

    # Half preview must keep both author rows after integer alignment.
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
        "aligned flux",
        "in window",
        "integer shift",
    }

    # Dephasing summary reports the branch-shift counts.
    shifts_row = data.summary_table.loc[
        data.summary_table["metric"] == "integer shifts"
    ]
    assert shifts_row["value"].iloc[0] == "k=0:1 k=1:1"


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
        bare_rf=5.8,
        g=0.07,
        samples_df=pd.DataFrame(),
        fluxs=np.empty(0, dtype=np.float64),
        t1_curve_fit=None,
        params_table=pd.DataFrame(),
        samples_preview=pd.DataFrame(),
        available_columns=(),
    )
    calibration = T2FluxCalibration(
        context=context,
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
        f01_mhz=np.full_like(fluxs, 350.0),
        f01_measured=np.ones_like(fluxs, dtype=bool),
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
        f01_observed_mhz=np.empty(0, dtype=np.float64),
        f01_model_mhz=np.empty(0, dtype=np.float64),
        f01_used_mhz=np.empty(0, dtype=np.float64),
        f01_measured=np.empty(0, dtype=bool),
        raw_fluxs=np.empty(0, dtype=np.float64),
        aligned_fluxs=np.empty(0, dtype=np.float64),
        integer_shifts=np.empty(0, dtype=np.float64),
    )


def _synthetic_calibration(samples: pd.DataFrame) -> T2FluxCalibration:
    context = T2CurveContext(
        result_dir="/tmp/result",
        image_dir="/tmp/result/t2_curve",
        samples_filename="samples.csv",
        params=(3.4, 0.9, 0.6),
        bare_rf=5.8,
        g=0.07,
        samples_df=samples,
        fluxs=samples["flux"].to_numpy(dtype=np.float64),
        t1_curve_fit=None,
        params_table=pd.DataFrame(),
        samples_preview=samples,
        available_columns=tuple(samples.columns),
    )
    return T2FluxCalibration(
        context=context,
        t2e_df=samples,
        t2r_df=pd.DataFrame(),
        t2e_mask=np.ones(len(samples), dtype=bool),
        t2r_mask=np.zeros(len(samples), dtype=bool),
        freq_rows=samples,
        summary_table=pd.DataFrame(),
    )
