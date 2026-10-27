from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import zcu_tools.notebook.analysis.t1_curve.workflow as workflow
from zcu_tools.notebook.analysis.t1_curve import (
    PurcellEffectParams,
    T1CurveContext,
    T1CurveData,
    T1FitParams,
    T1FitResult,
    T1FluxCalibration,
    T1MechanismOptions,
    T1PreparedData,
    analyze_t1_mechanism_limit,
    build_t1_channel_curves,
    calculate_purcell_t1_limit,
    calibrate_t1_flux,
    fit_t1_curve,
    load_t1_curve_context,
    load_t1_purcell_params,
    make_t1_fit_init,
    mechanisms_to_fixed_params,
    plot_t1_mechanism_limit,
    prepare_t1_curve_data,
    subtract_relaxation_limit,
)


@pytest.mark.parametrize("Temp", [0.0, -0.01, np.nan, np.inf, -np.inf])
def test_purcell_params_reject_invalid_temperature(Temp: float) -> None:
    with pytest.raises(ValueError, match="Temp must be positive and finite"):
        PurcellEffectParams(kappa_ghz=0.001, bare_rf=7.2, g=0.03, Temp=Temp)


@pytest.mark.parametrize("Temp, expected", [(None, 0.04), (0.08, 0.08)])
def test_purcell_params_resolve_own_or_qubit_temperature(
    Temp: float | None, expected: float
) -> None:
    purcell = PurcellEffectParams(kappa_ghz=0.001, bare_rf=7.2, g=0.03, Temp=Temp)

    assert purcell.resolve_Temp(0.04) == expected


def test_load_t1_curve_context_reads_params_and_samples(tmp_path) -> None:
    (tmp_path / "params.json").write_text(
        """
{
  "fluxdep_fit": {
    "params": {"EJ": 3.4, "EC": 0.9, "EL": 0.6},
    "flux_half": 0.0,
    "flux_int": 0.5,
    "flux_period": 1.0,
    "plot_transitions": {"r_f": 5.7}
  }
}
""".strip(),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "run_id": ["run-1"],
            "point": [None],
            "at": ["2026-10-10T00:00:00Z"],
            "source": ["manual"],
            "flux": [0.5],
            "flux_int": [0.5],
            "flux_period": [1.0],
            "freq_mhz": [350.0],
            "t1_us": [40.0],
        }
    ).to_csv(tmp_path / "samples.csv", index=False)

    ctx = load_t1_curve_context(
        result_dir=str(tmp_path), fluxs=np.array([0.5]), preview_rows=1
    )

    assert ctx.params == pytest.approx((3.4, 0.9, 0.6))
    assert "bare_rf (GHz)" not in set(ctx.params_table["parameter"])
    assert "g (GHz)" not in set(ctx.params_table["parameter"])
    assert len(ctx.samples_preview) == 1


@pytest.mark.parametrize("Temp", [None, 0.08])
def test_load_t1_purcell_params_reads_dispersive_bare_rf_and_g(tmp_path, Temp) -> None:
    (tmp_path / "params.json").write_text(
        """
{
  "fluxdep_fit": {
    "params": {"EJ": 3.4, "EC": 0.9, "EL": 0.6},
    "flux_half": 0.0,
    "flux_int": 0.5,
    "flux_period": 1.0,
    "plot_transitions": {"r_f": 5.7}
  },
  "dispersive": {
    "bare_rf": 5.793127423605109,
    "g": 0.07390631094081157
  }
}
""".strip(),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "run_id": ["run-1"],
            "point": [None],
            "at": ["2026-10-10T00:00:00Z"],
            "source": ["manual"],
            "flux": [0.5],
            "flux_int": [0.5],
            "flux_period": [1.0],
            "freq_mhz": [350.0],
            "t1_us": [40.0],
        }
    ).to_csv(tmp_path / "samples.csv", index=False)

    ctx = load_t1_curve_context(
        result_dir=str(tmp_path), fluxs=np.array([0.5]), preview_rows=1
    )
    purcell = load_t1_purcell_params(ctx, kappa_ghz=14.8e-3, Temp=Temp)

    assert purcell.kappa_ghz == pytest.approx(14.8e-3)
    assert purcell.bare_rf == pytest.approx(5.793127423605109)
    assert purcell.g == pytest.approx(0.07390631094081157)
    assert purcell.Temp == Temp


def test_calibrate_t1_flux_selects_finite_measurement_rows() -> None:
    samples = pd.DataFrame(
        {
            "flux": [-0.5, -0.49, -0.48],
            "freq_mhz": [350.0, 360.0, 370.0],
            "t1_us": [40.0, np.nan, 42.0],
        }
    )
    context = _synthetic_context(samples)

    cal = calibrate_t1_flux(context)

    assert len(cal.freq_rows) == 3
    assert len(cal.t1_df) == 2
    np.testing.assert_allclose(cal.context.fluxs, [-0.5, -0.49, -0.48])


def test_prepare_t1_curve_data_default_keeps_nan_error_points() -> None:
    samples = pd.DataFrame(
        {
            "flux": [-0.5, -0.49],
            "freq_mhz": [350.0, 360.0],
            "t1_us": [40.0, 41.0],
            "t1_err_us": [np.nan, 0.2],
        }
    )
    calibration = _synthetic_calibration(samples)

    data = prepare_t1_curve_data(
        calibration,
        analysis_flux_range=(0.49, 0.52),
    )
    weighted_only = prepare_t1_curve_data(
        calibration,
        analysis_flux_range=(0.49, 0.52),
        use_weighted_points_only=True,
    )

    assert len(data.fit.T1_ns) == 2
    assert np.isnan(data.fit.T1err_ns[0])
    assert len(weighted_only.fit.T1_ns) == 1
    assert weighted_only.fit.fluxs[0] == pytest.approx(0.510)


def test_make_t1_fit_init_can_select_partial_mechanisms() -> None:
    init = make_t1_fit_init(
        active_mechanisms=("capacitive",),
        Temp=0.06,
        Q_cap=7.0e5,
        x_qp=1.0e-6,
        Q_ind=3.0e7,
    )

    assert init.Q_cap == pytest.approx(7.0e5)
    assert init.x_qp is None
    assert init.Q_ind is None
    assert mechanisms_to_fixed_params(("capacitive", "Temp")) == ("Q_cap", "Temp")


@pytest.mark.parametrize("mechanism", ["capacitive", "quasiparticle", "inductive"])
def test_mechanism_options_match_wrapper_probe_results(
    mechanism: workflow.MechanismName,
) -> None:
    data = _synthetic_prepared_data()
    wrapper = {
        "capacitive": workflow.analyze_t1_capacitive_limit,
        "quasiparticle": workflow.analyze_t1_quasiparticle_limit,
        "inductive": workflow.analyze_t1_inductive_limit,
    }[mechanism]
    options = T1MechanismOptions(
        Temp=0.035,
        omega_range=(2.5, 4.5),
        Temp_bounds=(None, 0.12),
        fit_constant=False,
        statistic="max",
        parameter_init=0.002,
    )

    generic = analyze_t1_mechanism_limit(data, mechanism=mechanism, options=options)
    wrapped = wrapper(
        data,
        Temp=0.035,
        omega_range=(2.5, 4.5),
        Temp_bounds=(None, 0.12),
        fit_constant=False,
        statistic="max",
        parameter_init=0.002,
    )

    assert wrapped.mechanism == mechanism
    assert wrapped.temperature == 0.035
    assert wrapped.parameter_init == 0.002
    assert not wrapped.fit_constant
    np.testing.assert_array_equal(wrapped.fit_omegas, [3.0, 4.0])
    np.testing.assert_allclose(wrapped.q_values, generic.q_values)
    np.testing.assert_allclose(wrapped.q_errors, generic.q_errors)
    np.testing.assert_allclose(wrapped.fit_q_values, generic.fit_q_values)
    pd.testing.assert_frame_equal(wrapped.pointwise_table, generic.pointwise_table)
    pd.testing.assert_frame_equal(wrapped.summary_table, generic.summary_table)
    summary = wrapped.summary_table.set_index("metric")["value"]
    assert summary["Temp bounds"] == "default..120.00 mK"


@pytest.mark.parametrize("mechanism", ["capacitive", "quasiparticle", "inductive"])
def test_mechanism_options_use_statistic_and_reject_empty_window(
    mechanism: workflow.MechanismName,
) -> None:
    data = _synthetic_prepared_data()
    options = T1MechanismOptions(Temp=0.06, statistic="max", fit_constant=False)

    probe = analyze_t1_mechanism_limit(data, mechanism=mechanism, options=options)

    q_max = float(np.max(probe.fit_q_values))
    expected = 1.0 / q_max if mechanism == "quasiparticle" else q_max
    assert probe.parameter_init == pytest.approx(expected)
    np.testing.assert_array_equal(probe.fit_omegas, data.fit.omegas)
    with pytest.raises(ValueError, match=f"No finite positive {mechanism} Q values"):
        analyze_t1_mechanism_limit(
            data, mechanism=mechanism, options=replace(options, omega_range=(8.0, 9.0))
        )


@pytest.mark.parametrize("purcell_Temp", [None, 0.08])
@pytest.mark.parametrize("fit_temperature", [False, True])
def test_mechanism_probe_subtracts_resolved_purcell_temperature(
    purcell_Temp: float | None, fit_temperature: bool
) -> None:
    data = _synthetic_prepared_data()
    intrinsic_probe = analyze_t1_mechanism_limit(
        data, mechanism="capacitive", options=T1MechanismOptions(Temp=0.04)
    )
    intrinsic_T1 = 7e5 * np.array([1.01, 0.99, 1.01]) / intrinsic_probe.dipoles
    purcell = PurcellEffectParams(kappa_ghz=0.1, bare_rf=7.2, g=0.03, Temp=purcell_Temp)
    bath_T1 = calculate_purcell_t1_limit(
        data.calibration.context,
        data.fit.fluxs,
        purcell,
        Temp=0.04 if purcell_Temp is None else purcell_Temp,
    )
    observed = replace(data.fit, T1_ns=1.0 / (1.0 / intrinsic_T1 + 1.0 / bath_T1))
    measured = replace(data, fit=observed, sample=observed)
    options = T1MechanismOptions(
        Temp=0.05 if fit_temperature else 0.04,
        purcell=purcell,
        fit_temperature=fit_temperature,
        Temp_bounds=(0.02, 0.1),
    )

    probe = analyze_t1_mechanism_limit(
        measured, mechanism="capacitive", options=options
    )

    expected_bath = calculate_purcell_t1_limit(
        data.calibration.context,
        data.fit.fluxs,
        purcell,
        Temp=probe.temperature if purcell_Temp is None else purcell_Temp,
    )
    np.testing.assert_allclose(probe.pointwise_table["Purcell T1 (ns)"], expected_bath)
    if purcell_Temp is not None:
        corrected = replace(data, fit=replace(data.fit, T1_ns=intrinsic_T1))
        expected_probe = analyze_t1_mechanism_limit(
            corrected,
            mechanism="capacitive",
            options=replace(options, purcell=None),
        )
        assert probe.temperature == pytest.approx(expected_probe.temperature, abs=1e-6)
        np.testing.assert_allclose(probe.q_values, expected_probe.q_values, rtol=1e-5)


@pytest.mark.parametrize("purcell_Temp", [None, 0.08])
def test_mechanism_plot_uses_resolved_purcell_temperature(
    purcell_Temp: float | None,
) -> None:
    data = _synthetic_prepared_data()
    purcell = PurcellEffectParams(
        kappa_ghz=0.001, bare_rf=7.2, g=0.03, Temp=purcell_Temp
    )
    probe = analyze_t1_mechanism_limit(
        data,
        mechanism="capacitive",
        options=T1MechanismOptions(Temp=0.04, purcell=purcell),
    )

    fig, ax = plot_t1_mechanism_limit(probe, t_flux_count=3)

    try:
        line = next(line for line in ax.lines if line.get_label() == "Purcell")
        fluxs = np.asarray(line.get_xdata(), dtype=np.float64)
        expected = calculate_purcell_t1_limit(
            data.calibration.context,
            fluxs,
            purcell,
            Temp=0.04 if purcell_Temp is None else purcell_Temp,
        )
        np.testing.assert_allclose(
            np.asarray(line.get_ydata(), dtype=np.float64), expected
        )
    finally:
        plt.close(fig)


def test_subtract_relaxation_limit_uses_rate_domain() -> None:
    observed_T1s = np.array([40_000.0, 50_000.0], dtype=np.float64)
    observed_T1errs = np.array([500.0, np.nan], dtype=np.float64)
    limit_T1s = 2.0 * observed_T1s

    correction = subtract_relaxation_limit(observed_T1s, observed_T1errs, limit_T1s)

    np.testing.assert_allclose(correction.intrinsic_T1_ns, 2.0 * observed_T1s)
    np.testing.assert_allclose(
        correction.intrinsic_T1err_ns[0], 4.0 * observed_T1errs[0]
    )
    assert np.isnan(correction.intrinsic_T1err_ns[1])
    np.testing.assert_array_equal(correction.valid_mask, [True, True])


def test_t1_parameter_text_shows_only_active_noise_params_and_temp() -> None:
    result = T1FitResult(
        params=T1FitParams(Q_cap=7.0e5, Temp=0.06),
        stderr=T1FitParams(Q_cap=0.0, Temp=0.0),
        fixed=(),
        free=("Q_cap", "Temp"),
        model_T1s=np.array([1.0], dtype=np.float64),
        residuals=np.array([0.0], dtype=np.float64),
        cost=0.0,
        reduced_chi2=12.3,
        success=True,
        message="ok",
        optimizer_result=None,
    )

    text = workflow.t1_parameter_text(result)

    assert "Q_cap" in text
    assert "Temp" in text
    assert "reduced chi2" not in text
    assert "Purcell kappa" not in text


@pytest.mark.parametrize(
    "purcell_Temp, expected", [(None, "shared with qubit Temp"), (0.08, "0.08")]
)
def test_fit_curve_summary_reports_purcell_temperature(
    purcell_Temp: float | None, expected: str
) -> None:
    combined = fit_t1_curve(
        _synthetic_prepared_data(),
        init=T1FitParams(Q_cap=7e5, Temp=0.03),
        fixed=("Q_cap", "Temp"),
        purcell=PurcellEffectParams(
            kappa_ghz=0.001, bare_rf=7.2, g=0.03, Temp=purcell_Temp
        ),
    )

    summary = combined.summary_table.set_index("metric")["value"]
    assert summary["Purcell Temp (K)"] == expected


@pytest.mark.parametrize("purcell_Temp", [None, 0.08])
def test_fit_curve_uses_resolved_purcell_temperature(
    purcell_Temp: float | None,
) -> None:
    data = _synthetic_prepared_data()
    init = T1FitParams(Q_cap=7e5, Temp=0.03)
    purcell = PurcellEffectParams(
        kappa_ghz=0.001, bare_rf=7.2, g=0.03, Temp=purcell_Temp
    )
    intrinsic = fit_t1_curve(data, init=init, fixed=("Q_cap", "Temp"))
    combined = fit_t1_curve(data, init=init, fixed=("Q_cap", "Temp"), purcell=purcell)
    purcell_T1 = calculate_purcell_t1_limit(
        data.calibration.context,
        data.fit.fluxs,
        purcell,
        Temp=0.03 if purcell_Temp is None else purcell_Temp,
    )

    expected = 1.0 / (1.0 / intrinsic.fit_result.model_T1s + 1.0 / purcell_T1)
    np.testing.assert_allclose(combined.fit_result.model_T1s, expected, rtol=1e-10)


@pytest.mark.parametrize("purcell_Temp", [None, 0.08])
@pytest.mark.parametrize("qubit_Temp", [0.03, 0.06])
def test_channel_curves_use_resolved_purcell_temperature(
    purcell_Temp: float | None, qubit_Temp: float
) -> None:
    data = _synthetic_prepared_data()
    purcell = PurcellEffectParams(
        kappa_ghz=0.001, bare_rf=7.2, g=0.03, Temp=purcell_Temp
    )
    combined = fit_t1_curve(
        data,
        init=T1FitParams(Q_cap=7e5, Temp=qubit_Temp),
        fixed=("Q_cap", "Temp"),
        purcell=purcell,
    )

    analysis = build_t1_channel_curves(combined, t_flux_count=3)

    expected = calculate_purcell_t1_limit(
        data.calibration.context,
        analysis.curves.fluxs,
        purcell,
        Temp=qubit_Temp if purcell_Temp is None else purcell_Temp,
    )
    assert analysis.curves.purcell_T1_ns is not None
    np.testing.assert_allclose(analysis.curves.purcell_T1_ns, expected)


def test_build_t1_channel_curves_uses_uniform_flux_grid() -> None:
    data = _synthetic_prepared_data()
    fit_result = T1FitResult(
        params=T1FitParams(Q_cap=7.0e5, x_qp=1.0e-6, Temp=0.06),
        stderr=T1FitParams(Q_cap=0.0, x_qp=0.0, Temp=0.0),
        fixed=(),
        free=("Q_cap", "x_qp", "Temp"),
        model_T1s=data.fit.T1_ns,
        residuals=np.zeros_like(data.fit.T1_ns),
        cost=0.0,
        reduced_chi2=0.0,
        success=True,
        message="ok",
        optimizer_result=None,
    )
    combined = workflow.T1CombinedFit(
        data=data,
        init=fit_result.params,
        bounds=None,
        fixed=(),
        residual_mode="log",
        loss="linear",
        max_nfev=100,
        fit_result=fit_result,
        params_table=pd.DataFrame(),
        summary_table=pd.DataFrame(),
    )

    channel_analysis = build_t1_channel_curves(
        combined,
        t_flux_count=5,
        flux_range=(0.4, 0.6),
    )

    np.testing.assert_allclose(channel_analysis.curves.fluxs, np.linspace(0.4, 0.6, 5))
    assert set(channel_analysis.curves.component_T1s_ns) == {
        "capacitive",
        "quasiparticle",
    }
    assert np.all(channel_analysis.curves.display_T1_ns > 0)


def _synthetic_calibration(samples: pd.DataFrame) -> T1FluxCalibration:
    context = _synthetic_context(samples)
    return T1FluxCalibration(
        context=context,
        t1_df=samples,
        freq_rows=samples,
        summary_table=pd.DataFrame(),
    )


def _synthetic_context(samples: pd.DataFrame) -> T1CurveContext:
    return T1CurveContext(
        result_dir="/tmp/result",
        image_dir="/tmp/result/t1_curve",
        samples_filename="samples.csv",
        params=(3.4, 0.9, 0.6),
        samples_df=samples,
        fluxs=samples["flux"].to_numpy(dtype=np.float64),
        params_table=pd.DataFrame(),
        samples_preview=samples,
        available_columns=tuple(samples.columns),
    )


def _synthetic_prepared_data() -> T1PreparedData:
    fluxs = np.array([0.49, 0.50, 0.51], dtype=np.float64)
    curve = T1CurveData(
        fluxs=fluxs,
        raw_fluxs=fluxs,
        integer_shifts=np.zeros(3, dtype=np.float64),
        f01_mhz=np.array([350.0, 360.0, 370.0], dtype=np.float64),
        f01_measured=np.ones(3, dtype=bool),
        T1_ns=np.array([40_000.0, 42_000.0, 41_000.0], dtype=np.float64),
        T1err_ns=np.array([500.0, np.nan, 600.0], dtype=np.float64),
        omegas=np.array([2.0, 3.0, 4.0], dtype=np.float64),
    )
    samples = pd.DataFrame(
        {
            "flux": [-0.5, -0.49, -0.48],
            "freq_mhz": curve.f01_mhz,
            "t1_us": 1e-3 * curve.T1_ns,
            "t1_err_us": 1e-3 * curve.T1err_ns,
        }
    )
    calibration = _synthetic_calibration(samples)
    return T1PreparedData(
        calibration=calibration,
        analysis_flux_range=(0.49, 0.51),
        max_rel_t1_err=0.25,
        use_weighted_points_only=False,
        sample=curve,
        fit=curve,
        kept_rows=len(curve.fluxs),
        source_rows=len(curve.fluxs),
        summary_table=pd.DataFrame(),
    )


def test_t1_rows_with_null_freq_reach_analysis_with_model_f01() -> None:
    samples = pd.DataFrame(
        {
            "flux": [0.5, -0.5, -0.49],
            "freq_mhz": [np.nan, np.nan, np.nan],
            "t1_us": [40.0, 41.0, 42.0],
            "t1_err_us": [0.5, 0.5, 0.5],
        }
    )
    calibration = _synthetic_calibration(samples)

    data = prepare_t1_curve_data(calibration, analysis_flux_range=(0.49, 0.53))

    assert len(data.fit.T1_ns) == 3
    assert not np.any(data.sample.f01_measured)
    assert np.all(np.isfinite(data.sample.f01_mhz))
    np.testing.assert_allclose(
        np.asarray(data.sample.fluxs), [0.5, 0.5, 0.51], atol=1e-9
    )


def test_t1_integer_equivalent_branches_align_to_same_flux() -> None:
    samples = pd.DataFrame(
        {
            "flux": [0.5, -0.5],
            "freq_mhz": [350.0, 351.0],
            "t1_us": [40.0, 42.0],
            "t1_err_us": [0.5, 0.5],
        }
    )
    calibration = _synthetic_calibration(samples)

    data = prepare_t1_curve_data(
        calibration,
        analysis_flux_range=(0.49, 0.53),
    )

    # Author coordinates on different integer branches align to the same window.
    np.testing.assert_allclose(data.sample.fluxs, [0.5, 0.5])
    np.testing.assert_allclose(data.sample.raw_fluxs, [0.5, -0.5])
    np.testing.assert_allclose(data.sample.integer_shifts, [0.0, 1.0])

    # Prepared summary reports the branch-shift counts.
    shifts_row = data.summary_table.loc[
        data.summary_table["metric"] == "integer shifts"
    ]
    assert shifts_row["value"].iloc[0] == "k=0:1 k=1:1"


def test_t1_invalid_analysis_window_fails_fast() -> None:
    samples = pd.DataFrame(
        {
            "run_id": ["run-1"],
            "point": [None],
            "at": ["2026-10-10T00:00:00Z"],
            "source": ["manual"],
            "flux": [-0.5],
            "freq_mhz": [np.nan],
            "t1_us": [40.0],
        }
    )
    calibration = _synthetic_calibration(samples)

    with pytest.raises(ValueError, match="one flux period"):
        prepare_t1_curve_data(calibration, analysis_flux_range=(0.0, 1.5))
    with pytest.raises(ValueError, match="finite"):
        prepare_t1_curve_data(calibration, analysis_flux_range=(np.nan, 0.6))
