from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import zcu_tools.notebook.analysis.t1_curve.workflow as workflow
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from zcu_tools.notebook.analysis.fit_tools import F01FluxCorrectionResult
from zcu_tools.notebook.analysis.t1_curve import (
    PurcellEffectParams,
    T1CurveContext,
    T1CurveData,
    T1FitParams,
    T1FitResult,
    T1FluxCalibration,
    T1PreparedData,
    build_t1_channel_curves,
    calibrate_t1_flux,
    fit_t1_curve,
    load_t1_curve_context,
    make_t1_fit_init,
    mechanisms_to_fixed_params,
    prepare_t1_curve_data,
    subtract_relaxation_limit,
)


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
            "calibrated mA": [0.0],
            "Freq (MHz)": [350.0],
            "T1 (us)": [40.0],
        }
    ).to_csv(tmp_path / "samples.csv", index=False)

    ctx = load_t1_curve_context(result_dir=str(tmp_path), preview_rows=1)

    assert ctx.params == pytest.approx((3.4, 0.9, 0.6))
    assert ctx.flux_period == pytest.approx(1.0)
    assert "bare_rf (GHz)" not in set(ctx.params_table["parameter"])
    assert "g (GHz)" not in set(ctx.params_table["parameter"])
    assert len(ctx.samples_preview) == 1


def test_calibrate_t1_flux_filters_finite_rows_and_records_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = pd.DataFrame(
        {
            "calibrated mA": [0.0, 0.01, np.nan],
            "Freq (MHz)": [350.0, 360.0, 370.0],
            "T1 (us)": [40.0, np.nan, 42.0],
        }
    )
    context = _synthetic_context(samples)

    def _choose_scale(
        raw_values: NDArray[np.float64],
        measured_freqs_mhz: NDArray[np.float64],
        **_kwargs: object,
    ) -> tuple[float, pd.DataFrame]:
        np.testing.assert_allclose(raw_values, [0.0, 0.01])
        np.testing.assert_allclose(measured_freqs_mhz, [350.0, 360.0])
        return 1000.0, pd.DataFrame({"scale": [1000.0]})

    monkeypatch.setattr(workflow, "choose_current_scale_from_f01", _choose_scale)

    cal = calibrate_t1_flux(context)

    assert cal.current_scale == pytest.approx(1000.0)
    assert len(cal.freq_rows) == 2
    assert len(cal.t1_df) == 1


def test_prepare_t1_curve_data_default_keeps_nan_error_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fluxs = np.array([0.500, 0.510], dtype=np.float64)
    samples = pd.DataFrame(
        {
            "calibrated mA": [0.0, 0.01],
            "Freq (MHz)": [350.0, 360.0],
            "T1 (us)": [40.0, 41.0],
            "T1err (us)": [np.nan, 0.2],
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

    data = prepare_t1_curve_data(calibration, analysis_flux_range=(0.49, 0.52))
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


def test_t1_mechanism_probe_uses_pointwise_q_for_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _synthetic_prepared_data()

    def _fake_arrays(
        _data: T1PreparedData,
        mechanism: workflow.MechanismName,
        _Temp: float,
        *,
        T1_ns: NDArray[np.float64] | None = None,
        T1err_ns: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        assert mechanism == "capacitive"
        assert T1_ns is not None
        assert T1err_ns is not None
        return (
            np.array([1.0e5, 2.0e5, 4.0e5], dtype=np.float64),
            np.array([1.0e4, 2.0e4, 4.0e4], dtype=np.float64),
            np.array([2.0, 2.0, 2.0], dtype=np.float64),
        )

    monkeypatch.setattr(workflow, "_calculate_mechanism_arrays", _fake_arrays)

    probe = workflow.analyze_t1_capacitive_limit(
        data,
        Temp=0.06,
        omega_range=(None, None),
        fit_constant=True,
        statistic="median",
    )

    assert probe.parameter_name == "Q_cap"
    assert probe.parameter_init == pytest.approx(2.0e5)
    assert probe.pointwise_table["Q"].tolist() == pytest.approx([1.0e5, 2.0e5, 4.0e5])


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


def test_calculate_purcell_t1_limit_reuses_lru_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow.clear_t1_purcell_cache()
    samples = pd.DataFrame(
        {
            "calibrated mA": [0.0],
            "Freq (MHz)": [350.0],
            "T1 (us)": [40.0],
        }
    )
    context = _synthetic_context(samples)
    fluxs = np.array([0.49, 0.50, 0.51], dtype=np.float64)
    call_count = 0

    def _fake_purcell_limit(
        fluxs: NDArray[np.float64],
        bare_rf: float,
        kappa: float,
        g: float,
        Temp: float,
        params: tuple[float, float, float],  # noqa: ARG001
        progress: bool,
    ) -> NDArray[np.float64]:
        nonlocal call_count
        call_count += 1
        assert bare_rf == pytest.approx(5.8)
        assert kappa == pytest.approx(14.8e-3)
        assert g == pytest.approx(0.07)
        assert progress is False
        return np.full_like(fluxs, 1000.0 + Temp)

    monkeypatch.setattr(workflow, "calculate_purcell_t1_vs_flux", _fake_purcell_limit)
    purcell = _synthetic_purcell()

    try:
        first = workflow.calculate_purcell_t1_limit(context, fluxs, purcell, Temp=0.06)
        second = workflow.calculate_purcell_t1_limit(
            context, fluxs.copy(), purcell, Temp=0.06
        )
        third = workflow.calculate_purcell_t1_limit(context, fluxs, purcell, Temp=0.07)
    finally:
        workflow.clear_t1_purcell_cache()

    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(third, 1000.07)
    assert call_count == 2


def test_t1_mechanism_probe_subtracts_purcell_before_q(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _synthetic_prepared_data()
    captured: dict[str, NDArray[np.float64]] = {}

    def _fake_purcell(
        context: T1CurveContext,  # noqa: ARG001
        fluxs: NDArray[np.float64],
        purcell: PurcellEffectParams,  # noqa: ARG001
        *,
        Temp: float,  # noqa: ARG001
    ) -> NDArray[np.float64]:
        return 2.0 * data.fit.T1_ns[: len(fluxs)]

    def _fake_arrays(
        _data: T1PreparedData,
        mechanism: workflow.MechanismName,
        _Temp: float,
        *,
        T1_ns: NDArray[np.float64] | None = None,
        T1err_ns: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        assert mechanism == "capacitive"
        assert T1_ns is not None
        assert T1err_ns is not None
        captured["T1_ns"] = T1_ns
        captured["T1err_ns"] = T1err_ns
        return (
            np.array([1.0e5, 2.0e5, 4.0e5], dtype=np.float64),
            np.array([1.0e4, np.nan, 4.0e4], dtype=np.float64),
            np.array([2.0, 2.0, 2.0], dtype=np.float64),
        )

    monkeypatch.setattr(workflow, "calculate_purcell_t1_limit", _fake_purcell)
    monkeypatch.setattr(workflow, "_calculate_mechanism_arrays", _fake_arrays)

    probe = workflow.analyze_t1_capacitive_limit(
        data,
        Temp=0.06,
        purcell=_synthetic_purcell(),
    )

    np.testing.assert_allclose(captured["T1_ns"], 2.0 * data.fit.T1_ns)
    np.testing.assert_allclose(
        captured["T1err_ns"][[0, 2]], 4.0 * data.fit.T1err_ns[[0, 2]]
    )
    assert np.isnan(captured["T1err_ns"][1])
    assert probe.purcell is not None
    assert probe.purcell_correction is not None
    assert "Purcell T1 (ns)" in probe.pointwise_table


def test_t1_mechanism_dipole_plot_uses_t1_after_purcell_subtraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _synthetic_prepared_data()
    captured: dict[str, object] = {}

    def _fake_purcell(
        context: T1CurveContext,  # noqa: ARG001
        fluxs: NDArray[np.float64],
        purcell: PurcellEffectParams,  # noqa: ARG001
        *,
        Temp: float,  # noqa: ARG001
    ) -> NDArray[np.float64]:
        return 2.0 * data.fit.T1_ns[: len(fluxs)]

    def _fake_arrays(
        _data: T1PreparedData,
        mechanism: workflow.MechanismName,
        _Temp: float,
        *,
        T1_ns: NDArray[np.float64] | None = None,
        T1err_ns: NDArray[np.float64] | None = None,  # noqa: ARG001
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        assert mechanism == "capacitive"
        assert T1_ns is not None
        return (
            np.array([1.0e5, 2.0e5, 4.0e5], dtype=np.float64),
            np.array([1.0e4, np.nan, 4.0e4], dtype=np.float64),
            np.array([2.0, 3.0, 4.0], dtype=np.float64),
        )

    def _fake_plot_t1_vs_elements(
        dipoles: NDArray[np.float64],
        T1s: NDArray[np.float64],
        T1errs: NDArray[np.float64] | None = None,
        dipole_name: str = "d_{01}",  # noqa: ARG001
        Q_name: str = r"$Q_{cap}$",
        product2val: Callable[[float], float] = lambda x: x,  # noqa: ARG005
    ) -> tuple[Figure, Axes]:
        captured["dipoles"] = dipoles
        captured["T1s"] = T1s
        captured["T1errs"] = T1errs
        captured["Q_name"] = Q_name
        return plt.subplots()

    monkeypatch.setattr(workflow, "calculate_purcell_t1_limit", _fake_purcell)
    monkeypatch.setattr(workflow, "_calculate_mechanism_arrays", _fake_arrays)
    monkeypatch.setattr(workflow, "plot_t1_vs_elements", _fake_plot_t1_vs_elements)

    probe = workflow.analyze_t1_capacitive_limit(
        data,
        Temp=0.06,
        purcell=_synthetic_purcell(),
    )
    fig, _ = workflow.plot_t1_mechanism_dipole(probe)
    plt.close(fig)

    np.testing.assert_allclose(
        cast(NDArray[np.float64], captured["T1s"]),
        2.0 * data.fit.T1_ns,
    )
    np.testing.assert_allclose(
        cast(NDArray[np.float64], captured["dipoles"]), [2, 3, 4]
    )
    assert captured["Q_name"] == r"$Q_{cap}$"


def test_t1_mechanism_limit_combines_plot_level_purcell_into_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _synthetic_prepared_data()
    captured: dict[str, object] = {}
    pure_t1_limits = [20.0, 10.0, 5.0]

    def _fake_arrays(
        _data: T1PreparedData,
        mechanism: workflow.MechanismName,
        _Temp: float,
        *,
        T1_ns: NDArray[np.float64] | None = None,
        T1err_ns: NDArray[np.float64] | None = None,  # noqa: ARG001
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        assert mechanism == "capacitive"
        assert T1_ns is not None
        return (
            np.array([1.0e5, 2.0e5, 4.0e5], dtype=np.float64),
            np.array([1.0e4, 2.0e4, 4.0e4], dtype=np.float64),
            np.array([2.0, 2.0, 2.0], dtype=np.float64),
        )

    def _fake_eff(
        _params: tuple[float, float, float],
        fluxs: NDArray[np.float64],
        _noise_channels: list[tuple[str, dict[str, float]]],
        _Temp: float,
        **_kwargs: object,
    ) -> NDArray[np.float64]:
        return np.full_like(fluxs, pure_t1_limits.pop(0))

    def _fake_purcell(
        context: T1CurveContext,  # noqa: ARG001
        fluxs: NDArray[np.float64],
        purcell: PurcellEffectParams,  # noqa: ARG001
        *,
        Temp: float,  # noqa: ARG001
    ) -> NDArray[np.float64]:
        return np.full_like(fluxs, 40.0)

    def _fake_plot_eff(
        _s_dev_values: NDArray[np.float64],
        _s_T1s: NDArray[np.float64],
        _s_T1errs: NDArray[np.float64],
        t1_effs: NDArray[np.float64],
        _flux_half: float,
        _flux_period: float,
        _t_fluxs: NDArray[np.float64],
        *,
        label: str = r"$t_1^{eff}$",
        title: str | None = None,  # noqa: ARG001
        xlabel: str = "Current (mA)",  # noqa: ARG001
        component_t1s: dict[str, NDArray[np.float64]] | None = None,
        parameter_text: str | None = None,
    ) -> tuple[Figure, Axes]:
        captured["t1_effs"] = t1_effs
        captured["component_t1s"] = component_t1s
        captured["label"] = label
        captured["parameter_text"] = parameter_text
        return plt.subplots()

    monkeypatch.setattr(workflow, "_calculate_mechanism_arrays", _fake_arrays)
    monkeypatch.setattr(workflow, "calculate_eff_t1_vs_flux_fast", _fake_eff)
    monkeypatch.setattr(workflow, "calculate_purcell_t1_limit", _fake_purcell)
    monkeypatch.setattr(workflow, "plot_eff_t1_with_sample", _fake_plot_eff)
    probe = workflow.analyze_t1_capacitive_limit(data, Temp=0.06)

    fig, _ = workflow.plot_t1_mechanism_limit(
        probe,
        t_flux_count=5,
        flux_range=(0.4, 0.6),
        purcell=_synthetic_purcell(),
    )
    plt.close(fig)

    component_t1s = cast(dict[str, NDArray[np.float64]], captured["component_t1s"])
    assert captured["label"] == "capacitive + Purcell"
    assert "Purcell" in component_t1s
    np.testing.assert_allclose(component_t1s["Purcell"], 40.0)
    np.testing.assert_allclose(component_t1s["capacitive lower"], 40.0 / 3.0)
    np.testing.assert_allclose(component_t1s["capacitive upper"], 40.0 / 9.0)
    np.testing.assert_allclose(
        cast(NDArray[np.float64], captured["t1_effs"]),
        8.0,
    )
    assert "Purcell kappa" in str(captured["parameter_text"])


def test_fit_t1_curve_wrapper_passes_shared_policies_without_writeback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _synthetic_prepared_data()
    captured: dict[str, object] = {}

    def _fake_fit(
        fluxs: NDArray[np.float64],
        T1s: NDArray[np.float64],
        params: tuple[float, float, float],
        **kwargs: object,
    ) -> T1FitResult:
        captured["fluxs"] = fluxs
        captured["T1s"] = T1s
        captured["params"] = params
        captured.update(kwargs)
        return T1FitResult(
            params=kwargs["init"],  # type: ignore[arg-type]
            stderr=T1FitParams(Q_cap=0.0, Temp=0.0),
            fixed=(),
            free=("Q_cap", "Temp"),
            model_T1s=T1s,
            residuals=np.zeros_like(T1s),
            cost=0.0,
            reduced_chi2=0.0,
            success=True,
            message="ok",
            optimizer_result=None,
        )

    class _UnexpectedQubitParams:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("fit_t1_curve should not write params.json")

    monkeypatch.setattr(workflow, "fit_t1_noise_params", _fake_fit)
    monkeypatch.setattr(workflow, "QubitParams", _UnexpectedQubitParams)
    error_policy = workflow.MeasurementErrorPolicy(nan_policy="bin_median")
    flux_weighting = workflow.FluxResidualWeighting(
        mode="equal_flux_bin",
        bin_width=0.01,
    )

    combined = fit_t1_curve(
        data,
        init=T1FitParams(Q_cap=7.0e5, Temp=0.06),
        T1_error_policy=error_policy,
        flux_weighting=flux_weighting,
    )

    np.testing.assert_allclose(
        cast(NDArray[np.float64], captured["fluxs"]), data.fit.fluxs
    )
    np.testing.assert_allclose(
        cast(NDArray[np.float64], captured["T1s"]), data.fit.T1_ns
    )
    assert captured["T1_error_policy"] is error_policy
    assert captured["flux_weighting"] is flux_weighting
    assert combined.fit_result.success


def test_fit_t1_curve_passes_purcell_rate_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _synthetic_prepared_data()
    captured: dict[str, object] = {}

    def _fake_purcell(
        context: T1CurveContext,  # noqa: ARG001
        fluxs: NDArray[np.float64],
        purcell: PurcellEffectParams,  # noqa: ARG001
        *,
        Temp: float,
    ) -> NDArray[np.float64]:
        captured["purcell_temp"] = Temp
        return np.array([100.0, 200.0, 400.0], dtype=np.float64)[: len(fluxs)]

    def _fake_fit(
        fluxs: NDArray[np.float64],
        T1s: NDArray[np.float64],
        params: tuple[float, float, float],
        **kwargs: object,
    ) -> T1FitResult:
        captured["fluxs"] = fluxs
        captured["T1s"] = T1s
        captured["params"] = params
        captured.update(kwargs)
        return T1FitResult(
            params=kwargs["init"],  # type: ignore[arg-type]
            stderr=T1FitParams(Q_cap=0.0, Temp=0.0),
            fixed=(),
            free=("Q_cap", "Temp"),
            model_T1s=T1s,
            residuals=np.zeros_like(T1s),
            cost=0.0,
            reduced_chi2=0.0,
            success=True,
            message="ok",
            optimizer_result=None,
        )

    monkeypatch.setattr(workflow, "calculate_purcell_t1_limit", _fake_purcell)
    monkeypatch.setattr(workflow, "fit_t1_noise_params", _fake_fit)

    combined = fit_t1_curve(
        data,
        init=T1FitParams(Q_cap=7.0e5, Temp=0.06),
        purcell=_synthetic_purcell(),
    )

    rate_fn = cast(
        Callable[[T1FitParams], NDArray[np.float64]],
        captured["extra_relaxation_rate_fn"],
    )
    assert callable(rate_fn)
    rates = rate_fn(T1FitParams(Q_cap=7.0e5, Temp=0.07))
    np.testing.assert_allclose(rates, [0.01, 0.005, 0.0025])
    assert captured["purcell_temp"] == pytest.approx(0.07)
    assert combined.purcell is not None


def test_plot_t1_flux_calibration_shows_before_after_positions() -> None:
    data = _synthetic_prepared_data()
    sample = replace(
        data.sample,
        raw_fluxs=np.array([0.488, 0.500, 0.514], dtype=np.float64),
        fluxs=np.array([0.490, 0.500, 0.510], dtype=np.float64),
        f01_correction_accepted=np.array([True, False, True]),
        flux_corrections=np.array([0.002, 0.000, -0.004], dtype=np.float64),
    )
    data = replace(data, sample=sample)

    fig, ax = workflow.plot_t1_flux_calibration(data)

    labels = {collection.get_label() for collection in ax.collections}
    assert "raw flux" in labels
    assert "f01 corrected flux" in labels
    assert "kept raw flux" in labels
    assert len(ax.lines) == len(sample.fluxs)
    assert ax.get_ylabel() == "f01 frequency (MHz)"
    for line, f01_mhz in zip(ax.lines, sample.f01_mhz, strict=True):
        np.testing.assert_allclose(
            np.asarray(line.get_ydata(), dtype=np.float64),
            [f01_mhz, f01_mhz],
        )
    plt.close(fig)


def test_build_t1_channel_curves_uses_uniform_flux_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    def _fake_eff(
        _params: tuple[float, float, float],
        fluxs: NDArray[np.float64],
        noise_channels: list[tuple[str, dict[str, float]]],
        _Temp: float,
        **_kwargs: object,
    ) -> NDArray[np.float64]:
        if len(noise_channels) == 1:
            channel_name = noise_channels[0][0]
            return np.full_like(
                fluxs, 10.0 if channel_name == "t1_capacitive" else 20.0
            )
        return np.full_like(fluxs, 7.5)

    monkeypatch.setattr(workflow, "calculate_eff_t1_vs_flux_fast", _fake_eff)

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
    np.testing.assert_allclose(channel_analysis.curves.display_T1_ns, 7.5)


def test_build_t1_channel_curves_adds_purcell_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _synthetic_prepared_data()
    fit_result = T1FitResult(
        params=T1FitParams(Q_cap=7.0e5, Temp=0.06),
        stderr=T1FitParams(Q_cap=0.0, Temp=0.0),
        fixed=(),
        free=("Q_cap", "Temp"),
        model_T1s=data.fit.T1_ns,
        residuals=np.zeros_like(data.fit.T1_ns),
        cost=0.0,
        reduced_chi2=0.0,
        success=True,
        message="ok",
        optimizer_result=None,
    )
    purcell = _synthetic_purcell()
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
        purcell=purcell,
    )

    def _fake_eff(
        _params: tuple[float, float, float],
        fluxs: NDArray[np.float64],
        _noise_channels: list[tuple[str, dict[str, float]]],
        _Temp: float,
        **_kwargs: object,
    ) -> NDArray[np.float64]:
        return np.full_like(fluxs, 7.5)

    def _fake_purcell(
        context: T1CurveContext,  # noqa: ARG001
        fluxs: NDArray[np.float64],
        purcell: PurcellEffectParams,  # noqa: ARG001
        *,
        Temp: float,  # noqa: ARG001
    ) -> NDArray[np.float64]:
        return np.full_like(fluxs, 30.0)

    monkeypatch.setattr(workflow, "calculate_eff_t1_vs_flux_fast", _fake_eff)
    monkeypatch.setattr(workflow, "calculate_purcell_t1_limit", _fake_purcell)

    channel_analysis = build_t1_channel_curves(
        combined,
        t_flux_count=5,
        flux_range=(0.4, 0.6),
    )

    assert "Purcell" in channel_analysis.curves.component_T1s_ns
    assert channel_analysis.curves.purcell_T1_ns is not None
    np.testing.assert_allclose(channel_analysis.curves.purcell_T1_ns, 30.0)
    np.testing.assert_allclose(channel_analysis.curves.display_T1_ns, 6.0)


def _synthetic_calibration(samples: pd.DataFrame) -> T1FluxCalibration:
    context = _synthetic_context(samples)
    return T1FluxCalibration(
        context=context,
        current_scale=1.0,
        scale_report=pd.DataFrame(),
        t1_df=samples,
        freq_rows=samples,
        summary_table=pd.DataFrame(),
    )


def _synthetic_context(samples: pd.DataFrame) -> T1CurveContext:
    context = T1CurveContext(
        result_dir="/tmp/result",
        image_dir="/tmp/result/t1_curve",
        samples_filename="samples.csv",
        params=(3.4, 0.9, 0.6),
        flux_half=0.0,
        flux_int=0.5,
        flux_period=1.0,
        samples_df=samples,
        params_table=pd.DataFrame(),
        samples_preview=samples,
        available_columns=tuple(samples.columns),
    )
    return context


def _synthetic_purcell() -> PurcellEffectParams:
    return PurcellEffectParams(kappa_ghz=14.8e-3, bare_rf=5.8, g=0.07)


def _synthetic_prepared_data() -> T1PreparedData:
    fluxs = np.array([0.49, 0.50, 0.51], dtype=np.float64)
    curve = T1CurveData(
        current_raw=fluxs,
        values=fluxs,
        raw_fluxs=fluxs,
        fluxs=fluxs,
        f01_mhz=np.array([350.0, 360.0, 370.0], dtype=np.float64),
        T1_ns=np.array([40_000.0, 42_000.0, 41_000.0], dtype=np.float64),
        T1err_ns=np.array([500.0, np.nan, 600.0], dtype=np.float64),
        omegas=np.array([2.0, 3.0, 4.0], dtype=np.float64),
        f01_correction_accepted=np.ones(3, dtype=bool),
        flux_corrections=np.zeros(3, dtype=np.float64),
    )
    samples = pd.DataFrame(
        {
            "calibrated mA": curve.current_raw,
            "Freq (MHz)": curve.f01_mhz,
            "T1 (us)": 1e-3 * curve.T1_ns,
            "T1err (us)": 1e-3 * curve.T1err_ns,
        }
    )
    calibration = T1FluxCalibration(
        context=_synthetic_context(samples),
        current_scale=1.0,
        scale_report=pd.DataFrame(),
        t1_df=samples,
        freq_rows=samples,
        summary_table=pd.DataFrame(),
    )
    return T1PreparedData(
        calibration=calibration,
        analysis_flux_range=(0.49, 0.51),
        max_abs_flux_correction=0.03,
        max_rel_t1_err=0.25,
        use_weighted_points_only=False,
        sample=curve,
        fit=curve,
        kept_rows=len(curve.fluxs),
        source_rows=len(curve.fluxs),
        summary_table=pd.DataFrame(),
    )
