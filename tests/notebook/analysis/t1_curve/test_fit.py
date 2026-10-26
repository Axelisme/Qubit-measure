from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import zcu_tools.notebook.analysis.t1_curve.fit as fit_mod
import zcu_tools.notebook.analysis.t1_curve.t1_curve_fit as public_fit_mod
from numpy.typing import NDArray
from zcu_tools.notebook.analysis.fit_tools import find_weak_parameters
from zcu_tools.notebook.analysis.t1_curve import (
    FluxResidualWeighting,
    MeasurementErrorPolicy,
    T1FitParams,
    fit_t1_noise_params,
)
from zcu_tools.simulate.fluxonium import calculate_eff_t1_vs_flux_fast

_PARAMS = (3.469, 0.952, 0.582)


def test_t1_curve_fit_module_exports_public_api() -> None:
    assert public_fit_mod.T1FitParams is T1FitParams
    assert public_fit_mod.fit_t1_noise_params is fit_t1_noise_params


def _sample_fluxs() -> NDArray[np.float64]:
    return np.linspace(0.08, 0.92, 18, dtype=np.float64)


def _identifiability_samples(
    init: T1FitParams, flux_range: tuple[float, float] = (0.08, 0.49)
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    fluxs = np.linspace(*flux_range, 8, dtype=np.float64)
    T1s = calculate_eff_t1_vs_flux_fast(
        _PARAMS, fluxs, _noise_channels(init), init.Temp
    )
    return fluxs, T1s


def test_joint_fit_reports_log_covariance_in_free_order() -> None:
    init = T1FitParams(Q_cap=8e5, x_qp=1.5e-6, Q_ind=3e7, Temp=0.06)
    fluxs, T1s = _identifiability_samples(init)
    result = fit_t1_noise_params(
        fluxs,
        T1s * (1 + 0.005 * np.sin(np.arange(T1s.size))),
        _PARAMS,
        init=init,
        max_nfev=5,
    )

    info = result.correlation
    assert info is not None
    assert info.parameter_names == result.free == ("Q_cap", "x_qp", "Q_ind", "Temp")
    assert info.covariance.shape == info.correlation.shape == (4, 4)
    np.testing.assert_allclose(info.covariance, info.covariance.T)
    np.testing.assert_allclose(np.diag(info.correlation), 1)
    np.testing.assert_allclose(
        np.sqrt(np.diag(info.covariance))
        * np.asarray(
            [
                result.params.Q_cap,
                result.params.x_qp,
                result.params.Q_ind,
                result.params.Temp,
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                result.stderr.Q_cap,
                result.stderr.x_qp,
                result.stderr.Q_ind,
                result.stderr.Temp,
            ],
            dtype=np.float64,
        ),
    )
    assert info.reduced_chi2 == pytest.approx(result.reduced_chi2)
    assert info.rank == 4
    assert np.isfinite(info.condition_number)


def test_negligible_inductive_channel_is_flagged_by_log_uncertainty() -> None:
    init = T1FitParams(Q_cap=8e5, Q_ind=1e10, Temp=0.06)
    fluxs, T1s = _identifiability_samples(init)
    cap_only = calculate_eff_t1_vs_flux_fast(
        _PARAMS, fluxs, [("t1_capacitive", {"Q_cap": 8e5})], init.Temp
    )
    # The inductive contribution stays below the synthetic measurement scatter.
    np.testing.assert_allclose(T1s, cap_only, rtol=0.003)
    result = fit_t1_noise_params(
        fluxs,
        T1s * (1 + 0.02 * np.sin(np.arange(T1s.size))),
        _PARAMS,
        init=init,
        fixed=("Q_cap", "Temp"),
        bounds={"Q_ind": (1e9, 1e11)},
    )

    info = result.correlation
    assert info is not None
    assert info.parameter_names == ("Q_ind",)
    log_std = np.sqrt(info.covariance[0, 0])
    assert np.isfinite(log_std) and log_std > 1
    weak = find_weak_parameters(info)
    assert [entry.parameter for entry in weak] == ["Q_ind"]
    assert any("sigma(ln Q_ind)" in reason for reason in weak[0].reasons)


def test_well_constrained_capacitive_parameter_is_not_weak() -> None:
    init = T1FitParams(Q_cap=8e5, Temp=0.06)
    fluxs, T1s = _identifiability_samples(init)
    result = fit_t1_noise_params(
        fluxs,
        T1s * (1 + 0.001 * np.sin(np.arange(T1s.size))),
        _PARAMS,
        init=init,
        fixed=("Temp",),
    )

    assert result.correlation is not None
    # Fixed Temp and inactive Q_ind/x_qp have no rows or columns.
    assert result.correlation.parameter_names == ("Q_cap",)
    assert result.correlation.covariance.shape == (1, 1)
    assert find_weak_parameters(result.correlation) == ()


def test_nearly_degenerate_channels_report_strong_correlation() -> None:
    init = T1FitParams(Q_cap=8e5, x_qp=1.5e-6, Temp=0.06)
    fluxs, T1s = _identifiability_samples(init, flux_range=(0.2, 0.21))
    result = fit_t1_noise_params(fluxs, T1s, _PARAMS, init=init, fixed=("Temp",))

    assert result.correlation is not None
    assert abs(result.correlation.correlation[0, 1]) > 0.95
    assert [entry.parameter for entry in find_weak_parameters(result.correlation)] == [
        "Q_cap",
        "x_qp",
    ]


def test_fit_t1_noise_params_keeps_fixed_values() -> None:
    true_params = T1FitParams(Q_cap=8e5, x_qp=1.5e-6, Q_ind=3e7, Temp=0.06)
    fluxs = _sample_fluxs()
    T1s = calculate_eff_t1_vs_flux_fast(
        _PARAMS, fluxs, _noise_channels(true_params), true_params.Temp
    )

    result = fit_t1_noise_params(
        fluxs,
        T1s,
        _PARAMS,
        init=T1FitParams(Q_cap=5e5, x_qp=2.0e-6, Q_ind=3e7, Temp=0.06),
        fixed=("Q_ind", "Temp"),
    )

    assert result.fixed == ("Q_ind", "Temp")
    assert result.free == ("Q_cap", "x_qp")
    assert result.params.Q_ind == true_params.Q_ind
    assert result.params.Temp == true_params.Temp
    assert result.stderr.Q_ind == 0.0
    assert result.stderr.Temp == 0.0
    assert result.correlation is not None
    assert result.correlation.parameter_names == ("Q_cap", "x_qp")


def test_fit_t1_noise_params_all_fixed_skips_optimizer() -> None:
    init = T1FitParams(Q_cap=8e5, x_qp=1.5e-6, Q_ind=3e7, Temp=0.06)
    fluxs = _sample_fluxs()
    T1s = calculate_eff_t1_vs_flux_fast(
        _PARAMS, fluxs, _noise_channels(init), init.Temp
    )

    result = fit_t1_noise_params(
        fluxs,
        T1s,
        _PARAMS,
        init=init,
        fixed=("Q_cap", "x_qp", "Q_ind", "Temp"),
    )

    assert result.correlation is None
    assert result.optimizer_result is None
    assert result.success
    assert result.message == "all parameters fixed"
    assert result.cost == pytest.approx(0.0, abs=1e-24)
    np.testing.assert_allclose(result.residuals, 0.0, atol=1e-12)


def test_t1_fit_result_constructor_accepts_base_public_signature() -> None:
    result = fit_mod.T1FitResult(
        params=T1FitParams(Q_cap=8e5, Temp=0.06),
        stderr=T1FitParams(Q_cap=0.0, Temp=0.0),
        fixed=("Q_cap", "Temp"),
        free=(),
        model_T1s=np.array([10.0], dtype=np.float64),
        residuals=np.array([0.0], dtype=np.float64),
        cost=0.0,
        reduced_chi2=0.0,
        success=True,
        message="ok",
        optimizer_result=None,
    )

    assert result.correlation is None
    assert result.T1_error_resolution is None
    assert result.flux_weights.effective_observation_count == 0.0


def test_fit_t1_noise_params_fills_nan_errors_by_flux_bin() -> None:
    fluxs = np.array([0.490, 0.491, 0.510], dtype=np.float64)
    init = T1FitParams(Q_cap=8e5, Temp=0.06)
    T1s = calculate_eff_t1_vs_flux_fast(
        _PARAMS, fluxs, _noise_channels(init), init.Temp
    )
    T1errs = np.array([100.0, np.nan, np.nan], dtype=np.float64)

    result = fit_t1_noise_params(
        fluxs,
        T1s,
        _PARAMS,
        init=init,
        fixed=("Q_cap", "Temp"),
        T1errs=T1errs,
        T1_error_policy=MeasurementErrorPolicy(
            nan_policy="bin_median",
            fallback_error=250.0,
        ),
        flux_weighting=FluxResidualWeighting(
            mode="equal_flux_bin",
            bin_width=0.005,
            origin=0.49,
        ),
    )

    assert result.T1_error_resolution is not None
    np.testing.assert_allclose(
        result.T1_error_resolution.effective_errors,
        [100.0, 100.0, 100.0],
    )
    np.testing.assert_array_equal(
        result.T1_error_resolution.bin_fill_mask,
        [False, True, False],
    )
    np.testing.assert_array_equal(
        result.T1_error_resolution.global_fill_mask,
        [False, False, True],
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bounds": {"bad": (1.0, 2.0)}}, "unknown bound"),
        ({"fixed": ("bad",)}, "unknown fixed"),
        ({"bounds": {"Q_cap": (9e5, 1e6)}}, "within bounds"),
        ({"bounds": {"Q_cap": (None, 1e6)}}, "only Temp lower bound"),
    ],
)
def test_fit_t1_noise_params_validates_params_and_bounds(
    kwargs: dict[str, Any], match: str
) -> None:
    fluxs = _sample_fluxs()
    T1s = np.full_like(fluxs, 1e5)

    with pytest.raises(ValueError, match=match):
        fit_t1_noise_params(
            fluxs,
            T1s,
            _PARAMS,
            init=T1FitParams(Q_cap=8e5, x_qp=1.5e-6, Q_ind=3e7, Temp=0.06),
            **kwargs,
        )


@pytest.mark.parametrize(
    ("init", "kwargs", "match"),
    [
        (T1FitParams(Temp=0.06), {}, "at least one T1 noise parameter"),
        (
            T1FitParams(Temp=0.06, Q_cap=8e5, x_qp=1.5e-6),
            {"fixed": ("Q_ind",)},
            "inactive",
        ),
        (
            T1FitParams(Temp=0.06, Q_cap=8e5, x_qp=1.5e-6),
            {"bounds": {"Q_ind": (1.0e5, 1.0e10)}},
            "inactive",
        ),
    ],
)
def test_fit_t1_noise_params_validates_whitelist_metadata(
    init: T1FitParams, kwargs: dict[str, Any], match: str
) -> None:
    fluxs = _sample_fluxs()
    T1s = np.full_like(fluxs, 1e5)

    with pytest.raises(ValueError, match=match):
        fit_t1_noise_params(
            fluxs,
            T1s,
            _PARAMS,
            init=init,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("fluxs", "T1s", "T1errs", "match"),
    [
        (np.array([0.1, 0.2]), np.array([1.0]), None, "same shape"),
        (np.array([0.1]), np.array([-1.0]), None, "finite and positive"),
        (np.array([0.1]), np.array([1.0]), np.array([0.0]), "positive finite"),
        (np.array([0.1]), np.array([1.0]), np.array([np.inf]), "positive finite"),
    ],
)
def test_fit_t1_noise_params_validates_data(
    fluxs: NDArray[np.float64],
    T1s: NDArray[np.float64],
    T1errs: NDArray[np.float64] | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        fit_t1_noise_params(
            fluxs,
            T1s,
            _PARAMS,
            init=T1FitParams(Q_cap=8e5, x_qp=1.5e-6, Q_ind=3e7, Temp=0.06),
            T1errs=T1errs,
        )


def _noise_channels(
    params: T1FitParams,
) -> list[tuple[str, dict[str, float]]]:
    channels: list[tuple[str, dict[str, float]]] = []
    if params.Q_cap is not None:
        channels.append(("t1_capacitive", {"Q_cap": params.Q_cap}))
    if params.x_qp is not None:
        channels.append(("t1_quasiparticle_tunneling", {"x_qp": params.x_qp}))
    if params.Q_ind is not None:
        channels.append(("t1_inductive", {"Q_ind": params.Q_ind}))
    return channels


def test_fit_t1_noise_params_uses_provided_noise_params_as_whitelist() -> None:
    init = T1FitParams(Temp=0.06, x_qp=1.5e-6)
    fluxs = np.array([0.49, 0.51], dtype=np.float64)
    T1s = calculate_eff_t1_vs_flux_fast(
        _PARAMS, fluxs, _noise_channels(init), init.Temp
    )

    result = fit_t1_noise_params(fluxs, T1s, _PARAMS, init=init, fixed=("x_qp", "Temp"))

    assert result.params.x_qp == init.x_qp
    assert result.params.Q_cap is None
    assert result.params.Q_ind is None
    assert result.stderr.Q_cap is None
    assert result.stderr.Q_ind is None
    assert result.fixed == ("x_qp", "Temp")
    assert result.free == ()
    np.testing.assert_allclose(result.model_T1s, T1s)
