import numpy as np
import pytest
import zcu_tools.analysis.fitting.base.base as base_module
from zcu_tools.analysis.fitting.base import (
    asym_lorfunc,
    cosfunc,
    decaycos,
    dual_expfunc,
    expfunc,
    fit_asym_lor,
    fit_dualexp,
    fit_gauss,
    fitcos,
    fitdecaycos,
    fitexp,
    fitlor,
    fitsinc,
    gauss_func,
    lorfunc,
    sincfunc,
)
from zcu_tools.analysis.fitting.base.base import fit_func


def test_fitexp_recovers_parameters():
    x = np.linspace(0, 10, 200)
    true = (0.2, 1.3, 2.5)
    y = expfunc(x, *true)
    pOpt, _ = fitexp(x, y)
    assert np.allclose(pOpt, true, rtol=1e-3)


def test_fit_dualexp_recovers_parameters():
    x = np.linspace(0, 30, 400)
    true = (0.1, 0.8, 1.5, 0.6, 8.0)
    y = dual_expfunc(x, *true)
    pOpt, _ = fit_dualexp(x, y)
    y_fit = dual_expfunc(x, *pOpt)
    assert np.max(np.abs(y_fit - y)) < 1e-3


def test_fitcos_recovers_parameters():
    x = np.linspace(0, 4, 400)
    true = (0.1, 0.7, 1.2, 45.0)
    y = cosfunc(x, *true)
    pOpt, _ = fitcos(x, y)
    assert abs(pOpt[0] - true[0]) < 1e-3
    assert abs(pOpt[1] - true[1]) < 1e-3
    assert abs(pOpt[2] - true[2]) < 1e-3
    assert abs((pOpt[3] - true[3] + 180) % 360 - 180) < 1


def test_fitdecaycos_recovers_parameters():
    x = np.linspace(0, 8, 600)
    true = (0.05, 0.6, 1.0, 0.0, 10.0)
    y = decaycos(x, *true)
    pOpt, _ = fitdecaycos(x, y)
    y_fit = decaycos(x, *pOpt)
    assert np.max(np.abs(y_fit - y)) < 1e-2
    assert abs(pOpt[2] - true[2]) < 1e-2
    assert abs(pOpt[4] - true[4]) / true[4] < 1e-1


def test_fitlor_recovers_parameters():
    x = np.linspace(-5, 5, 400)
    true = (0.1, 0.0, 1.0, 1.0, 0.5)
    y = lorfunc(x, *true)
    pOpt, _ = fitlor(x, y)
    assert abs(pOpt[3] - true[3]) < 1e-3
    assert abs(pOpt[4] - true[4]) < 5e-2


def test_fit_asym_lor_symmetric_case():
    x = np.linspace(-5, 5, 400)
    true = (0.1, 0.0, 1.0, 0.5, 0.6, 0.0)
    y = asym_lorfunc(x, *true)
    pOpt, _ = fit_asym_lor(x, y)
    y_fit = asym_lorfunc(x, *pOpt)
    assert np.max(np.abs(y_fit - y)) < 1e-3


def test_fitsinc_recovers_parameters():
    x = np.linspace(-5, 5, 400)
    true = (0.0, 0.0, 1.0, 0.7, 0.9)
    y = sincfunc(x, *true)
    pOpt, _ = fitsinc(x, y)
    assert abs(pOpt[3] - true[3]) < 1e-2
    assert abs(pOpt[4] - true[4]) < 1e-2


def test_fit_gauss_recovers_parameters():
    x = np.linspace(-5, 5, 400)
    true = (0.0, 1.0, 0.3, 0.8)
    y = gauss_func(x, *true)
    pOpt, _ = fit_gauss(x, y)
    assert abs(pOpt[2] - true[2]) < 1e-2
    assert abs(pOpt[3] - true[3]) < 1e-2


def test_fit_func_warns_when_falling_back_to_init_p(
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_curve_fit(*args, **kwargs):
        raise RuntimeError("no convergence")

    monkeypatch.setattr(base_module.sp.optimize, "curve_fit", fail_curve_fit)

    with pytest.warns(RuntimeWarning, match="returning init_p fallback"):
        p_opt, p_cov = fit_func(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            lambda x, a, b: a * x + b,
            init_p=[1.0, None],
        )

    assert p_opt[0] == 1.0
    assert np.isnan(p_opt[1])
    assert np.all(np.isinf(p_cov))


def test_fit_func_all_none_fixedparams_matches_unfixed_fit():
    x = np.linspace(0.0, 2.0, 50)
    y = 2.0 * x + 1.0 + 0.02 * np.sin(3.0 * x)

    def model(x, slope, intercept):
        return slope * x + intercept

    expected_opt, expected_cov = fit_func(x, y, model, init_p=[1.0, 0.0])
    p_opt, p_cov = fit_func(x, y, model, init_p=[1.0, 0.0], fixedparams=[None, None])

    np.testing.assert_allclose(p_opt, expected_opt)
    np.testing.assert_allclose(p_cov, expected_cov)
    np.testing.assert_allclose(p_opt, [2.0, 1.0], atol=0.03)
    assert np.all(np.isfinite(p_cov))


@pytest.mark.parametrize(
    "bounds, expected_slope",
    [(None, 5.0), (([0.0, -np.inf], [3.0, np.inf]), 3.0)],
)
def test_fit_func_fixed_parameter_preserves_bounds_and_covariance(
    bounds, expected_slope
):
    x = np.linspace(0.0, 2.0, 50)
    y = 5.0 * x + 1.0
    p_opt, p_cov = fit_func(
        x,
        y,
        lambda x, slope, intercept: slope * x + intercept,
        init_p=[2.0, 0.0],
        bounds=bounds,
        fixedparams=[None, 1.0],
    )

    np.testing.assert_allclose(p_opt, [expected_slope, 1.0], atol=1e-6)
    assert p_cov.shape == (2, 2)
    assert np.all(np.isfinite(p_cov))
    np.testing.assert_array_equal(p_cov[1, :], [0.0, 0.0])
    np.testing.assert_array_equal(p_cov[:, 1], [0.0, 0.0])
