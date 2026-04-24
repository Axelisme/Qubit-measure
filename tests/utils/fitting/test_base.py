import numpy as np
from zcu_tools.utils.fitting.base import (
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
