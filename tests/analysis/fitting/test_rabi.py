import numpy as np
from zcu_tools.analysis.fitting.base import cosfunc, decaycos
from zcu_tools.analysis.fitting.rabi import fit_rabi


def test_fit_rabi_no_decay():
    xs = np.linspace(0, 4, 400)
    y0, yscale, freq, phase = 0.0, 1.0, 0.5, 0.0
    ys = cosfunc(xs, y0, yscale, freq, phase)
    pi_x, pi_x_err, pi2_x, pi2_x_err, fit_freq, freq_err, _, _ = fit_rabi(
        xs, ys, decay=False
    )
    assert abs(fit_freq - freq) < 1e-3
    assert abs(pi_x - 1.0) < 1e-2
    assert abs(pi2_x - 0.5) < 1e-2
    assert freq_err >= 0
    assert pi_x_err >= 0
    assert pi2_x_err >= 0


def test_fit_rabi_with_decay():
    xs = np.linspace(0, 6, 400)
    y0, yscale, freq, phase, tau = 0.0, 1.0, 0.4, 0.0, 10.0
    ys = decaycos(xs, y0, yscale, freq, phase, tau)
    pi_x, pi_x_err, _, pi2_x_err, fit_freq, freq_err, _, _ = fit_rabi(
        xs, ys, decay=True
    )
    assert abs(fit_freq - freq) / freq < 5e-2
    assert abs(pi_x - 1.25) < 5e-2
    assert freq_err >= 0
    assert pi_x_err >= 0
    assert pi2_x_err >= 0


def test_fit_rabi_retains_parameter_tuple_shape_for_both_models():
    xs = np.linspace(0, 6, 400)
    for decay in (False, True):
        expected_params = (0.2, 0.8, 0.5, 0.0, 10.0) if decay else (0.2, 0.8, 0.5, 0.0)
        model = decaycos if decay else cosfunc
        signals = model(xs, *expected_params)

        pi_x, pi_x_err, pi2_x, pi2_x_err, freq, freq_err, fitted, fit_result = fit_rabi(
            xs, signals, decay=decay, init_phase=0.0
        )
        params, covariance = fit_result

        assert isinstance(params, tuple)
        assert len(params) == (5 if decay else 4)
        np.testing.assert_allclose(params, expected_params, rtol=1e-5, atol=1e-7)
        assert covariance.shape == (len(params), len(params))
        assert np.all(np.isfinite(covariance))
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
        np.testing.assert_allclose(fitted, signals, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose((pi_x, pi2_x, freq), (1.0, 0.5, 0.5), atol=1e-6)
        assert 0 <= pi_x_err < 1e-6
        assert 0 <= pi2_x_err < 1e-6
        assert 0 <= freq_err < 1e-6
