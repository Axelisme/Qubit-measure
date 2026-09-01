import numpy as np
from zcu_tools.utils.fitting.multi_decay import (
    TransitionRates,
    fit_dual_transition_rates,
    fit_transition_rates,
    model_func,
)


def test_fit_transition_rates_recovers_rates():
    times = np.linspace(0, 20, 200)
    T_ge, T_eg, T_eo, T_oe, T_go, T_og = 0.1, 0.05, 0.08, 0.04, 0.02, 0.01
    pg0, pe0 = 1.0, 0.0

    pops = model_func(times, T_ge, T_eg, T_eo, T_oe, T_go, T_og, pg0, pe0)

    rates, _, _, _ = fit_transition_rates(times, pops)
    true_rates = np.array([T_ge, T_eg, T_eo, T_oe, T_go, T_og])
    assert np.allclose(rates, true_rates, atol=2e-2)


def test_fit_dual_transition_rates_returns_named_global_result():
    rng = np.random.default_rng(20260901)
    times = np.linspace(0, 20, 160)
    true_rates = TransitionRates(0.1, 0.05, 0.08, 0.04, 0.02, 0.01)
    populations1 = model_func(times, *true_rates.as_tuple(), 0.96, 0.03)
    populations2 = model_func(times, *true_rates.as_tuple(), 0.04, 0.95)
    populations1 += rng.normal(0.0, 2e-4, populations1.shape)
    populations2 += rng.normal(0.0, 2e-4, populations2.shape)

    result = fit_dual_transition_rates(times, populations1, populations2)

    assert result.diagnostics.valid
    assert result.diagnostics.covariance_accurate
    np.testing.assert_allclose(
        result.rates.as_array(), true_rates.as_array(), atol=3e-3
    )
    assert np.all(np.isfinite(result.rate_errors.as_array()))
    assert np.all(result.rate_errors.as_array() > 0.0)
    np.testing.assert_allclose(result.initial_populations1, (0.96, 0.03), atol=2e-3)
    np.testing.assert_allclose(result.initial_populations2, (0.04, 0.95), atol=2e-3)
    assert result.fitted_populations1.shape == populations1.shape
    assert result.fitted_populations2.shape == populations2.shape
    assert result.covariance.shape == (10, 10)
    assert result.shared_fit.parameter_names[:6] == TransitionRates.NAMES
