import logging

import matplotlib.pyplot as plt
import numpy as np
import pytest
from zcu_tools.analysis.fitting.multi_decay import (
    TransitionRates,
    excited_population_decay_times,
    fit_dual_transition_rates,
    fit_dual_with_vadality,
    fit_transition_rates,
    fit_with_vadality,
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


@pytest.mark.parametrize(
    ("rates", "expected"),
    [
        # Symmetric rates: e->e weights are 1/2 at 1/6 µs and 1/6 at 1/12 µs.
        (TransitionRates(1, 1, 4, 4, 4, 4), (1 / 6, 1 / 12)),
        # Increase g<->e and reduce other links to exchange the mode lifetimes.
        (TransitionRates(5, 5, 2, 2, 2, 2), (1 / 12, 1 / 6)),
    ],
    ids=["slow-dominant", "fast-dominant"],
)
def test_excited_population_decay_times_order_by_amplitude(rates, expected):
    times = excited_population_decay_times(rates)

    assert (times.t1, times.t1_b) == pytest.approx(expected)


def test_equal_excited_population_amplitudes_choose_longer_and_log(caplog):
    # e<->other with other->g: P(e | e) is the sum of two exponentials with
    # coefficients 1/2 and rates 2 ± sqrt(2).
    rates = TransitionRates(0, 0, 2, 1, 0, 1)
    with caplog.at_level(logging.INFO, logger="zcu_tools.analysis.fitting.multi_decay"):
        times = excited_population_decay_times(rates)

    assert (times.t1, times.t1_b) == pytest.approx(
        (1 / (2 - np.sqrt(2)), 1 / (2 + np.sqrt(2)))
    )
    (message,) = caplog.messages
    assert "equal" in message.lower() and "longer" in message.lower()


@pytest.mark.parametrize("dual", [False, True], ids=["single", "dual"])
def test_prefix_fit_diagnostics_complete_on_agg(dual):
    previous_backend = plt.get_backend()
    previous_figures = set(plt.get_fignums())
    plt.switch_backend("Agg")
    try:
        times = np.linspace(0.0, 20.0, 20)
        rates = TransitionRates(0.1, 0.05, 0.08, 0.04, 0.02, 0.01)
        ground = model_func(times, *rates.as_tuple(), 0.96, 0.03)
        if dual:
            excited = model_func(times, *rates.as_tuple(), 0.04, 0.95)
            result = fit_dual_with_vadality(times, ground, excited)
        else:
            result = fit_with_vadality(times, ground)
        assert result is None
    finally:
        for figure_number in set(plt.get_fignums()) - previous_figures:
            plt.close(figure_number)
        plt.switch_backend(previous_backend)
