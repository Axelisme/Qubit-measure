import numpy as np
from zcu_tools.utils.fitting.multi_decay import fit_transition_rates, model_func


def test_fit_transition_rates_recovers_rates():
    times = np.linspace(0, 20, 200)
    T_ge, T_eg, T_eo, T_oe, T_go, T_og = 0.1, 0.05, 0.08, 0.04, 0.02, 0.01
    pg0, pe0 = 1.0, 0.0

    pops = model_func(times, T_ge, T_eg, T_eo, T_oe, T_go, T_og, pg0, pe0)

    rates, _, _, _ = fit_transition_rates(times, pops)
    true_rates = np.array([T_ge, T_eg, T_eo, T_oe, T_go, T_og])
    assert np.allclose(rates, true_rates, atol=2e-2)
