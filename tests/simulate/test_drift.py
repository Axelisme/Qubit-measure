"""Log-OU numerical contracts with supplied times and RNG streams."""

import math

import numpy as np
import pytest
from zcu_tools.simulate.drift import LogOU


def test_log_ou_exact_transition_and_stationary_limits():
    process = LogOU(10.0, sigma=0.5, tau_s=2.0, rng=np.random.default_rng(123))
    assert process.sample(100.0) == 10.0
    assert process.sample(100.0) == 10.0
    # PCG64(123)'s first standard normal is -0.9891213503478509.
    # At dt=tau*ln(2), retention is 1/2, innovation std is sigma*sqrt(3/4).
    value = process.sample(100 + 2 * math.log(2))
    assert math.log(value / 10) == pytest.approx(-0.4283021082132524)
    distant = process.sample(10_000)
    assert math.log(distant / 10) == pytest.approx(-0.1838933257339416)
    with pytest.raises(ValueError, match="time"):
        process.sample(9999)


def test_log_ou_long_sequence_has_log_stationary_mean_and_variance():
    process = LogOU(2.0, sigma=0.3, tau_s=1.0, rng=np.random.default_rng(44))
    logs = np.array([math.log(process.sample(float(t))) for t in range(30_000)])
    assert logs[100:].mean() == pytest.approx(math.log(2), abs=0.012)
    assert logs[100:].var() == pytest.approx(0.09, abs=0.004)
