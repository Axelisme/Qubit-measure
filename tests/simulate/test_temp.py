from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.simulate.temp import boltzmann_distribution, effective_temperature


def test_boltzmann_distribution_golden_values() -> None:
    actual = boltzmann_distribution(np.array([0.0, 1000.0]), eff_T=50.0)

    np.testing.assert_allclose(actual, [0.72309149, 0.27690851])


def test_effective_temperature_two_point_round_trip() -> None:
    freqs = np.array([0.0, 1000.0])
    pops = boltzmann_distribution(freqs, eff_T=50.0)

    temp, err = effective_temperature(list(zip(pops, freqs)))

    assert temp == pytest.approx(50.0)
    assert err == 0.0
