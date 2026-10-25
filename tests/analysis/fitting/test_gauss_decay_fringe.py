"""Gaussian-envelope fringes through the public numerical fitting seam."""

import numpy as np
import pytest
from zcu_tools.analysis import fitting


def test_gaussian_fringe_recovers_zero_centered_envelope_and_frequency() -> None:
    xs = np.linspace(0.3, 24.0, 240)
    expected = 0.23 + 0.8 * np.exp(-((xs / 6.0) ** 2)) * np.cos(
        2 * np.pi * (0.7 * xs + 37.0 / 360)
    )
    fit = getattr(fitting, "fit_gauss_decay_fringe", None)
    assert fit is not None, "Public Gaussian-envelope fringe fitting is missing"
    lifetime, error, frequency, frequency_error, curve, _ = fit(xs, expected)
    assert lifetime == pytest.approx(6.0, rel=1e-3)
    assert frequency == pytest.approx(0.7, rel=1e-3)
    assert np.isfinite(error) and np.isfinite(frequency_error)
    np.testing.assert_allclose(curve, expected, atol=1e-6)
