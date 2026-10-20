"""Vacuum-start intracavity fields in the emitted rotating frame."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate import adc

pytestmark = pytest.mark.sim


def test_cavity_fields_preserve_complex_ringup_ringdown_and_superposition():
    # One isolated cavity: alpha=2*v*b/kappa*(1-exp(-kappa*t/2)).
    model = adc.ReadoutModel(
        "isolated",
        ("g", "e"),
        np.array([[6.0, 7.0], [6.0, 7.0]]),
        6.0,
        np.array([0.02, 0.04]),
        np.array([0.06, 0.0]),
        np.array([[0.0, 0.7], [0.2, 0.0]]),
        receiver=adc.Receiver(gain=7, time_of_flight_ns=100),
    )
    pulse = adc.ReadoutPulse(2, 22, 0.3j, 6.0)
    times = np.array([-1.0, 2.0, 7.0, 22.0, 32.0])
    solve = getattr(adc, "cavity_fields", None)
    assert callable(solve), "ADC must expose emitted intracavity fields"
    fields = adc.cavity_fields(model, (pulse,), times)
    expected = (
        2
        * np.sqrt(0.03)
        * 0.3j
        / 0.08
        * np.array(
            [
                0,
                0,
                1 - np.exp(-0.2),
                1 - np.exp(-0.8),
                (1 - np.exp(-0.8)) * np.exp(-0.4),
            ]
        )
    )
    assert fields.shape == (5, 2, 2)
    np.testing.assert_allclose(
        fields[:, :, 0], np.tile(expected[:, None], (1, 2)), atol=1e-10
    )
    np.testing.assert_array_equal(fields[:, :, 1], 0)
    doubled = adc.cavity_fields(model, (pulse, pulse), times)
    np.testing.assert_allclose(doubled, 2 * fields, atol=1e-10)
    cancelled = adc.cavity_fields(
        model, (pulse, replace(pulse, amplitude=-pulse.amplitude)), times
    )
    np.testing.assert_allclose(cancelled, 0, atol=1e-12)
    np.testing.assert_array_equal(
        adc.cavity_fields(model, (), times), np.zeros_like(fields)
    )


@pytest.mark.parametrize(
    "times", [np.array([]), np.array([[1.0]]), np.array([1.0, 1.0]), np.array([np.nan])]
)
def test_cavity_fields_reject_invalid_emitted_grid(times):
    model = adc.ReadoutModel(
        "grid",
        ("g",),
        np.array([[6.0, 7.0]]),
        6.0,
        np.ones(2),
        np.ones(2),
        np.zeros((1, 1)),
    )
    solve = getattr(adc, "cavity_fields", None)
    assert callable(solve), "ADC must expose emitted intracavity fields"
    with pytest.raises(ValueError, match="times"):
        adc.cavity_fields(model, (), times)
