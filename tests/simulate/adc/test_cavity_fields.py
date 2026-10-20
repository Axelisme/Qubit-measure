"""Vacuum-start intracavity fields in the emitted rotating frame."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import trapezoid
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


def test_cavity_field_average_four_us_is_finite_and_additive():
    model = adc.ReadoutModel(
        "long-flat",
        ("g", "e"),
        np.array([[7.19, 7.08], [7.19003, 7.0802]]),
        0,
        np.array([0.0009, 0.0008]),
        np.array([0.006, 0.005]),
        np.array([[0.0, 0.3], [0.1, 0.0]]),
    )
    pulses = (adc.ReadoutPulse(0, 5500, 0.15, 7.19),)
    average = adc.cavity_field_average(model, pulses, 1500, 5500)
    first = adc.cavity_field_average(model, pulses, 1500, 2250)
    second = adc.cavity_field_average(model, pulses, 2250, 5500)
    assert np.isfinite(average.photons).all()
    assert np.isfinite(average.field_products).all()
    np.testing.assert_allclose(
        average.field_products,
        (750 * first.field_products + 3250 * second.field_products) / 4000,
        rtol=1e-7,
        atol=1e-13,
    )
    products = average.field_products.reshape(4, 4)
    np.testing.assert_allclose(products, products.conj().T, atol=1e-13)
    np.testing.assert_allclose(average.photons.ravel(), products.diagonal().real)
    # Directional rates do not mix the fixed state branches.
    no_rates = adc.cavity_field_average(
        replace(model, rates_per_ns=np.zeros((2, 2))), pulses, 1500, 5500
    )
    np.testing.assert_allclose(
        average.field_products, no_rates.field_products, atol=1e-13
    )


@pytest.mark.parametrize("shaped", [False, True])
def test_cavity_field_average_preserves_coherent_carriers_and_ringdown(shaped):
    model = adc.ReadoutModel(
        "moments",
        ("g", "e"),
        np.array([[6.0, 6.11], [6.002, 6.105]]),
        0,
        np.array([0.02, 0.03]),
        np.array([0.06, 0.04]),
        np.zeros((2, 2)),
    )
    envelope = (
        adc.PiecewiseLinearEnvelope(
            np.array([0.0, 6.0, 14.0, 20.0]), np.array([0.0, 1.0, 0.4j, 0.0])
        )
        if shaped
        else None
    )
    pulses = (
        adc.ReadoutPulse(0, 20, 0.3j, 6.0, envelope=envelope),
        adc.ReadoutPulse(8, 30, 0.2, 6.07, 0.3),
    )
    average = adc.cavity_field_average(model, pulses, 5, 45)
    times = np.linspace(5, 45, 8001)
    fields = adc.cavity_fields(model, pulses, times)
    products = np.einsum("tsr,tuq->tsruq", fields, fields.conj())
    expected = trapezoid(products, times, axis=0) / 40
    np.testing.assert_allclose(average.field_products, expected, rtol=2e-6, atol=1e-10)
    np.testing.assert_allclose(
        average.photons,
        trapezoid(np.abs(fields) ** 2, times, axis=0) / 40,
        rtol=2e-6,
        atol=1e-10,
    )
    ringdown = adc.cavity_field_average(model, pulses, 30, 45)
    assert np.max(ringdown.photons) > 0
    cancelled = adc.cavity_field_average(
        model, (pulses[0], replace(pulses[0], amplitude=-pulses[0].amplitude)), 5, 45
    )
    np.testing.assert_allclose(cancelled.field_products, 0, atol=1e-13)


def test_cavity_field_average_matches_isolated_resonant_integral():
    model = adc.ReadoutModel(
        "isolated-integral",
        ("g",),
        np.array([[6.0, 7.0]]),
        6.0,
        np.array([0.02, 0.0]),
        np.array([0.06, 0.0]),
        np.zeros((1, 1)),
    )
    pulse = adc.ReadoutPulse(0, 4000, 0.3, 6.0)
    average = adc.cavity_field_average(model, (pulse,), 0, 4000)
    # Integral of n_ss*(1-exp(-a*t))², a=kappa/2=.04/ns.
    expected = (2 * np.sqrt(0.03) * 0.3 / 0.08) ** 2 * (
        1 - 2 * (-np.expm1(-160)) / 160 + (-np.expm1(-320)) / 320
    )
    np.testing.assert_allclose(average.photons, [[expected, 0]], rtol=1e-10, atol=1e-13)
    vacuum = adc.cavity_field_average(model, (), -10, 20)
    np.testing.assert_array_equal(vacuum.photons, 0)
    np.testing.assert_array_equal(vacuum.field_products, 0)


@pytest.mark.parametrize("interval", [(5.0, 5.0), (6.0, 5.0), (0.0, np.inf)])
def test_cavity_field_average_rejects_invalid_interval(interval):
    model = adc.ReadoutModel(
        "interval",
        ("g",),
        np.array([[6.0, 7.0]]),
        6.0,
        np.ones(2),
        np.ones(2),
        np.zeros((1, 1)),
    )
    with pytest.raises(ValueError, match="interval"):
        adc.cavity_field_average(model, (), *interval)


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
