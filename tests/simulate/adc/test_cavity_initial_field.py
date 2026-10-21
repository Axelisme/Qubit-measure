"""Shared nonzero cavity memory at the public ADC boundary."""

from dataclasses import replace
from inspect import signature

import numpy as np
import pytest
from scipy.integrate import solve_ivp, trapezoid
from zcu_tools.simulate import adc


def _model():
    return adc.ReadoutModel(
        "memory",
        ("g", "e"),
        np.array([[6.0, 6.11], [6.002, 6.105]]),
        6.03,
        np.array([0.02, 0.03]),
        np.array([0.06, 0.04]),
        np.zeros((2, 2)),
    )


@pytest.mark.parametrize("variant", range(12))
def test_flat_cache_owns_exact_moments_for_complete_inputs(variant):
    cache_type = getattr(adc, "CavityAverageCache", None)
    assert cache_type is not None, "ADC must expose the bounded flat-average owner"
    cache = cache_type()
    model = _model()
    pulses = (
        adc.ReadoutPulse(2, 14, 0.3j, 6.01, 0.2),
        adc.ReadoutPulse(4, 11, 0.15, 6.04, -0.3),
    )
    initial = np.array([0.2 + 0.1j, -0.3j])
    origin, end = 5.0, 20.0
    assert callable(getattr(cache, "average", None)), "ADC owner must solve averages"
    first = cache.average(
        model,
        pulses,
        5,
        end,
        initial_field=initial,
        initial_time_ns=origin,
    )
    if variant == 0:
        initial = initial + 0.01j
    elif variant == 1:
        origin = 4.0
    elif variant == 2:
        end = 21.0
    elif variant == 3:
        model = replace(model, frequencies_ghz=model.frequencies_ghz + 0.001)
    elif variant == 4:
        model = replace(model, kappa_internal_per_ns=np.zeros(2))
    elif variant == 5:
        model = replace(
            model, kappa_internal_per_ns=np.zeros(2), kappa_external_per_ns=np.zeros(2)
        )
    elif variant == 6:
        # A repeated eigenvalue with rank-one damping exercises defective A.
        model = replace(
            model,
            frequencies_ghz=np.tile([6.01, 6.01 + 0.02 / np.pi], (2, 1)),
            kappa_internal_per_ns=np.zeros(2),
            kappa_external_per_ns=np.full(2, 0.04),
        )
    elif variant == 7:
        pulses = (replace(pulses[0], phase_rad=0.21), pulses[1])
    elif variant == 8:
        pulses = (replace(pulses[0], frequency_ghz=6.011), pulses[1])
    elif variant == 9:
        pulses = tuple(reversed(pulses))
    elif variant == 10:
        pulses = (replace(pulses[0], amplitude=0.31j), pulses[1])
    else:
        pulses = (replace(pulses[0], end_ns=13), pulses[1])
    expected = adc.cavity_field_average(
        model, pulses, 5, end, initial_field=initial, initial_time_ns=origin
    )
    actual = cache.average(
        model,
        pulses,
        5,
        end,
        initial_field=initial,
        initial_time_ns=origin,
    )
    np.testing.assert_array_equal(actual.photons, expected.photons)
    np.testing.assert_array_equal(actual.field_products, expected.field_products)
    assert cache.misses == 2 and cache.hits == 0
    actual.photons[:] = -1
    actual.field_products[:] = -1
    first.field_products[:] = -2
    again = cache.average(
        model,
        pulses,
        5,
        end,
        initial_field=initial,
        initial_time_ns=origin,
    )
    np.testing.assert_array_equal(again.photons, expected.photons)
    np.testing.assert_array_equal(again.field_products, expected.field_products)
    assert cache.hits == 1 and 0 < cache.resident_bytes <= cache.max_bytes


def test_flat_cache_capacity_clear_bypass_and_failure():
    cache_type = getattr(adc, "CavityAverageCache", None)
    assert cache_type is not None, "ADC must expose the bounded flat-average owner"
    for capacity in (-1, 1.5, True):
        with pytest.raises(ValueError, match="nonnegative integer"):
            cache_type(capacity)
    model = _model()
    pulses = (adc.ReadoutPulse(0, 10, 0.2, 6.01),)
    cache = cache_type()
    assert callable(getattr(cache, "average", None)), "ADC owner must solve averages"
    expected = cache.average(model, pulses, 0, 20)
    resident = cache.resident_bytes
    assert resident > expected.photons.nbytes + expected.field_products.nbytes
    bounded = cache_type(resident)
    bounded.average(model, pulses, 0, 20)
    bounded.average(model, pulses, 0, 21)
    assert bounded.resident_bytes == resident and bounded.misses == 2
    bounded.average(model, pulses, 0, 20)
    assert bounded.hits == 1
    for capacity in (0, 1):
        disabled = cache_type(capacity)
        for _ in range(2):
            result = disabled.average(model, pulses, 0, 20)
            np.testing.assert_array_equal(
                result.field_products, expected.field_products
            )
        assert disabled.resident_bytes == 0 and disabled.hits == 0
    shaped = (replace(pulses[0], envelope=lambda t: 1.0),)
    cache.average(model, shaped, 0, 20)
    assert cache.misses == 1 and cache.hits == 0
    with pytest.raises(ValueError, match="finite and positive"):
        cache.average(model, pulses, 20, 0)
    with pytest.raises(ValueError, match="finite"):
        cache.average(model, (replace(pulses[0], envelope=lambda t: np.nan),), 0, 20)
    assert cache.resident_bytes == resident and cache.misses == 1
    cache.clear()
    assert cache.resident_bytes == 0 and cache.misses == 1 and cache.hits == 0
    cache.average(model, pulses, 0, 20)
    assert cache.misses == 2


def _require_initial_field():
    assert "initial_field" in signature(adc.cavity_fields).parameters
    assert "initial_field" in signature(adc.cavity_field_average).parameters
    assert "initial_time_ns" in signature(adc.cavity_fields).parameters
    assert "initial_time_ns" in signature(adc.cavity_field_average).parameters


@pytest.mark.parametrize("shaped", [False, True], ids=["flat", "mixed-shaped"])
def test_initial_field_retains_common_reference_pulse_coordinates_and_moments(shaped):
    _require_initial_field()
    model = _model()
    envelope = (
        adc.PiecewiseLinearEnvelope(
            np.array([0.0, 6.0, 14.0, 20.0]), np.array([0.0, 1.0, 0.4j, 0.0])
        )
        if shaped
        else None
    )
    pulses = (
        adc.ReadoutPulse(1.3, 21.3, 0.3j, 6.03, 0.2, envelope=envelope),
        adc.ReadoutPulse(8.4, 30.7, 0.2, 6.07, -0.4),
    )
    initial = np.array([0.2 + 0.3j, -0.1j])
    times = np.linspace(5.0, 45.0, 4001)
    v = np.sqrt(model.kappa_external_per_ns / 2)
    matrices = np.array(
        [
            np.diag(model.kappa_internal_per_ns / 2 + 2j * np.pi * f) + np.outer(v, v)
            for f in model.frequencies_ghz - model.reference_frequency_ghz
        ]
    )

    def rhs(t, flat):
        fields = flat.reshape(2, 2)
        incident = sum(p.incident(t, model.reference_frequency_ghz) for p in pulses)
        return (-np.einsum("sru,su->sr", matrices, fields) + v * incident).ravel()

    reference = solve_ivp(
        rhs,
        (5.0, 45.0),
        np.tile(initial, (2, 1)).ravel(),
        t_eval=times,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
        max_step=0.05,
    ).y.T.reshape(-1, 2, 2)
    fields = adc.cavity_fields(
        model, pulses, times, initial_field=initial, initial_time_ns=5.0
    )
    np.testing.assert_allclose(fields, reference, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(fields[0], np.tile(initial, (2, 1)), atol=1e-12)
    average = adc.cavity_field_average(
        model, pulses, 5.0, 45.0, initial_field=initial, initial_time_ns=5.0
    )
    products = np.einsum("tsr,tuq->tsruq", reference, reference.conj())
    np.testing.assert_allclose(
        average.field_products,
        trapezoid(products, times, axis=0) / 40,
        rtol=1e-5,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        average.photons,
        average.field_products.reshape(4, 4).diagonal().real.reshape(2, 2),
    )
    np.testing.assert_allclose(
        average.field_products,
        average.field_products.transpose(2, 3, 0, 1).conj(),
        atol=1e-12,
    )


def test_initial_field_without_pulses_rings_down_and_has_analytic_average():
    _require_initial_field()
    model = adc.ReadoutModel(
        "isolated-memory",
        ("g", "e"),
        np.array([[6.0, 7.0], [6.0, 7.0]]),
        6.0,
        np.array([0.02, 0.0]),
        np.array([0.06, 0.0]),
        np.zeros((2, 2)),
    )
    initial = np.array([0.2 + 0.3j, 0j])
    times = np.array([10.0, 15.0, 30.0])
    fields = adc.cavity_fields(
        model, (), times, initial_field=initial, initial_time_ns=10.0
    )
    expected = initial[0] * np.exp(-0.04 * (times - 10))
    np.testing.assert_allclose(fields[:, :, 0], np.tile(expected[:, None], (1, 2)))
    average = adc.cavity_field_average(
        model, (), 10.0, 30.0, initial_field=initial, initial_time_ns=10.0
    )
    expected_photons = abs(initial[0]) ** 2 * (-np.expm1(-1.6)) / 1.6
    np.testing.assert_allclose(average.photons, [[expected_photons, 0]] * 2, atol=1e-12)


@pytest.mark.parametrize(
    "initial", [np.zeros((2, 2)), np.zeros(1), np.array([np.nan, 0j])]
)
def test_initial_field_fail_fast(initial):
    _require_initial_field()
    with pytest.raises(ValueError, match="initial field"):
        adc.cavity_fields(_model(), (), np.array([0.0, 10.0]), initial_field=initial)
    with pytest.raises(ValueError, match="initial field"):
        adc.cavity_field_average(_model(), (), 0.0, 10.0, initial_field=initial)


def test_omitted_initial_field_equals_explicit_vacuum_at_pulse_origin():
    _require_initial_field()
    model = _model()
    pulses = (adc.ReadoutPulse(2.0, 22.0, 0.3j, 6.03),)
    times = np.array([2.0, 7.0, 22.0, 32.0])
    np.testing.assert_allclose(
        adc.cavity_fields(model, pulses, times),
        adc.cavity_fields(
            model, pulses, times, initial_field=np.zeros(2, dtype=complex)
        ),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        adc.cavity_field_average(model, pulses, 2.0, 32.0).field_products,
        adc.cavity_field_average(
            model, pulses, 2.0, 32.0, initial_field=np.zeros(2, dtype=complex)
        ).field_products,
        atol=1e-12,
    )
