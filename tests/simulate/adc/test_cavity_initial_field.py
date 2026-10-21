"""Shared nonzero cavity memory at the public ADC boundary."""

from inspect import signature

import numpy as np
import pytest
from scipy.integrate import solve_ivp, trapezoid
from zcu_tools.simulate import adc


def _model():
    return adc.ReadoutModel(
        "memory", ("g", "e"), np.array([[6.0, 6.11], [6.002, 6.105]]),
        6.03, np.array([0.02, 0.03]), np.array([0.06, 0.04]), np.zeros((2, 2)),
    )


def _require_initial_field():
    assert "initial_field" in signature(adc.cavity_fields).parameters
    assert "initial_field" in signature(adc.cavity_field_average).parameters


@pytest.mark.parametrize("shaped", [False, True], ids=["flat", "mixed-shaped"])
def test_initial_field_retains_common_reference_pulse_coordinates_and_moments(shaped):
    _require_initial_field()
    model = _model()
    envelope = adc.PiecewiseLinearEnvelope(
        np.array([0., 6., 14., 20.]), np.array([0., 1., .4j, 0.])
    ) if shaped else None
    pulses = (
        adc.ReadoutPulse(1.3, 21.3, .3j, 6.03, .2, envelope=envelope),
        adc.ReadoutPulse(8.4, 30.7, .2, 6.07, -.4),
    )
    initial = np.array([.2 + .3j, -.1j])
    times = np.linspace(5., 45., 4001)
    v = np.sqrt(model.kappa_external_per_ns / 2)
    matrices = np.array([
        np.diag(model.kappa_internal_per_ns / 2 + 2j * np.pi * f)
        + np.outer(v, v)
        for f in model.frequencies_ghz - model.reference_frequency_ghz
    ])

    def rhs(t, flat):
        fields = flat.reshape(2, 2)
        incident = sum(p.incident(t, model.reference_frequency_ghz) for p in pulses)
        return (-np.einsum("sru,su->sr", matrices, fields) + v * incident).ravel()

    reference = solve_ivp(
        rhs, (5., 45.), np.tile(initial, (2, 1)).ravel(), t_eval=times,
        method="DOP853", rtol=1e-11, atol=1e-13, max_step=.05,
    ).y.T.reshape(-1, 2, 2)
    fields = adc.cavity_fields(model, pulses, times, initial_field=initial)
    np.testing.assert_allclose(fields, reference, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(fields[0], np.tile(initial, (2, 1)), atol=1e-12)
    average = adc.cavity_field_average(model, pulses, 5., 45., initial_field=initial)
    products = np.einsum("tsr,tuq->tsruq", reference, reference.conj())
    np.testing.assert_allclose(
        average.field_products, trapezoid(products, times, axis=0) / 40,
        rtol=1e-5, atol=2e-8,
    )
    np.testing.assert_allclose(
        average.photons, average.field_products.reshape(4, 4).diagonal().real.reshape(2, 2)
    )
    np.testing.assert_allclose(
        average.field_products, average.field_products.transpose(2, 3, 0, 1).conj(), atol=1e-12
    )


def test_initial_field_without_pulses_rings_down_and_has_analytic_average():
    _require_initial_field()
    model = adc.ReadoutModel(
        "isolated-memory", ("g", "e"), np.array([[6., 7.], [6., 7.]]),
        6., np.array([.02, 0.]), np.array([.06, 0.]), np.zeros((2, 2)),
    )
    initial = np.array([.2 + .3j, 0j])
    times = np.array([10., 15., 30.])
    fields = adc.cavity_fields(model, (), times, initial_field=initial)
    expected = initial[0] * np.exp(-.04 * (times - 10))
    np.testing.assert_allclose(fields[:, :, 0], np.tile(expected[:, None], (1, 2)))
    average = adc.cavity_field_average(model, (), 10., 30., initial_field=initial)
    expected_photons = abs(initial[0]) ** 2 * (-np.expm1(-1.6)) / 1.6
    np.testing.assert_allclose(average.photons, [[expected_photons, 0]] * 2, atol=1e-12)


@pytest.mark.parametrize("initial", [np.zeros((2, 2)), np.zeros(1), np.array([np.nan, 0j])])
def test_initial_field_fail_fast(initial):
    _require_initial_field()
    for solve, args in (
        (adc.cavity_fields, (np.array([0., 10.]),)),
        (adc.cavity_field_average, (0., 10.)),
    ):
        with pytest.raises(ValueError, match="initial field"):
            solve(_model(), (), *args, initial_field=initial)


def test_omitted_initial_field_equals_explicit_vacuum_at_pulse_origin():
    _require_initial_field()
    model = _model()
    pulses = (adc.ReadoutPulse(2., 22., .3j, 6.03),)
    times = np.array([2., 7., 22., 32.])
    np.testing.assert_allclose(
        adc.cavity_fields(model, pulses, times),
        adc.cavity_fields(model, pulses, times, initial_field=np.zeros(2)), atol=1e-12,
    )
    np.testing.assert_allclose(
        adc.cavity_field_average(model, pulses, 2., 32.).field_products,
        adc.cavity_field_average(model, pulses, 2., 32., initial_field=np.zeros(2)).field_products,
        atol=1e-12,
    )
