"""Continuous cavity moments and endpoint fields through the ADC span seam."""

import numpy as np
from numpy.polynomial.legendre import leggauss
from zcu_tools.simulate import adc


def test_span_integrates_interfering_tones_and_ringdown_across_fixed_branches():
    model = adc.ReadoutModel(
        "shared-line",
        ("dark", "coalesced", "g", "e"),
        np.array([[6, 6], [6.01, 6.01 + 0.02 / np.pi], [6.003, 6.096], [6.11, 6.053]]),
        6.03,
        np.zeros(2),
        np.full(2, 0.04),
        np.zeros((4, 4)),
    )
    pulses = (
        adc.ReadoutPulse(0, 18, 0.3j, 6.0, 0.2),
        adc.ReadoutPulse(6, 14, 0.2, 6.07, -0.4),
    )
    initial = np.array([0.2 + 0.3j, -0.1j])
    start, end = 4.0, 48.0

    result = adc.CavityAverageCache().span(
        model, pulses, start, end, initial_field=initial, initial_time_ns=start
    )

    # Integrate the public first-moment solution independently, splitting at
    # each switch so the quadrature sees only smooth fields on each interval.
    boundaries = np.array([start, 6, 14, 18, end])
    nodes, weights = leggauss(24)
    half = np.diff(boundaries) / 2
    mid = boundaries[:-1] + half
    times = (mid[:, None] + half[:, None] * nodes).ravel()
    fields = adc.cavity_fields(
        model,
        pulses,
        np.append(times, end),
        initial_field=initial,
        initial_time_ns=start,
    )
    mean_weights = (half[:, None] * weights).ravel() / (end - start)
    products = np.einsum(
        "t,tsr,tuq->sruq", mean_weights, fields[:-1], fields[:-1].conj()
    )
    photons = np.einsum("t,tsr->sr", mean_weights, abs(fields[:-1]) ** 2)
    np.testing.assert_allclose(
        result.average.field_products, products, atol=2e-12, rtol=1e-10
    )
    np.testing.assert_allclose(result.average.photons, photons, atol=2e-12, rtol=1e-10)
    np.testing.assert_allclose(result.final_fields, fields[-1], atol=2e-12, rtol=1e-10)


def test_long_idle_span_preserves_undamped_dark_fields_and_cross_branch_phases():
    frequencies = np.array([6.0, 6.0, 6.011, 6.016])
    model = adc.ReadoutModel(
        "dark-line",
        ("a", "b", "c", "d"),
        np.repeat(frequencies[:, None], 2, axis=1),
        6.0,
        np.zeros(2),
        np.full(2, 0.04),
        np.zeros((4, 4)),
    )
    end = 125_017.0
    pulses = (adc.ReadoutPulse(end, end + 10, 0.3, 6.0),)
    initial = np.array([0.3j, -0.3j])

    result = adc.CavityAverageCache().span(
        model, pulses, 0.0, end, initial_field=initial
    )

    # Equal resonators leave the antisymmetric field undamped. Only its DC
    # branch phase changes, including between branches at identical frequency.
    phases = -2j * np.pi * (frequencies - model.reference_frequency_ghz) * end
    differences = phases[:, None] - phases[None, :]
    mean_phase = np.ones_like(differences)
    np.divide(
        np.expm1(differences), differences, out=mean_phase, where=differences != 0
    )
    products = np.einsum("su,r,q->sruq", mean_phase, initial, initial.conj())
    np.testing.assert_allclose(
        result.average.field_products, products, atol=3e-12, rtol=1e-9
    )
    np.testing.assert_allclose(
        result.final_fields, np.exp(phases[:, None]) * initial, atol=3e-12, rtol=1e-9
    )
