"""An explicit emitted-time origin distinguishes history from continuation."""

from inspect import signature

import numpy as np
import pytest
from zcu_tools.simulate import adc


def _require_origin():
    assert "initial_time_ns" in signature(adc.cavity_fields).parameters
    assert "initial_time_ns" in signature(adc.cavity_field_average).parameters


def _history(shaped):
    model = adc.ReadoutModel(
        "history",
        ("g",),
        np.array([[6.0, 6.11]]),
        6.03,
        np.array([0.02, 0.03]),
        np.array([0.06, 0.04]),
        np.zeros((1, 1)),
    )
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
    return model, pulses


@pytest.mark.parametrize("shaped", [False, True], ids=["flat", "mixed-shaped"])
def test_mid_pulse_historical_field_continuation_preserves_phase_and_envelope(shaped):
    _require_origin()
    model, pulses = _history(shaped)
    origin = 5.4
    inherited = adc.cavity_fields(model, pulses, np.array([0.0, origin]))[-1, 0]
    times = np.array([6.1, 11.0, 21.3, 30.7, 45.0])
    whole = adc.cavity_fields(model, pulses, times)
    continued = adc.cavity_fields(
        model, pulses, times, initial_field=inherited, initial_time_ns=origin
    )
    np.testing.assert_allclose(continued, whole, rtol=2e-8, atol=1e-10)
    full_average = adc.cavity_field_average(model, pulses, times[0], times[-1])
    continued_average = adc.cavity_field_average(
        model,
        pulses,
        times[0],
        times[-1],
        initial_field=inherited,
        initial_time_ns=origin,
    )
    np.testing.assert_allclose(
        continued_average.field_products,
        full_average.field_products,
        rtol=2e-8,
        atol=1e-10,
    )


@pytest.mark.parametrize("shaped", [False, True], ids=["flat", "mixed-shaped"])
def test_omission_equals_explicit_zero_at_zero_even_for_mid_pulse_queries(shaped):
    _require_origin()
    model, pulses = _history(shaped)
    times = np.array([5.4, 11.0, 30.7, 45.0])
    omitted = adc.cavity_fields(model, pulses, times)
    explicit = adc.cavity_fields(
        model, pulses, times, initial_field=np.zeros(2, complex), initial_time_ns=0.0
    )
    np.testing.assert_array_equal(omitted, explicit)
    assert np.max(np.abs(omitted[0])) > 0
    omitted_average = adc.cavity_field_average(model, pulses, times[0], times[-1])
    explicit_average = adc.cavity_field_average(
        model,
        pulses,
        times[0],
        times[-1],
        initial_field=np.zeros(2, complex),
        initial_time_ns=0.0,
    )
    np.testing.assert_array_equal(
        omitted_average.field_products, explicit_average.field_products
    )
    # A nonzero origin with an omitted field is still an explicit vacuum start.
    restarted = adc.cavity_fields(model, pulses, times, initial_time_ns=times[0])
    np.testing.assert_array_equal(restarted[0], 0)


@pytest.mark.parametrize("origin", [-1.0, np.nan, np.inf, -np.inf])
def test_initial_cavity_time_must_be_finite_and_nonnegative(origin):
    _require_origin()
    model, pulses = _history(False)
    with pytest.raises(ValueError, match="initial time"):
        adc.cavity_fields(model, pulses, np.array([5.0, 10.0]), initial_time_ns=origin)
    with pytest.raises(ValueError, match="initial time"):
        adc.cavity_field_average(model, pulses, 5.0, 10.0, initial_time_ns=origin)


@pytest.mark.parametrize("query,origin", [(-1.0, 0.0), (5.0, 6.0)])
def test_cavity_query_before_origin_names_both_absolute_times(query, origin):
    _require_origin()
    model, pulses = _history(False)
    with pytest.raises(ValueError, match=rf"{query}.*{origin}"):
        adc.cavity_fields(
            model, pulses, np.array([query, 10.0]), initial_time_ns=origin
        )
    with pytest.raises(ValueError, match=rf"{query}.*{origin}"):
        adc.cavity_field_average(model, pulses, query, 10.0, initial_time_ns=origin)
