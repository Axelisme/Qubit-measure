"""Byte-bounded physical solver and exact-content reuse at the ADC seam."""

import gc
import weakref
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate import adc


def model():
    return adc.ReadoutModel(
        "same-identity",
        ("g", "e"),
        np.array([[6.0, 6.1], [6.01, 6.11]]),
        6.0,
        np.array([0.02, 0.03]),
        np.array([0.04, 0.05]),
        np.array([[0.0, 0.001], [0.0, 0.0]]),
    )


def batch(*, start=3.0, traces=False, envelope=None):
    return adc.ReadoutBatch(
        model(),
        (
            (
                adc.ReadoutPulse(
                    start, start + 10, 0.2, 6.003, phase_rad=0.4, envelope=envelope
                ),
            ),
        ),
        ((adc.SampleGrid(start + np.arange(12, dtype=float), 6.007, 0.2),),),
        np.array([0], dtype=np.int64),
        traces,
    )


def cache(budget=8 * 1024**2):
    cache_type = getattr(adc, "ReadoutCache", None)
    assert cache_type is not None, "ADC must provide byte-bounded solver/content reuse"
    return cache_type(budget)


@pytest.mark.parametrize("change", ["model", "receiver", "rtol", "atol"])
def test_numeric_model_receiver_and_policy_are_exact_keys(change):
    holder = cache()
    inputs = batch()
    first = holder.responses(inputs)
    copied = replace(inputs, model=replace(inputs.model))
    second = holder.responses(copied)
    assert holder.builds == 1
    assert holder.hits == 1
    assert first.sums.tobytes() == second.sums.tobytes()
    kwargs = {}
    if change == "model":
        copied = replace(
            copied,
            model=replace(
                copied.model, frequencies_ghz=copied.model.frequencies_ghz + 0.001
            ),
        )
    elif change == "receiver":
        copied = replace(
            copied, model=replace(copied.model, receiver=adc.Receiver(gain=2.0))
        )
    else:
        kwargs[change] = 1e-9
    result = holder.responses(copied, **kwargs)
    assert holder.builds == 2
    expected = adc.ConditionalReadout(copied.model, **kwargs).responses(copied)
    assert result.sums.tobytes() == expected.sums.tobytes()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ripple_depth", 0.2),
        ("ripple_period_ghz", 0.04),
        ("ripple_phase_rad", 0.3),
        ("slope_db_per_ghz", -3.0),
        ("pivot_ghz", 7.0),
    ],
)
def test_background_parameters_are_exact_receiver_cache_keys(field, value):
    holder = adc.ReadoutCache()
    inputs = batch()
    holder.responses(inputs)
    changed = replace(
        inputs,
        model=replace(
            inputs.model, receiver=replace(inputs.model.receiver, **{field: value})
        ),
    )
    actual = holder.responses(changed)
    assert holder.builds == 2
    expected = adc.ConditionalReadout(changed.model).responses(changed)
    np.testing.assert_array_equal(actual.sums, expected.sums)
    cavity = adc.CavityAverageCache()
    before = cavity.average(inputs.model, inputs.pulses[0], 3.0, 15.0)
    after = cavity.average(changed.model, changed.pulses[0], 3.0, 15.0)
    assert cavity.misses == 2
    # Receiver backgrounds cannot alter photon occupations or Stark inputs.
    np.testing.assert_array_equal(after.field_products, before.field_products)


def test_absolute_timing_and_trace_requests_keep_their_original_numeric_paths():
    holder = cache()
    for inputs in (
        batch(),
        batch(start=13.3),
        batch(traces=True),
        batch(start=13.3, traces=True),
    ):
        actual = holder.responses(inputs)
        expected = adc.ConditionalReadout(inputs.model).responses(inputs)
        assert actual.sums.tobytes() == expected.sums.tobytes()
        if inputs.include_traces:
            assert actual.traces is not None and expected.traces is not None
            assert actual.traces.tobytes() == expected.traces.tobytes()
    assert holder.builds == 1
    assert holder.hits == 3


def test_byte_limit_evicts_growing_traces_and_line_facts():
    probe = cache()
    probe.responses(batch())
    limit = probe.resident_bytes + 100
    holder = cache(limit)
    holder.responses(batch())
    assert 0 < holder.resident_bytes <= limit
    large = replace(
        batch(traces=True),
        grids=((adc.SampleGrid(np.arange(2000, dtype=float), 6.0),),),
    )
    holder.responses(large)
    assert holder.resident_bytes <= limit
    builds = holder.builds
    holder.responses(batch())
    assert holder.builds == builds + 1, "oversized solver must not remain resident"
    holder.clear()
    assert holder.resident_bytes == 0


def test_callback_view_charges_its_entire_backing_allocation():
    class Envelope:
        __slots__ = ("payload",)

        def __init__(self):
            self.payload = np.zeros(128 * 1024, dtype=float)[:: 128 * 1024]

        def __call__(self, time_ns):
            return 1.0

    holder = cache(128 * 1024)
    holder.responses(batch(envelope=Envelope()))
    assert holder.resident_bytes == 0, "a tiny view must not hide its 1 MiB owner"
    holder.responses(batch())
    assert holder.builds == 2


def test_opaque_callbacks_stay_alive_and_are_isolated_by_identity():
    class Envelope:
        def __init__(self, value):
            self.value = value

        def __call__(self, time_ns):
            return self.value

    holder = cache()
    envelope = Envelope(1.0)
    reference = weakref.ref(envelope)
    first = holder.responses(batch(envelope=envelope))
    del envelope
    assert reference() is not None
    other = batch(envelope=Envelope(0.5))
    second = holder.responses(other)
    assert first.sums.tobytes() != second.sums.tobytes()
    holder.clear()
    # SciPy solve_ivp closure cycles can temporarily retain the callback.
    gc.collect()
    assert reference() is None
