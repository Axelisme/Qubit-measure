"""Conditional response reuse and exact constant-pulse physics at the ADC seam."""

from dataclasses import replace

import numpy as np
import pytest
import scipy.linalg
from zcu_tools.simulate import adc


def model() -> adc.ReadoutModel:
    return adc.ReadoutModel(
        "dc",
        ("g", "e", "f"),
        np.array([[6.001, 6.013], [6.004, 6.017], [6.007, 6.021]]),
        6.0,
        np.array([0.03, 0.05]),
        np.array([0.08, 0.06]),
        np.array([[0.0, 0.03, 0.002], [0.01, 0.0, 0.04], [0.001, 0.005, 0.0]]),
        adc.Receiver(gain=2.3, phase_rad=0.31, offset=np.array([0.4, -0.2])),
    )


def batch(m: adc.ReadoutModel, *, pulse=None) -> adc.ReadoutBatch:
    pulse = (
        adc.ReadoutPulse(3.0, 23.0, 1.2 + 0.3j, 6.0, phase_rad=0.2)
        if pulse is None
        else pulse
    )
    grids = (
        adc.SampleGrid(np.array([-4.0, 3.0, 5.1, 12.0, 23.0, 28.0, 43.0]), 6.002, 0.1),
        adc.SampleGrid(np.array([-3.0, 4.0, 6.0, 13.0, 24.0, 29.0, 44.0]), 6.004, 0.3),
    )
    return adc.ReadoutBatch(
        m, ((pulse,),), (grids,), np.array([0, 0], dtype=np.int64), True
    )


@pytest.mark.parametrize("detuned", [False, True])
def test_batched_response_agrees_with_dop853_for_all_reads(detuned, monkeypatch):
    m = model()
    inputs = batch(m)
    if detuned:
        inputs = replace(
            inputs, pulses=((replace(inputs.pulses[0][0], frequency_ghz=6.008),),)
        )
    reference = adc.ConditionalReadout(m, rtol=1e-12, atol=1e-14)
    expected = np.stack(
        [reference.response(inputs.pulses[0], g).traces for g in inputs.grids[0]]
    )
    expm = scipy.linalg.expm
    calls = []

    def observe_expm(matrix):
        calls.append(matrix.shape)
        return expm(matrix)

    monkeypatch.setattr(scipy.linalg, "expm", observe_expm)
    result = adc.ConditionalReadout(m, rtol=1e-12, atol=1e-14).responses(inputs)
    assert result.traces is not None
    np.testing.assert_allclose(result.traces[0], expected, atol=1e-9, rtol=0)
    np.testing.assert_allclose(result.sums[0], expected.sum(axis=-2), atol=1e-9, rtol=0)
    np.testing.assert_array_equal(result.sample_counts, [7, 7])
    assert bool(calls) == (not detuned)


def test_response_cache_reuses_content_across_batches_and_trace_requests():
    evaluations = []

    def shaped(local):
        evaluations.append(local)
        return 1.0 + 0.2 * np.sin(local)

    m = model()
    inputs = batch(m, pulse=adc.ReadoutPulse(3.0, 23.0, 1.0, 6.0, envelope=shaped))
    reader = adc.ConditionalReadout(m)
    sums = reader.responses(replace(inputs, include_traces=False))
    assert sums.traces is None
    count = len(evaluations)
    assert count > 0
    # Equal numeric grids/models and duplicated contents do not re-evaluate physics.
    copied = replace(
        inputs,
        model=replace(m),
        pulses=inputs.pulses * 2,
        grids=tuple(tuple(replace(g) for g in inputs.grids[0]) for _ in range(2)),
    )
    traces = reader.responses(copied)
    assert len(evaluations) == count
    np.testing.assert_array_equal(traces.sums[0], sums.sums[0])
    np.testing.assert_array_equal(traces.sums[1], sums.sums[0])
    assert traces.traces is not None
    np.testing.assert_array_equal(traces.traces.sum(axis=-2), traces.sums)


@pytest.mark.parametrize("change", ["model", "pulse", "grid", "demodulation"])
def test_response_cache_distinguishes_changed_physical_content(change):
    m = model()
    inputs = batch(m)
    reader = adc.ConditionalReadout(m)
    before = reader.responses(inputs)
    if change == "model":
        inputs = replace(
            inputs, model=replace(m, frequencies_ghz=m.frequencies_ghz + 0.007)
        )
    elif change == "pulse":
        inputs = replace(
            inputs, pulses=((replace(inputs.pulses[0][0], amplitude=2.0),),)
        )
    else:
        grid = inputs.grids[0][0]
        grid = (
            replace(grid, times_ns=grid.times_ns + 1.0)
            if change == "grid"
            else replace(grid, phase_rad=0.9)
        )
        inputs = replace(inputs, grids=((grid, inputs.grids[0][1]),))
    after = reader.responses(inputs)
    fresh = adc.ConditionalReadout(inputs.model).responses(inputs)
    assert np.max(np.abs(before.sums - after.sums)) > 1e-3
    np.testing.assert_array_equal(after.sums, fresh.sums)


def test_shaped_response_preserves_envelope_and_solver_failures():
    m = model()
    inputs = batch(
        m, pulse=adc.ReadoutPulse(3.0, 23.0, 1.0, 6.0, envelope=lambda t: np.nan)
    )
    with pytest.raises(ValueError, match="envelope.*finite"):
        adc.ConditionalReadout(m).responses(inputs)


def test_response_rejects_content_dependent_sample_counts_and_ragged_traces():
    m = model()
    inputs = batch(m)
    shorter = replace(inputs.grids[0][0], times_ns=np.array([0.0, 1.0]))
    inconsistent = replace(
        inputs, pulses=inputs.pulses * 2, grids=(inputs.grids[0], (shorter, shorter))
    )
    reader = adc.ConditionalReadout(m)
    with pytest.raises(ValueError, match="sample|grid"):
        reader.responses(inconsistent)
    ragged = replace(inputs, grids=((shorter, inputs.grids[0][1]),))
    with pytest.raises(ValueError, match="sample|grid"):
        reader.responses(ragged)
    sums_only = reader.responses(replace(ragged, include_traces=False))
    np.testing.assert_array_equal(sums_only.sample_counts, [2, 7])
