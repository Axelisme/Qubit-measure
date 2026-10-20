"""Deferred ADC shots: physical paths, independent means and QICK layouts."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from zcu_tools.core.simulation import SamplingIdentity
from zcu_tools.simulate import adc


def model(rates=None, *, receiver=None):
    return adc.ReadoutModel(
        "integrated-paths",
        ("g", "e", "f"),
        np.array([[6.0, 6.1], [6.04, 6.1], [5.96, 6.1]]),
        6.0,
        np.array([0.04, 0.02]),
        np.array([0.16, 0.0]),
        np.zeros((3, 3)) if rates is None else rates,
        receiver=adc.Receiver(gain=100) if receiver is None else receiver,
    )


def batch(readout_model, *, shaped=False):
    envelope = (
        adc.PiecewiseLinearEnvelope(
            np.array([0, 15, 40.0]), np.array([0, 1 + 0.2j, 0.4], complex)
        )
        if shaped
        else None
    )
    pulse = adc.ReadoutPulse(5, 65, 0.8 + 0.2j, 6.007, 0.2, envelope)
    grids = (
        adc.SampleGrid(np.arange(-5, 86, 2, dtype=float), 6.003, 0.1),
        adc.SampleGrid(np.array([-5, 5, 10, 25, 45, 65, 85.0]), 6.003, 0.1),
    )
    return adc.ReadoutBatch(
        readout_model, ((pulse,),), (grids,), np.array([0], dtype=np.int64)
    )


def prepare(populations, batches, *, reps):
    # Assert the new public capability at the existing ADC package seam, so a
    # missing implementation is a behaviour failure rather than an import error.
    factory = getattr(adc, "PreparedRawShots", None)
    assert factory is not None, "ADC must prepare deferred physical raw shots"
    return factory(populations, batches, reps=reps)


@pytest.mark.parametrize("shaped", [False, True])
@pytest.mark.parametrize("avg_level", [0, 1, 2])
def test_zero_jump_raw_preserves_receivers_windows_and_loop_layout(shaped, avg_level):
    receiver = adc.Receiver(
        gain=17, phase_rad=0.3, offset=np.array([0.2, -0.4]), time_of_flight_ns=2.5
    )
    readout_model = replace(
        model(receiver=receiver), mismatch_phase_rad=np.array([0.6, 0.0])
    )
    content = batch(readout_model, shaped=shaped)
    content = replace(content, content_index=np.zeros(6, dtype=np.int64))
    populations = np.zeros((2, 3, 3))
    populations[..., 1] = 1
    prepared = prepare(populations, {3: content}, reps=7)
    source = prepared.bind(SamplingIdentity(17, (2, 3, 1)))
    raw = source.materialize(7, avg_level)[3]
    expected = adc.ADC(readout_model).responses(content).sums[0, :, 1]
    flat = np.moveaxis(raw, avg_level, 2)
    assert raw.dtype == np.int64
    assert flat.shape == (2, 3, 7, 2, 2)
    assert_array_equal(flat, np.broadcast_to(np.rint(expected), flat.shape))
    raw[...] = -999
    assert_array_equal(
        source.materialize(7, avg_level)[3],
        np.moveaxis(np.broadcast_to(np.rint(expected), flat.shape), 2, avg_level),
    )


@pytest.mark.parametrize(
    "rates",
    [
        np.array([[0, 0.025, 0], [0, 0, 0], [0, 0, 0]]),
        np.array([[0, 0.05, 0], [0, 0, 0.05], [0, 0, 0]]),
        np.array([[0, 0.2, 0.1], [0.1, 0, 0.2], [0.2, 0.1, 0]]),
    ],
    ids=["single-jump", "cascade", "many-jumps"],
)
@pytest.mark.parametrize("shaped", [False, True])
def test_integrated_shots_form_bridge_and_match_conditional_expectation(rates, shaped):
    readout_model = model(rates)
    content = batch(readout_model, shaped=shaped)
    populations = (
        np.array([0.0, 1.0, 0.0])
        if not rates[:, 2].any()
        else np.array([0.0, 0.0, 1.0])
    )
    source = prepare(populations, {0: content}, reps=1600).bind(SamplingIdentity(52))
    raw = source.materialize(1600, 0)[0]
    expected = (
        adc.ADC(readout_model).responses(content).sums[0, :, populations.argmax()]
    )
    error = np.abs(raw.mean(axis=0) - expected)
    stderr = raw.std(axis=0) / np.sqrt(len(raw))
    assert np.all(error <= 6 * stderr + 0.51)
    # A pure initial state used to produce one conditional centre. Physical
    # paths have a bridge, including repeated transitions beyond two jumps.
    assert np.unique(raw[:, 0], axis=0).shape[0] > 150
    assert raw[:, 0].var(axis=0).sum() > 10000
    assert_array_equal(source.materialize(1600, 0)[0], raw)


def test_deferred_raw_does_not_change_statistics_stream_or_retain_noisy_results():
    content = batch(model(np.array([[0, 0.025, 0], [0, 0, 0.04], [0, 0, 0]])))
    content = replace(content, grids=((content.grids[0][0], content.grids[0][0]),))
    conditional = adc.ADC(content.model).responses(content)
    channel = adc.ChannelReadout(
        0, conditional, np.array(0, dtype=np.int64), content.model.receiver
    )
    probabilities = np.array([0.1, 0.2, 0.7])
    sampler = adc.PreparedStatistics(probabilities, (channel,), reps=13)
    prepared = prepare(probabilities, {0: content}, reps=13)
    arguments = dict(seed=12, acquire_sequence=2, round_index=3)
    before = sampler.draw(**arguments)
    first = prepared.bind(before.raw_identity).materialize(13, 0)[0]
    sampler.draw(seed=12, acquire_sequence=2, round_index=4)
    after = sampler.draw(**arguments)
    assert_array_equal(before.state_counts, after.state_counts)
    assert_array_equal(before.channels[0].sums, after.channels[0].sums)
    assert_array_equal(prepared.bind(after.raw_identity).materialize(13, 0)[0], first)
    assert not np.array_equal(
        prepared.bind(SamplingIdentity(13)).materialize(13, 0)[0], first
    )


@pytest.mark.parametrize(
    "reps,avg_level", [(6, 0), (-1, 0), (7, -1), (7, 1), (True, 0)]
)
def test_raw_rejects_reps_or_layout_mismatch(reps, avg_level):
    source = prepare(np.array([1.0, 0, 0]), {0: batch(model())}, reps=7).bind(
        SamplingIdentity(3)
    )
    with pytest.raises(ValueError, match="reps|avg_level|layout"):
        source.materialize(reps, avg_level)


def test_zero_reps_is_empty_and_receiver_overflow_fails():
    content = batch(model())
    source = prepare(np.array([1.0, 0, 0]), {0: content}, reps=0).bind(
        SamplingIdentity(3)
    )
    assert source.materialize(0, 0)[0].shape == (0, 2, 2)
    overflow = replace(
        content,
        model=replace(
            content.model, receiver=adc.Receiver(offset=np.array([2.0**63, 0]))
        ),
    )
    source = prepare(np.array([1.0, 0, 0]), {0: overflow}, reps=1).bind(
        SamplingIdentity(3)
    )
    with pytest.raises(ValueError, match="int64"):
        source.materialize(1, 0)


def test_state_independent_coalescent_cavities_keep_exact_sum_despite_many_jumps():
    detuning = 0.04 / (2 * np.pi)
    readout_model = replace(
        model(np.array([[0, 0.2, 0.1], [0.1, 0, 0.2], [0.2, 0.1, 0]])),
        frequencies_ghz=np.tile([6 - detuning, 6 + detuning], (3, 1)),
        kappa_internal_per_ns=np.array([0.02, 0.02]),
        kappa_external_per_ns=np.array([0.08, 0.08]),
    )
    content = batch(readout_model)
    raw = (
        prepare(np.array([0.2, 0.3, 0.5]), {0: content}, reps=30)
        .bind(SamplingIdentity(2))
        .materialize(30, 0)[0]
    )
    expected = (
        adc.ADC(replace(readout_model, rates_per_ns=np.zeros((3, 3))))
        .responses(content)
        .sums[0, :, 0]
    )
    assert_allclose(raw, np.broadcast_to(np.rint(expected), raw.shape), atol=0)
