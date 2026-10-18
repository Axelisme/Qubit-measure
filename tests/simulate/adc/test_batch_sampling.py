"""Joint batched sampling at the ADC seam, independent of evolution and QICK."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.core.simulation import RoundSignals, RoundStatistics, SamplingIdentity
from zcu_tools.simulate import adc


def legacy_rng_state():
    """Return the global MT19937 tuple rather than a non-legacy RNG dictionary."""
    state = np.random.get_state(legacy=True)
    if isinstance(state, dict):
        raise TypeError("global RNG must expose the legacy MT19937 state")
    return state


def channels(*, covariance=None) -> tuple[adc.ChannelReadout, ...]:
    traces = np.array([[[[[1.0, 2.0], [3.0, 4.0]], [[11.0, 12.0], [13.0, 14.0]]]]])
    traces = np.concatenate((traces, traces + 100), axis=0)
    conditional = adc.ConditionalSums(
        traces.sum(axis=-2), np.array([2], dtype=np.int64), traces
    )
    receiver = adc.Receiver(
        noise_covariance=np.zeros((2, 2)) if covariance is None else covariance
    )
    index = np.array([[0, 1]], dtype=np.int64)
    return tuple(adc.ChannelReadout(ch, conditional, index, receiver) for ch in (3, 1))


def test_decimated_shots_share_states_and_keep_content_and_sample_order():
    result = adc.sample_traces(
        np.array([[[0.25, 0.75], [0.0, 1.0]]]),
        channels(),
        reps=2000,
        seed=71,
        acquire_sequence=2,
        round_index=3,
    )
    first, second = result.channels
    assert first.channel == 3
    assert second.channel == 1
    assert first.iq.shape == (4000, 1, 2, 2)
    np.testing.assert_array_equal(first.iq, second.iq)
    np.testing.assert_array_equal(
        first.iq[2000:, 0, 0], np.tile([111.0, 112.0], (2000, 1))
    )
    np.testing.assert_array_equal(first.iq[:, 0, 1] - first.iq[:, 0, 0], 2)
    assert abs(np.mean(first.iq[:2000, 0, 0, 0] == 11) - 0.75) < 0.035


def test_decimated_noise_is_per_sample_independent_and_has_no_rounding_term():
    inputs = channels(covariance=np.array([[2.0, 0.5], [0.5, 1.0]]))
    result = adc.sample_traces(
        np.array([[[1.0, 0.0], [0.0, 1.0]]]),
        inputs,
        reps=12000,
        seed=72,
        acquire_sequence=2,
        round_index=3,
    )
    residuals = []
    for channel in result.channels:
        expected = np.repeat(
            np.array([[[1.0, 2.0], [3.0, 4.0]], [[111.0, 112.0], [113.0, 114.0]]]),
            12000,
            axis=0,
        )
        residual = channel.iq[:, 0] - expected
        np.testing.assert_allclose(residual.mean(axis=(0, 1)), 0, atol=0.025)
        np.testing.assert_allclose(
            np.cov(residual.reshape(-1, 2).T), [[2.0, 0.5], [0.5, 1.0]], atol=0.035
        )
        assert abs(np.cov(residual[:, 0, 0], residual[:, 1, 0])[0, 1]) < 0.04
        residuals.append(residual)
    assert abs(np.cov(residuals[0].ravel(), residuals[1].ravel())[0, 1]) < 0.025


@pytest.mark.parametrize("sampler", [adc.round_statistics, adc.sample_traces])
def test_round_draws_reproduce_without_mutating_inputs_or_global_rng(sampler):
    populations = np.array([[[0.25, 0.75], [1.0, 0.0]]])
    original = populations.copy()
    inputs = channels()
    global_state = legacy_rng_state()
    kwargs = dict(reps=17, seed=73, acquire_sequence=2, round_index=3)
    first = sampler(populations, inputs, **kwargs)
    second = sampler(populations, inputs, **kwargs)
    if isinstance(first, RoundStatistics) and isinstance(second, RoundStatistics):
        np.testing.assert_array_equal(first.state_counts, second.state_counts)
        np.testing.assert_array_equal(first.channels[0].sums, second.channels[0].sums)
        assert first.raw_identity == SamplingIdentity(73, (2, 3, 1))
    elif isinstance(first, RoundSignals) and isinstance(second, RoundSignals):
        np.testing.assert_array_equal(first.channels[0].iq, second.channels[0].iq)
    else:
        pytest.fail("sampler returned inconsistent result types")
    np.testing.assert_array_equal(populations, original)
    after = legacy_rng_state()
    assert global_state[0] == after[0]
    np.testing.assert_array_equal(global_state[1], after[1])
    assert global_state[2:] == after[2:]


@pytest.mark.parametrize("sampler", [adc.round_statistics, adc.sample_traces])
@pytest.mark.parametrize(
    "invalid",
    [
        "negative",
        "nan",
        "sum",
        "states",
        "points",
        "reps",
        "identity",
        "duplicate",
        "unequal_reads",
    ],
)
def test_round_draws_reject_invalid_joint_inputs(sampler, invalid):
    populations = np.array([[[0.25, 0.75], [1.0, 0.0]]])
    inputs = channels()
    reps, seed = 17, 73
    if invalid == "negative":
        populations[0, 0] = [-1e-12, 1 + 1e-12]
    elif invalid == "nan":
        populations[0, 0, 0] = np.nan
    elif invalid == "sum":
        populations[0, 0] = [0.3, 0.8]
    elif invalid == "states":
        populations = np.ones((1, 2, 1))
    elif invalid == "points":
        populations = populations[0]
    elif invalid == "reps":
        reps = -1
    elif invalid == "identity":
        seed = -1
    elif invalid == "duplicate":
        inputs = (inputs[0], inputs[0])
    else:
        conditional = adc.ConditionalSums(
            np.zeros((2, 2, 2, 2)), np.array([2, 3], dtype=np.int64)
        )
        inputs = (replace(inputs[0], conditional=conditional),)
    with pytest.raises(
        ValueError,
        match="probabilit|population|state|point|reps|identity|channel|sample",
    ):
        sampler(
            populations, inputs, reps=reps, seed=seed, acquire_sequence=2, round_index=3
        )


def test_decimated_requires_traces_and_zero_reps_returns_empty_shots():
    inputs = channels()
    populations = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    result = adc.sample_traces(
        populations, inputs, reps=0, seed=1, acquire_sequence=0, round_index=0
    )
    assert result.channels[0].iq.shape == (0, 1, 2, 2)
    missing = replace(
        inputs[0], conditional=replace(inputs[0].conditional, traces=None)
    )
    with pytest.raises(ValueError, match="traces"):
        adc.sample_traces(
            populations, (missing,), reps=1, seed=1, acquire_sequence=0, round_index=0
        )


@pytest.mark.parametrize("sampler", [adc.round_statistics, adc.sample_traces])
def test_round_draws_accept_engine_trace_tolerance_without_mutation(sampler):
    def rng():
        return np.random.Generator(
            np.random.PCG64(np.random.SeedSequence((1, 0, 0, 0)))
        )

    # Put the first uniform between the raw and normalized CDF boundaries.
    boundary = rng().random() * (1 + 4e-7)
    populations = np.array([[[boundary, 1 + 9e-7 - boundary], [0.4, 0.6]]])
    original = populations.copy()
    reps = 1_000_000_000 if sampler is adc.round_statistics else 3
    try:
        result = sampler(
            populations,
            channels(),
            reps=reps,
            seed=1,
            acquire_sequence=0,
            round_index=0,
        )
    except ValueError as error:
        pytest.fail(f"in-budget engine trace residual must be accepted: {error}")
    if isinstance(result, RoundStatistics):
        expected = rng().multinomial(reps, populations)
        repaired = rng().multinomial(
            reps, populations / populations.sum(axis=-1, keepdims=True)
        )
        np.testing.assert_array_equal(result.state_counts, expected)
        assert not np.array_equal(expected, repaired)
    else:
        np.testing.assert_array_equal(result.channels[0].iq[0, 0, 0], [1, 2])
    np.testing.assert_array_equal(populations, original)


def test_prepared_statistics_own_inputs_and_factor_once_but_draw_fresh(monkeypatch):
    assert hasattr(adc, "PreparedStatistics"), (
        "ADC must separate invariant preparation from draws"
    )
    populations = np.array([[[0.25, 0.75], [1.0, 0.0]]])
    inputs = channels(covariance=np.array([[2.0, 0.5], [0.5, 1.0]]))
    kwargs = dict(seed=73, acquire_sequence=2, round_index=3)
    expected = adc.round_statistics(populations, inputs, reps=17, **kwargs)
    factor = np.linalg.eigh
    calls = []

    def observed(covariance):
        calls.append(1)
        return factor(covariance)

    monkeypatch.setattr(np.linalg, "eigh", observed)
    prepared = adc.PreparedStatistics(populations, inputs, reps=17)
    assert len(calls) == len(inputs)
    populations[:] = 0  # Plan must not borrow the mutable input probabilities.
    first = prepared.draw(**kwargs)
    second = prepared.draw(**kwargs)
    assert len(calls) == len(inputs), "round draws must not refactor covariance"
    assert first.state_counts.tobytes() == expected.state_counts.tobytes()
    for actual, repeated, original in zip(
        first.channels, second.channels, expected.channels, strict=True
    ):
        assert (
            actual.sums.tobytes() == original.sums.tobytes() == repeated.sums.tobytes()
        )
    assert first.raw_identity == expected.raw_identity
    assert not prepared.populations.flags.writeable
    with pytest.raises(ValueError, match="identity"):
        prepared.draw(seed=-1, acquire_sequence=2, round_index=3)
