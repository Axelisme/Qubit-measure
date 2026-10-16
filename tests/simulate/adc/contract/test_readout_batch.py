"""Construction promises for batched ADC readout inputs and conditional means."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate import adc


def model() -> adc.ReadoutModel:
    return adc.ReadoutModel(
        "dc",
        ("g", "e"),
        np.zeros((2, 2)),
        0.0,
        np.zeros(2),
        np.zeros(2),
        np.zeros((2, 2)),
    )


def batch():
    pulse = adc.ReadoutPulse(0.0, 4.0, 1.0, 0.0)
    grid = adc.SampleGrid(np.arange(4.0), 0.0)
    return adc.ReadoutBatch(
        model(), ((pulse,),), ((grid,),), np.array([0, 0], dtype=np.int64)
    )


def conditional():
    return adc.ConditionalSums(np.zeros((1, 1, 2, 2)), np.array([4], dtype=np.int64))


def test_batch_accepts_shared_content_and_optional_traces():
    result = replace(batch(), include_traces=True)
    assert result.content_index.tolist() == [0, 0]
    assert result.include_traces


@pytest.mark.parametrize(
    "change",
    [
        {"pulses": ()},
        {"grids": ()},
        {"grids": ((),)},
        {"pulses": ((),)},
        {"content_index": np.array([-1], dtype=np.int64)},
        {"content_index": np.array([1], dtype=np.int64)},
        {"content_index": np.array([0.5])},
        {"content_index": np.array([[0]], dtype=np.int64)},
    ],
)
def test_batch_rejects_misaligned_or_invalid_content(change):
    valid = batch()
    with pytest.raises(ValueError, match="content|pulse|grid|read"):
        replace(valid, **change)


def test_conditional_accepts_sums_and_aligned_traces():
    result = adc.ConditionalSums(
        np.full((2, 1, 2, 2), 4.0),
        np.array([4], dtype=np.int64),
        np.ones((2, 1, 2, 4, 2)),
    )
    assert result.traces is not None
    np.testing.assert_array_equal(result.traces.sum(axis=-2), result.sums)
    assert conditional().traces is None


@pytest.mark.parametrize(
    "change",
    [
        {"sums": np.zeros((1, 2, 2))},
        {"sums": np.zeros((1, 1, 2, 3))},
        {"sums": np.zeros((0, 1, 2, 2))},
        {"sums": np.full((1, 1, 2, 2), np.nan)},
        {"sample_counts": np.array([0], dtype=np.int64)},
        {"sample_counts": np.array([-2], dtype=np.int64)},
        {"sample_counts": np.array([4.5])},
        {"sample_counts": np.array([4, 4], dtype=np.int64)},
        {"traces": np.zeros((1, 1, 2, 3, 2))},
        {"traces": np.zeros((1, 1, 3, 4, 2))},
        {"traces": np.full((1, 1, 2, 4, 2), np.inf)},
    ],
)
def test_conditional_rejects_invalid_layout_or_values(change):
    valid = conditional()
    with pytest.raises(ValueError, match="sums|sample_counts|traces"):
        replace(valid, **change)


def test_channel_accepts_multidimensional_point_content():
    result = adc.ChannelReadout(
        3, conditional(), np.zeros((2, 3), dtype=np.int64), adc.Receiver()
    )
    assert result.channel == 3
    assert result.content_index.shape == (2, 3)


@pytest.mark.parametrize(
    "change",
    [
        {"channel": -1},
        {"channel": True},
        {"content_index": np.array([1], dtype=np.int64)},
        {"content_index": np.array([-1], dtype=np.int64)},
        {"content_index": np.array([0.5])},
    ],
)
def test_channel_rejects_invalid_index(change):
    valid = adc.ChannelReadout(
        0, conditional(), np.array([0], dtype=np.int64), adc.Receiver()
    )
    with pytest.raises(ValueError, match="channel|content_index"):
        replace(valid, **change)
