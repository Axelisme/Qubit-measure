"""Pin unprojected RNG and arithmetic across the read-event migration.

The fixture records the pre-projection public outputs with nonzero rates/noise,
two ADCs, two grids and two points. Retire this pin when ADC deliberately changes
its RNG or numeric-version contract and replaces the migration acceptance.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from zcu_tools.core.simulation import SamplingIdentity
from zcu_tools.simulate import adc


def unprojected_inputs():
    """Small shared-path fixture that exercises state, CTMC and receiver draws."""
    model = adc.ReadoutModel(
        "unprojected-parity",
        ("g", "e"),
        np.array([[6.0], [6.04]]),
        6.0,
        np.array([0.04]),
        np.array([0.16]),
        np.array([[0.0, 0.025], [0.015, 0.0]]),
        mismatch_phase_rad=np.zeros(1),
        receiver=adc.Receiver(
            gain=8, noise_covariance=np.array([[0.04, 0.01], [0.01, 0.09]])
        ),
    )
    grids = (
        adc.SampleGrid(np.array([10.0, 20.0]), 6.0),
        adc.SampleGrid(np.array([30.0, 40.0]), 6.0),
    )
    contents = tuple(
        (adc.ReadoutEvent((adc.ReadoutPulse(0.0, 60.0, amplitude, 6.0),), grids),)
        for amplitude in (1.0, 0.7)
    )
    batches = {
        3: adc.ReadoutBatch(model, contents, np.array([0, 1], dtype=np.int64)),
        1: adc.ReadoutBatch(
            replace(model, receiver=replace(model.receiver, gain=11.0, phase_rad=0.2)),
            contents,
            np.array([0, 1], dtype=np.int64),
        ),
    }
    populations = np.array([[0.2, 0.8], [0.65, 0.35]])
    channels = tuple(
        adc.ChannelReadout(
            channel,
            adc.ADC(batch.model).responses(batch),
            batch.content_index,
            batch.model.receiver,
        )
        for channel, batch in batches.items()
    )
    return adc.ProjectionReadout.unprojected(populations), batches, channels


@pytest.mark.parametrize("mode", ["raw", "traces", "statistics"])
def test_unprojected_multigrid_multiadc_outputs_match_preprojection_stream(mode):
    expected = json.loads(
        Path(__file__).with_name("_unprojected_baseline.json").read_text()
    )
    readout, batches, channels = unprojected_inputs()
    identity = SamplingIdentity(11, (2, 3, 1))
    if mode == "raw":
        actual = (
            adc.PreparedRawShots(readout, batches, reps=3)
            .bind(identity)
            .materialize(3, 1)
        )
        for channel, values in actual.items():
            assert_array_equal(values, expected[mode][str(channel)])
    elif mode == "traces":
        actual = adc.sample_traces(readout, batches, reps=3, identity=identity)
        for channel in actual.channels:
            assert_array_equal(channel.iq, expected[mode][str(channel.channel)])
    else:
        actual = adc.round_statistics(
            readout, channels, reps=3, seed=11, acquire_sequence=2, round_index=3
        )
        assert_array_equal(actual.state_counts, expected[mode]["counts"])
        assert actual.raw_identity == identity
        for channel in actual.channels:
            baseline = expected[mode][str(channel.channel)]
            assert_array_equal(channel.sums, baseline["sums"])
            assert_array_equal(channel.conditional_sums, baseline["conditional_sums"])
            assert_array_equal(channel.shot_covariance, baseline["shot_covariance"])
