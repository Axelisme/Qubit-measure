"""FTF terminal-only sampling parity against explicit pre-projective inputs."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.core.simulation import SamplingIdentity
from zcu_tools.experiment.ftf import plan_template
from zcu_tools.simulate import adc

from tests.experiment._ftf_templates import settings, snapshot, template

pytestmark = pytest.mark.sim


def test_terminal_only_plan_preserves_raw_statistics_and_decimated_values() -> None:
    base = template()
    window = base.readouts[0]
    window = replace(
        window,
        trigger_us=np.column_stack((window.trigger_us, window.trigger_us + 0.004)),
        frequency_mhz=np.full((2, 2), 6000.0),
        phase_deg=np.full((2, 2), 45.0),
        generator_channels=np.zeros((2, 2), dtype=np.int64),
    )
    planned = (
        plan_template(settings(), snapshot(), replace(base, readouts=(window,)))
        .readouts[0]
        .batch
    )
    # The known terminal-only input has one shared event and two grids, not
    # independent events per trigger. Its pulse phase includes the 1 ns origin.
    baseline = adc.ReadoutBatch(
        planned.model,
        tuple(
            (
                adc.ReadoutEvent(
                    (
                        adc.ReadoutPulse(
                            start, start + 20, 0.2j, 6.0, np.deg2rad(-2130.0)
                        ),
                    ),
                    tuple(
                        adc.SampleGrid(
                            np.arange(start + shift, start + shift + 4), 6.0, np.pi / 4
                        )
                        for shift in (0, 4)
                    ),
                ),
            )
            for start in (8.0, 10.0)
        ),
        np.array([0, 1], dtype=np.int64),
    )
    readout = adc.ProjectionReadout.unprojected(np.array([[0.25, 0.75], [0.6, 0.4]]))
    identity = SamplingIdentity(123, (7, 0, 1))
    results = []
    for batch in (planned, baseline):
        raw = (
            adc.PreparedRawShots(readout, {0: batch}, reps=5)
            .bind(identity)
            .materialize(5, 0)
        )
        conditional = adc.ADC(batch.model).responses(batch)
        statistics = adc.round_statistics(
            readout,
            (
                adc.ChannelReadout(
                    0, conditional, batch.content_index, batch.model.receiver
                ),
            ),
            reps=5,
            seed=123,
            acquire_sequence=7,
            round_index=0,
        )
        traces = adc.sample_traces(readout, {0: batch}, reps=5, identity=identity)
        results.append((raw[0], statistics, traces.channels[0].iq))
    actual, expected = results
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1].state_counts, expected[1].state_counts)
    np.testing.assert_array_equal(
        actual[1].channels[0].sums, expected[1].channels[0].sums
    )
    np.testing.assert_array_equal(actual[2], expected[2])
