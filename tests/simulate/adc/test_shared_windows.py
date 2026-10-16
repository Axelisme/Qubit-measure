"""Multiple ADC windows retain one joint-initial draw per independent shot."""

import numpy as np
from zcu_tools.core.simulation import SamplingIdentity
from zcu_tools.simulate.adc import (
    ADC,
    PopulationInput,
    ReadoutModel,
    ReadoutPulse,
    Receiver,
    SampleGrid,
)


def test_windows_share_initial_state_but_not_receiver_noise() -> None:
    adc = ADC(
        ReadoutModel(
            "shared",
            ("a", "b"),
            np.array([[6.0, 6.1], [6.01, 6.12]]),
            6.0,
            np.array([0.001, 0.001]),
            np.array([0.002, 0.002]),
            np.zeros((2, 2)),
            Receiver(noise_covariance=np.eye(2)),
        )
    )
    grid = SampleGrid(np.arange(8, dtype=float), 6.0)
    windows = adc.sample_windows(
        PopulationInput(("a", "b"), np.array([0.4, 0.6])),
        (ReadoutPulse(0.0, 8.0, 1.0, 6.0),),
        (grid, grid),
        SamplingIdentity(123),
        shots=32,
    )
    np.testing.assert_array_equal(windows[0].initial_states, windows[1].initial_states)
    assert np.any(windows[0].traces != windows[1].traces)
    np.testing.assert_array_equal(windows[0].sums, windows[0].traces.sum(axis=1))
