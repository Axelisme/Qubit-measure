"""Multiple ADC windows retain one joint-initial draw per independent shot."""

import numpy as np
from zcu_tools.simulate.adc import ADC, PopulationInput, ReadoutModel, ReadoutPulse, Receiver, SampleGrid, SamplingIdentity


def test_windows_share_initial_state_but_not_receiver_noise() -> None:
    adc = ADC(ReadoutModel("shared", ("a", "b"), np.array([[6., 6.1], [6.01, 6.12]]), 6., np.array([.001, .001]), np.array([.002, .002]), np.zeros((2, 2)), Receiver(noise_covariance=np.eye(2))))
    grid = SampleGrid(np.arange(8, dtype=float), 6.)
    windows = adc.sample_windows(PopulationInput(("a", "b"), np.array([.4, .6])), (ReadoutPulse(0., 8., 1., 6.),), (grid, grid), SamplingIdentity(123), shots=32)
    np.testing.assert_array_equal(windows[0].initial_states, windows[1].initial_states)
    assert np.any(windows[0].traces != windows[1].traces)
    np.testing.assert_array_equal(windows[0].sums, windows[0].traces.sum(axis=1))
