"""Independent shared-line readout references, not QICK buffer tests."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm


def readout(
    *,
    frequencies=None,
    rates=None,
    gain=1.0,
    phase=0.0,
    offset=(0.0, 0.0),
    covariance=None,
):
    from zcu_tools.simulate.adc import ReadoutModel, Receiver

    return ReadoutModel(
        identity="synthetic-readout-v1",
        state_labels=("g", "e", "f"),
        frequencies_ghz=np.zeros((3, 2)) if frequencies is None else frequencies,
        reference_frequency_ghz=6.0,
        kappa_internal_per_ns=np.array([0.03, 0.05]),
        kappa_external_per_ns=np.array([0.08, 0.06]),
        rates_per_ns=np.zeros((3, 3)) if rates is None else rates,
        receiver=Receiver(
            gain,
            phase,
            np.array(offset),
            np.zeros((2, 2)) if covariance is None else covariance,
        ),
    )


def signal_model(**kwargs):
    # Every state has the same absolute frequencies unless explicitly overridden.
    kwargs.setdefault("frequencies", np.tile([6.001, 6.013], (3, 1)))
    return readout(**kwargs)


def pulse_and_grid():
    from zcu_tools.simulate.adc import ReadoutPulse, SampleGrid

    return (ReadoutPulse(0, 20, 1.2 + 0.3j, 6.0),), SampleGrid(
        np.array([-4, 0, 4, 12, 20, 28, 40.0]), 6.0
    )


def test_shared_line_finite_pulse_ringdown_demodulation_and_receiver():
    from zcu_tools.simulate.adc import ADC

    phase, gain = 0.31, 2.3
    m = signal_model(gain=gain, phase=phase, offset=(0.4, -0.2))
    pulses, grid = pulse_and_grid()
    response = ADC(m).response(pulses, grid)
    v = np.sqrt(np.array([0.08, 0.06]) / 2)
    a = np.diag(
        np.array([0.03, 0.05]) / 2 + 2j * np.pi * np.array([0.001, 0.013])
    ) + np.outer(v, v)
    amplitude = 1.2 + 0.3j
    expected = []
    for t in grid.times_ns:
        on = max(0.0, min(float(t), 20.0))
        field = np.linalg.solve(a, (np.eye(2) - expm(-a * on)) @ (v * amplitude))
        if t > 20:
            field = expm(-a * (t - 20)) @ field
        outgoing = (amplitude if 0 <= t < 20 else 0) - v @ field
        value = gain * np.exp(1j * phase) * outgoing
        expected.append([value.real + 0.4, value.imag - 0.2])
    assert_allclose(response.traces[0], expected, atol=2e-8)
    assert_allclose(response.traces[1], expected, atol=2e-8)
    assert np.linalg.norm(response.traces[0, -1] - [0.4, -0.2]) > 0.01


def test_directional_multilevel_cascade_and_finite_temperature():
    from zcu_tools.simulate.adc import ADC, ReadoutPulse, SampleGrid

    rate = 0.04
    rates = np.array([[0, rate, 0], [0, 0, rate], [0, 0, 0]])
    grid = SampleGrid(np.array([40.0]), 6.0)
    pulses = (ReadoutPulse(0, 40, 1.0, 6.0),)
    response = ADC(signal_model(rates=rates)).response(pulses, grid)
    decay = np.exp(-rate * 40)
    assert_allclose(
        response.final_populations[2],
        [1 - decay * (1 + rate * 40), rate * 40 * decay, decay],
        atol=2e-9,
    )
    up, down = 0.01, 0.03
    thermal = np.array([[0, down, 0], [up, 0, 0], [0, 0, 0]])
    result = ADC(signal_model(rates=thermal)).response(pulses, grid)
    equilibrium = up / (up + down)
    excited = equilibrium * (1 - np.exp(-(up + down) * 40))
    assert_allclose(result.final_populations[0], [1 - excited, excited, 0], atol=2e-9)


def test_state_conditioned_moments_match_independent_block_exponential():
    from zcu_tools.simulate.adc import ADC, ReadoutPulse, SampleGrid

    rates = np.array([[0, 0.06, 0.008], [0.01, 0, 0.02], [0, 0.004, 0]])
    frequencies = np.array([[6.0, 6.02], [6.03, 5.99], [5.98, 6.04]])
    m = readout(frequencies=frequencies, rates=rates)
    grid = SampleGrid(np.array([6.0, 18.0, 34.0]), 6.0)
    pulses = (ReadoutPulse(0, 40, 1.0, 6.0),)
    actual = ADC(m).response(pulses, grid)
    # Independent augmented linear generator: [p_s, m_s,r] in state-major order.
    q = rates - np.diag(rates.sum(axis=0))
    v = np.sqrt(np.array([0.08, 0.06]) / 2)
    generator = np.zeros((9, 9), complex)
    generator[:3, :3] = q
    generator[3:, 3:] = np.kron(q, np.eye(2))
    for s in range(3):
        block = slice(3 + 2 * s, 5 + 2 * s)
        a = np.diag(
            np.array([0.03, 0.05]) / 2 + 2j * np.pi * (frequencies[s] - 6.0)
        ) + np.outer(v, v)
        generator[block, block] -= a
        generator[block, s] = v
    initial = np.zeros(9, complex)
    initial[1] = 1
    expected = []
    for t in grid.times_ns:
        evolved = expm(generator * t) @ initial
        out = 1 - evolved[3:].reshape(3, 2).sum(axis=0) @ v
        expected.append([out.real, out.imag])
    assert_allclose(actual.traces[1], expected, atol=3e-8)
    # Wrong mean-frequency substitution has a visibly different response.
    mean_frequency = np.average(frequencies, axis=0, weights=[0.4, 0.5, 0.1])
    wrong = ADC(readout(frequencies=np.tile(mean_frequency, (3, 1)))).response(
        pulses, grid
    )
    assert np.linalg.norm(actual.traces[1] - wrong.traces[1]) > 0.01


def test_shot_sampling_noise_reproducibility_and_sums_share_one_realization():
    from zcu_tools.core.simulation import SamplingIdentity
    from zcu_tools.simulate.adc import ADC, PopulationInput

    covariance = np.array([[0.16, 0.06], [0.06, 0.09]])
    m = signal_model(covariance=covariance)
    pulses, grid = pulse_and_grid()
    populations = PopulationInput(m.state_labels, np.array([0.2, 0.3, 0.5]))
    identity = SamplingIdentity(789, (4, 2, 8))
    adc = ADC(m)
    first = adc.sample(populations, pulses, grid, identity, shots=40000)
    second = adc.sample(populations, pulses, grid, identity, shots=40000)
    assert first.traces.dtype == np.float64
    assert first.sums.dtype == np.float64
    assert_allclose(first.traces, second.traces, atol=0, rtol=0)
    assert_allclose(first.sums, first.traces.sum(axis=1), atol=0, rtol=0)
    assert_allclose(
        np.bincount(first.initial_states, minlength=3) / 40000,
        populations.values,
        atol=0.015,
    )
    deterministic = adc.response(pulses, grid).traces[0, 0]
    residual = first.traces[:, 0] - deterministic
    assert_allclose(np.cov(residual.T), covariance, atol=0.025)
    changed = adc.sample(
        populations, pulses, grid, SamplingIdentity(789, (4, 2, 9)), shots=8
    )
    assert not np.array_equal(first.traces[:8], changed.traces)


@pytest.mark.parametrize(
    "covariance", [np.zeros((2, 2)), np.array([[0.04, 0.04], [0.04, 0.04]])]
)
def test_zero_gain_zero_or_singular_psd_noise_and_state_order(covariance):
    from zcu_tools.core.simulation import SamplingIdentity
    from zcu_tools.simulate.adc import ADC, PopulationInput

    m = signal_model(gain=0, offset=(0.7, -0.3), covariance=covariance)
    pulses, grid = pulse_and_grid()
    population = PopulationInput(m.state_labels, np.array([1.0, 0, 0]))
    result = ADC(m).sample(population, pulses, grid, SamplingIdentity(12), shots=30)
    residual = result.traces - np.array([0.7, -0.3])
    assert_allclose(residual[..., 0], residual[..., 1], atol=2e-9)
    wrong_order = PopulationInput(("e", "g", "f"), np.array([1.0, 0, 0]))
    with pytest.raises(ValueError, match="state order"):
        ADC(m).sample(wrong_order, pulses, grid, SamplingIdentity(12), shots=1)


def test_demodulation_carrier_and_finite_complex_envelope():
    # Zero external coupling isolates the direct transmitted incident field.
    from dataclasses import replace

    from zcu_tools.simulate.adc import ADC, ReadoutPulse, SampleGrid

    m = replace(signal_model(), kappa_external_per_ns=np.zeros(2))
    pulse = ReadoutPulse(0, 20, 1 + 2j, 6.013, envelope=lambda t: 1 - t / 20)
    times = np.array([-2, 2, 6, 10, 24.0])
    grid = SampleGrid(times, 6.004, phase_rad=0.2)
    actual = ADC(m).response((pulse,), grid).traces[0]
    values = np.asarray(
        np.where(
            (times >= 0) & (times < 20),
            (1 + 2j) * (1 - times / 20) * np.exp(-2j * np.pi * 0.009 * times - 0.2j),
            0,
        ),
        dtype=np.complex128,
    )
    assert_allclose(actual, np.column_stack((values.real, values.imag)), atol=2e-8)


def test_lookback_does_not_move_readout_population_origin():
    from zcu_tools.simulate.adc import ADC, ReadoutPulse, SampleGrid

    rates = np.array([[0, 0.04, 0], [0, 0, 0], [0, 0, 0]])
    frequencies = np.array([[6.0, 6.02], [6.03, 5.99], [5.98, 6.04]])
    adc = ADC(readout(frequencies=frequencies, rates=rates, offset=(0.3, -0.2)))
    pulses = (ReadoutPulse(10, 40, 1.0, 6.0),)
    after = SampleGrid(np.array([14.0, 22, 34]), 6.0)
    lookback = SampleGrid(np.array([-10.0, -2, 6, 14, 22, 34]), 6.0)
    response = adc.response(pulses, lookback)
    no_lookback = adc.response(pulses, after)
    excited = np.exp(-0.04 * (34 - 10))
    assert_allclose(response.final_populations[1], [1 - excited, excited, 0], atol=2e-9)
    assert_allclose(response.traces[:, 3:], no_lookback.traces, atol=2e-8)
    assert_allclose(response.traces[:, :3], np.tile([0.3, -0.2], (3, 3, 1)), atol=2e-9)


@pytest.mark.parametrize("scale", [1e-12, 1e-100])
@pytest.mark.parametrize(
    "covariance", [np.diag([-0.5, 0.0]), np.array([[1.0, 2.0], [2.0, 1.0]])]
)
def test_receiver_rejects_non_psd_covariance_at_its_own_scale(scale, covariance):
    from zcu_tools.simulate.adc import Receiver

    with pytest.raises(ValueError, match="positive semidefinite"):
        Receiver(noise_covariance=scale * covariance)


def test_invalid_rates_covariance_and_population_are_rejected():
    from zcu_tools.simulate.adc import PopulationInput, Receiver

    with pytest.raises(ValueError, match="positive semidefinite"):
        Receiver(noise_covariance=np.array([[1, 2], [2, 1.0]]))
    with pytest.raises(ValueError, match="nonnegative"):
        signal_model(rates=np.array([[0, -0.1, 0], [0, 0, 0], [0, 0, 0]]))
    with pytest.raises(ValueError, match="sum"):
        PopulationInput(("g", "e"), np.array([0.2, 0.1]))
