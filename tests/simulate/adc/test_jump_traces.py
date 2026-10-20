"""Single-carrier shot paths observed through the public ADC trace seam."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import minimize_scalar
from zcu_tools.core.simulation import SamplingIdentity
from zcu_tools.simulate.adc import (
    ADC,
    PiecewiseLinearEnvelope,
    PopulationInput,
    ReadoutModel,
    ReadoutPulse,
    Receiver,
    SampleGrid,
)

pytestmark = pytest.mark.sim


def model(rates: np.ndarray) -> ReadoutModel:
    """One coupled cavity and one dark cavity, with resolvable state shifts."""
    return ReadoutModel(
        "jump-traces",
        ("g", "e", "f"),
        np.array([[6.0, 6.1], [6.04, 6.1], [5.96, 6.1]]),
        6.0,
        np.array([0.04, 0.02]),
        np.array([0.16, 0.0]),
        rates,
    )


@pytest.mark.parametrize(
    "rates",
    [
        np.zeros((3, 3)),
        np.array([[0, 0.025, 0], [0, 0, 0.025], [0, 0, 0]]),
        np.array([[0, 0.03, 0.01], [0.02, 0, 0.04], [0, 0.01, 0]]),
    ],
    ids=["no-jumps", "cascade", "thermal-and-direct"],
)
def test_shot_path_mean_and_variance(rates: np.ndarray) -> None:
    adc = ADC(model(rates))
    pulse = ReadoutPulse(10, 85, 1.0, 6.0)
    grid = SampleGrid(np.array([0, 10, 20, 40, 65, 85, 100.0]), 6.0)
    population = PopulationInput(adc.model.state_labels, np.array([0, 0, 1.0]))
    conditional = adc.response((pulse,), grid)
    result = adc.sample(population, (pulse,), grid, SamplingIdentity(341), shots=2400)
    # Six standard errors bound all coordinates, including ringdown. Unlike
    # comparing only means, the variance rejects deterministic conditional shots.
    error = np.abs(result.traces.mean(axis=0) - conditional.traces[2])
    stderr = result.traces.std(axis=0) / np.sqrt(len(result.traces))
    assert np.all(error <= 6 * stderr + 2e-8)
    if np.any(rates):
        assert result.traces[:, 4].var(axis=0).sum() > 0.01
    else:
        assert_allclose(
            result.traces,
            np.broadcast_to(conditional.traces[2], result.traces.shape),
            atol=2e-8,
        )
    assert_array_equal(result.sums, result.traces.sum(axis=1))
    assert_array_equal(adc.response((pulse,), grid).traces, conditional.traces)


def test_single_jump_preserves_cavity_memory_and_exponential_survival() -> None:
    rate = 0.025
    adc = ADC(model(np.array([[0, rate, 0], [0, 0, 0], [0, 0, 0]])))
    pulse = ReadoutPulse(0, 90, 1.0, 6.0)
    grid = SampleGrid(np.arange(0, 81, 2, dtype=float), 6.0)
    population = PopulationInput(adc.model.state_labels, np.array([0, 1.0, 0]))
    result = adc.sample(population, (pulse,), grid, SamplingIdentity(52), shots=600)
    no_jump = (
        ADC(replace(adc.model, rates_per_ns=np.zeros((3, 3))))
        .response((pulse,), grid)
        .traces[1]
    )
    unchanged = np.max(np.abs(result.traces - no_jump), axis=(1, 2)) < 2e-8
    expected_survival = np.exp(-rate * grid.times_ns[-1])
    assert abs(unchanged.mean() - expected_survival) < 0.05

    # A worked single-cavity solution fits an unknown continuous jump time.
    # The old cavity field survives the jump; resetting to vacuum or switching
    # between two no-jump traces cannot fit the whole ringdown transient.
    v = np.sqrt(0.16 / 2)
    a_e, a_g = 0.1 + 2j * np.pi * 0.04, 0.1 + 0j

    def expected(jump: float) -> np.ndarray:
        times = grid.times_ns
        before = v / a_e * (1 - np.exp(-a_e * times))
        at_jump = v / a_e * (1 - np.exp(-a_e * jump))
        elapsed = np.maximum(times - jump, 0)
        after = v / a_g + (at_jump - v / a_g) * np.exp(-a_g * elapsed)
        outgoing = 1 - v * np.where(times < jump, before, after)
        return np.column_stack((outgoing.real, -outgoing.imag))

    selected = [
        trace
        for trace in result.traces
        if np.linalg.norm(trace[10] - no_jump[10]) > 0.1
    ]
    assert selected
    trace = selected[0]
    fit = minimize_scalar(
        lambda jump: float(np.sum((expected(jump) - trace) ** 2)),
        bounds=(0, 20),
        method="bounded",
    )
    assert 0 < fit.x < 20
    assert_allclose(trace, expected(float(fit.x)), atol=2e-7)


def test_shaped_paths_share_windows_delay_mismatch_and_seed() -> None:
    base = model(np.array([[0, 0.025, 0], [0, 0, 0.04], [0, 0, 0]]))
    receiver = Receiver(gain=1.7, phase_rad=0.3, offset=np.array([0.2, -0.4]))
    base = replace(base, receiver=receiver)
    envelope = PiecewiseLinearEnvelope(
        np.array([0, 15, 40.0]), np.array([0, 1 + 0.2j, 0.4], complex)
    )
    pulse = ReadoutPulse(5, 65, 0.8 + 0.2j, 6.007, phase_rad=0.2, envelope=envelope)
    grid = SampleGrid(np.array([-5, 5, 10, 25, 45, 65, 85.0]), 6.003, phase_rad=0.1)
    population = PopulationInput(base.state_labels, np.array([0.0, 0.0, 1.0]))
    identity = SamplingIdentity(89, (3, 1))
    adc = ADC(base)
    first, shared = adc.sample_windows(
        population, (pulse,), (grid, grid), identity, shots=80
    )
    repeated = adc.sample(population, (pulse,), grid, identity, shots=80)
    assert_array_equal(first.traces, shared.traces)
    assert_array_equal(first.traces, repeated.traces)
    changed = adc.sample(
        population, (pulse,), grid, SamplingIdentity(90, (3, 1)), shots=80
    )
    assert not np.array_equal(first.traces, changed.traces)
    assert first.traces[:, 4].var(axis=0).sum() > 0.01

    delay = 2.5
    delayed_grid = replace(grid, times_ns=grid.times_ns + delay)
    delayed = ADC(
        replace(base, receiver=replace(receiver, time_of_flight_ns=delay))
    ).sample(population, (pulse,), delayed_grid, identity, shots=80)
    z = first.traces[..., 0] + 1j * first.traces[..., 1] - (0.2 - 0.4j)
    expected = z * np.exp(-2j * np.pi * grid.demodulation_frequency_ghz * delay) + (
        0.2 - 0.4j
    )
    assert_allclose(
        delayed.traces, np.stack((expected.real, expected.imag), axis=-1), atol=3e-8
    )

    phi = 0.6
    mismatched = ADC(replace(base, mismatch_phase_rad=np.array([phi, 0]))).sample(
        population, (pulse,), grid, identity, shots=80
    )
    # Isolate the scattered signal by subtracting the uncoupled direct field.
    direct_model = replace(base, kappa_external_per_ns=np.zeros(2))
    direct = ADC(direct_model).response((pulse,), grid).traces[0]
    scatter = (first.traces - direct)[..., 0] + 1j * (first.traces - direct)[..., 1]
    delta = mismatched.traces - direct
    assert_allclose(
        delta[..., 0] + 1j * delta[..., 1], scatter * np.exp(1j * phi), atol=3e-8
    )
