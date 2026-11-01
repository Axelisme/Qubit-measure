"""Model-driven array synthesis and raw IQ population sampling contracts."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray
from zcu_tools.experiment.probe import generators


@pytest.mark.parametrize("shape", [(4,), (2, 4), (4, 3), (2, 1, 3), ()])
def test_synthesize_forwards_native_model_call_and_preserves_model_shape(shape):
    xs = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    calls = []

    def model(coordinates: NDArray[np.float64], *params: float):
        calls.append((coordinates, params))
        return expected

    actual = generators.synthesize(model, xs, [7.0, -2.0], noise_std=0.0, seed=17)

    assert len(calls) == 1
    assert calls[0][0] is xs
    assert calls[0][1] == (7.0, -2.0)
    assert actual.shape == shape
    assert_array_equal(actual, expected)
    assert_array_equal(xs, [[1.0, 2.0], [3.0, 4.0]])


@pytest.mark.parametrize("complex_output", [False, True])
def test_synthesize_adds_independent_gaussian_noise_with_local_seed(complex_output):
    xs = np.arange(3.0)
    values = np.full((2, 30000), 2.0 + 3.0j if complex_output else 2.0)
    state = np.random.get_state()

    def model(coordinates: NDArray[np.float64]):
        return values

    noisy = generators.synthesize(model, xs, [], noise_std=0.4, seed=19)
    repeated = generators.synthesize(model, xs, [], noise_std=0.4, seed=19)
    different = generators.synthesize(model, xs, [], noise_std=0.4, seed=20)

    assert_array_equal(noisy, repeated)
    assert not np.array_equal(noisy, different)
    noise = noisy - values
    assert noise.real.mean() == pytest.approx(0.0, abs=0.005)
    assert noise.real.std() == pytest.approx(0.4, abs=0.005)
    assert np.mean(np.abs(noise.real) > 0.8) == pytest.approx(0.0455, abs=0.004)
    assert abs(np.corrcoef(noise.real)[0, 1]) < 0.02
    if complex_output:
        assert noise.imag.mean() == pytest.approx(0.0, abs=0.005)
        assert noise.imag.std() == pytest.approx(0.4, abs=0.005)
        assert abs(np.corrcoef(noise.real.ravel(), noise.imag.ravel())[0, 1]) < 0.02
    after = np.random.get_state()
    assert state[0] == after[0]
    assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    assert_allclose(values, 2.0 + 3.0j if complex_output else 2.0, atol=0, rtol=0)


@pytest.mark.parametrize(
    "xs, noise_std, message",
    [
        (np.array([]), 0.0, "xs"),
        (np.empty((2, 0)), 0.0, "xs"),
        (np.array([np.nan]), 0.0, "xs"),
        (np.array([np.inf]), 0.0, "xs"),
        (np.array([-np.inf]), 0.0, "xs"),
        (np.array([1.0]), -0.1, "noise_std"),
        (np.array([1.0]), np.nan, "noise_std"),
        (np.array([1.0]), np.inf, "noise_std"),
        (np.array([1.0]), -np.inf, "noise_std"),
    ],
)
def test_synthesize_rejects_only_declared_invalid_inputs_before_model(
    xs, noise_std, message
):
    def model(coordinates: NDArray[np.float64]):
        pytest.fail("invalid input reached model")

    with pytest.raises(ValueError, match=message):
        generators.synthesize(model, xs, [], noise_std=noise_std, seed=17)


def test_synthesize_leaves_parameters_output_and_exceptions_to_model():
    xs = np.array(1.0)
    values = np.array([np.nan, np.inf, -np.inf])

    def model(coordinates: NDArray[np.float64], value: float):
        assert np.isnan(value)
        return values

    assert_array_equal(
        generators.synthesize(model, xs, [np.nan], noise_std=0.0, seed=17), values
    )
    failure = RuntimeError("native model failure")

    def broken_model(coordinates: NDArray[np.float64], value: float):
        raise failure

    with pytest.raises(RuntimeError) as caught:
        generators.synthesize(broken_model, xs, [-1.0], noise_std=0.0, seed=17)
    assert caught.value is failure


def test_sample_iq_shots_draws_centers_from_population_with_local_seed():
    populations = np.array([0.0, 1.0, 0.2, 0.8, 0.5])
    state = np.random.get_state()
    samples = generators.sample_iq_shots(
        populations, g_center=-2.0 + 1j, e_center=3.0 - 1j,
        width=0.0, shots=30000, seed=17,
    )
    repeated = generators.sample_iq_shots(
        populations, g_center=-2.0 + 1j, e_center=3.0 - 1j,
        width=0.0, shots=30000, seed=17,
    )
    different = generators.sample_iq_shots(
        populations, g_center=-2.0 + 1j, e_center=3.0 - 1j,
        width=0.0, shots=30000, seed=18,
    )
    assert samples.shape == (5, 30000)
    assert samples.dtype == np.complex128
    assert_array_equal(samples, repeated)
    assert not np.array_equal(samples, different)
    assert np.all((samples == -2.0 + 1j) | (samples == 3.0 - 1j))
    assert_array_equal(samples[0], np.full(30000, -2.0 + 1j))
    assert_array_equal(samples[1], np.full(30000, 3.0 - 1j))
    assert_allclose(np.mean(samples == 3.0 - 1j, axis=1), populations, atol=0.007)
    assert abs(np.corrcoef((samples[2:4] == 3.0 - 1j).astype(float))[0, 1]) < 0.02
    after = np.random.get_state()
    assert state[0] == after[0]
    assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    assert_array_equal(populations, [0.0, 1.0, 0.2, 0.8, 0.5])
