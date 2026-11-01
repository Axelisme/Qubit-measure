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
