"""Uniform error for the public sampled-envelope simplification seam."""

import numpy as np
import pytest
from zcu_tools.simulate.evolution import EnvelopeSpec


@pytest.mark.parametrize("shape", ["gaussian", "isolated"])
def test_simplified_envelope_bounds_complex_interpolation_error(shape: str) -> None:
    times = np.linspace(0, 1000, 2001)
    if shape == "gaussian":
        real = np.exp(-((times - 500) / 100) ** 2)
        samples = (real + 0.4j * (times - 500) / 100 * real).astype(np.complex128)
    else:
        samples = np.zeros(len(times), dtype=np.complex128)
        samples[801] = 1 + 0.7j
    original = EnvelopeSpec("sampled", samples, times)
    epsilon = 1e-4
    simplified = original.simplified(epsilon)
    # The difference of two piecewise-linear curves has its norm maximum
    # at a knot of their union. Simplification must use original knots.
    assert np.isin(simplified.sample_ns, original.sample_ns).all()
    error = np.max(np.abs(original.values(times) - simplified.values(times)))
    assert error <= epsilon
    assert len(simplified.samples) < len(original.samples)


def test_zero_error_keeps_original_samples() -> None:
    original = EnvelopeSpec(
        "sampled", np.array([0, 1j, 0], dtype=np.complex128), np.array([0, 1, 2.0])
    )
    simplified = original.simplified(0)
    np.testing.assert_array_equal(simplified.sample_ns, original.sample_ns)
    np.testing.assert_array_equal(simplified.samples, original.samples)
