"""State validation, basis conversion and integer propagator composition."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm

GROUND = np.diag([1, 0]).astype(complex)
PLUS = np.full((2, 2), 0.5, dtype=complex)


def test_population_conversion_names_source_target_and_order():
    from zcu_tools.simulate.evolution import Basis, DensityState, populations

    computational = Basis("computational", ("g", "e"), np.eye(2, dtype=complex))
    hadamard = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    target = Basis("x-eigenstates", ("plus", "minus"), hadamard)
    result = populations(DensityState(computational, PLUS), target)
    assert result.basis.identity == "x-eigenstates"
    assert result.basis.labels == ("plus", "minus")
    assert_allclose(result.values, [1, 0], atol=2e-9)
    reverse = populations(DensityState(target, GROUND), computational)
    assert_allclose(reverse.values, [0.5, 0.5], atol=2e-9)


def test_invalid_density_fails_without_repair():
    from zcu_tools.simulate.evolution import Basis, DensityState

    basis = Basis("computational", ("g", "e"), np.eye(2, dtype=complex))
    with pytest.raises(ValueError, match="positive"):
        DensityState(basis, np.diag([1.1, -0.1]).astype(complex))
    with pytest.raises(ValueError, match="trace"):
        DensityState(basis, 2 * GROUND)


def test_binary_integer_power():
    from zcu_tools.simulate.evolution import propagator_power

    x = np.array([[0, 1], [1, 0]], dtype=complex)
    u = np.asarray(expm(-0.012j * x), dtype=np.complex128)
    assert_allclose(propagator_power(u, 0), np.eye(2))
    assert_allclose(
        propagator_power(u, 1000003), expm(-0.012j * x * 1000003), atol=2e-9
    )
    with pytest.raises(ValueError, match="nonnegative integer"):
        propagator_power(u, -1)
