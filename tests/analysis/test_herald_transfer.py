"""Recover state transfer through the public shared-readout calibration seam."""

import numpy as np
import pytest
from numpy.typing import NDArray

from zcu_tools.analysis.herald_transfer import selfcal_transfer


def exact_sums(
    centers: NDArray[np.complex128],
    sigma: float,
    populations: NDArray[np.float64],
    transitions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Analytic conditional Gaussian moments, with unequal shot counts."""
    x, y = centers.real, centers.imag
    moments = np.array(
        [np.ones(centers.size), x, y, x*x + sigma**2, x*y, y*y + sigma**2]
    )
    joint = populations[None, :, None] * transitions
    counts = np.arange(1, len(transitions) + 1) * 12_345
    return np.einsum("ai,xij,bj->xab", moments, joint, moments) * counts[:, None, None]


def test_two_state_exact_transfer_and_readout_in_original_units() -> None:
    centers = np.array([3.2 + 2.1j, 5.6 + 2.8j])
    populations = np.array([0.73, 0.27])
    transitions = np.array(
        [[[0.95, 0.05], [0.12, 0.88]],
         [[0.15, 0.85], [0.6, 0.4]],
         [[0.7, 0.3], [0.25, 0.75]]]
    )
    sums = exact_sums(centers, 0.38, populations, transitions)
    before = sums.copy()

    result = selfcal_transfer(sums, 2)

    np.testing.assert_allclose(result.centers, centers, atol=1e-6)
    np.testing.assert_allclose(result.sigma, 0.38, atol=1e-6)
    np.testing.assert_allclose(result.populations, populations, atol=1e-6)
    np.testing.assert_allclose(result.transition_matrices, transitions, atol=1e-6)
    expected_joint = populations[None, :, None] * transitions
    np.testing.assert_allclose(result.joint_probabilities, expected_joint, atol=1e-6)
    np.testing.assert_allclose(result.change_probability, [0.0689, 0.7825, 0.2865], atol=1e-6)
    assert result.singular_values.shape == (3,)
    np.testing.assert_allclose(result.singular_values[0], 1)
    assert result.singular_values[2] < 1e-10
    np.testing.assert_array_equal(sums, before)


def test_three_state_exact_transfer_sorted_by_herald_population() -> None:
    centers = np.array([1200 + 340j, 900 + 110j, 800 + 430j])
    populations = np.array([0.18, 0.67, 0.15])
    transitions = np.array(
        [[[0.8, 0.15, 0.05], [0.1, 0.65, 0.25], [0.2, 0.3, 0.5]],
         [[0.2, 0.4, 0.4], [0.7, 0.2, 0.1], [0.1, 0.6, 0.3]],
         [[0.9, 0.05, 0.05], [0.2, 0.7, 0.1], [0.4, 0.1, 0.5]]]
    )
    result = selfcal_transfer(exact_sums(centers, 48.0, populations, transitions), 3)
    order = [1, 0, 2]
    joint = populations[None, :, None] * transitions

    np.testing.assert_allclose(result.centers, centers[order], atol=1e-5)
    assert result.sigma == pytest.approx(48.0, abs=1e-5)
    np.testing.assert_allclose(result.populations, populations[order], atol=1e-7)
    np.testing.assert_allclose(result.transition_matrices, transitions[:, order][:, :, order], atol=1e-7)
    np.testing.assert_allclose(result.joint_probabilities, joint[:, order][:, :, order], atol=1e-7)
    np.testing.assert_allclose(result.change_probability, [0.345, 0.703, 0.294], atol=1e-7)
    assert result.singular_values.shape == (4,)
    assert result.singular_values[3] < 1e-10
