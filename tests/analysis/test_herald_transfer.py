"""Recover state transfer through the public shared-readout calibration seam."""

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.analysis.failure import AnalysisFailed
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
        [np.ones(centers.size), x, y, x * x + sigma**2, x * y, y * y + sigma**2]
    )
    joint = populations[None, :, None] * transitions
    counts = np.arange(1, len(transitions) + 1) * 12_345
    return np.einsum("ai,xij,bj->xab", moments, joint, moments) * counts[:, None, None]


def test_two_state_exact_transfer_and_readout_in_original_units() -> None:
    centers = np.array([3.2 + 2.1j, 5.6 + 2.8j])
    populations = np.array([0.73, 0.27])
    transitions = np.array(
        [
            [[0.95, 0.05], [0.12, 0.88]],
            [[0.15, 0.85], [0.6, 0.4]],
            [[0.7, 0.3], [0.25, 0.75]],
        ]
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
    np.testing.assert_allclose(
        result.change_probability, [0.0689, 0.7825, 0.2865], atol=1e-6
    )
    assert result.singular_values.shape == (3,)
    np.testing.assert_allclose(result.singular_values[0], 1)
    assert result.singular_values[2] < 1e-10
    np.testing.assert_array_equal(sums, before)


def test_three_state_exact_transfer_sorted_by_herald_population() -> None:
    centers = np.array([1200 + 340j, 900 + 110j, 800 + 430j])
    populations = np.array([0.18, 0.67, 0.15])
    transitions = np.array(
        [
            [[0.8, 0.15, 0.05], [0.1, 0.65, 0.25], [0.2, 0.3, 0.5]],
            [[0.2, 0.4, 0.4], [0.7, 0.2, 0.1], [0.1, 0.6, 0.3]],
            [[0.9, 0.05, 0.05], [0.2, 0.7, 0.1], [0.4, 0.1, 0.5]],
        ]
    )
    result = selfcal_transfer(exact_sums(centers, 48.0, populations, transitions), 3)
    order = [1, 0, 2]
    joint = populations[None, :, None] * transitions

    np.testing.assert_allclose(result.centers, centers[order], atol=1e-5)
    assert result.sigma == pytest.approx(48.0, abs=1e-5)
    np.testing.assert_allclose(result.populations, populations[order], atol=1e-7)
    np.testing.assert_allclose(
        result.transition_matrices, transitions[:, order][:, :, order], atol=1e-7
    )
    np.testing.assert_allclose(
        result.joint_probabilities, joint[:, order][:, :, order], atol=1e-7
    )
    np.testing.assert_allclose(
        result.change_probability, [0.3455, 0.785, 0.294], atol=1e-7
    )
    assert result.singular_values.shape == (4,)
    assert result.singular_values[3] < 1e-10


@pytest.fixture
def two_state_sums() -> NDArray[np.float64]:
    probabilities = np.linspace(0, 0.9, 7)
    transitions = np.array(
        [[[1 - p, p], [0.8 * p, 1 - 0.8 * p]] for p in probabilities]
    )
    return exact_sums(
        np.array([1.2 - 0.9j, 3.5 + 0.4j]), 0.45, np.array([0.7, 0.3]), transitions
    )


def test_unmeasured_points_do_not_enter_calibration(
    two_state_sums: NDArray[np.float64],
) -> None:
    expected = selfcal_transfer(two_state_sums, 2)
    sums = two_state_sums.copy()
    sums[3] = np.nan

    result = selfcal_transfer(sums, 2)

    np.testing.assert_allclose(result.centers, expected.centers, atol=1e-6)
    np.testing.assert_allclose(result.populations, expected.populations, atol=1e-6)
    assert result.sigma == pytest.approx(expected.sigma, abs=1e-6)
    measured = [0, 1, 2, 4, 5, 6]
    np.testing.assert_allclose(
        result.transition_matrices[measured],
        expected.transition_matrices[measured],
        atol=1e-6,
    )
    assert np.isnan(result.transition_matrices[3]).all()
    assert np.isnan(result.joint_probabilities[3]).all()
    assert np.isnan(result.change_probability[3])


@pytest.mark.parametrize("n_states", [0, 1, 4])
def test_rejects_unsupported_state_counts(
    two_state_sums: NDArray[np.float64], n_states: int
) -> None:
    with pytest.raises(ValueError, match="n_states"):
        selfcal_transfer(two_state_sums, n_states)


@pytest.mark.parametrize("shape", [(6, 6), (3, 3, 3), (2, 6, 6, 1), (7, 6, 5)])
def test_rejects_wrong_moment_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="shape"):
        selfcal_transfer(np.zeros(shape))


@pytest.mark.parametrize(
    "case", ["single", "two_as_three", "collinear", "empty", "nan", "zero"]
)
def test_unidentifiable_data_fails_instead_of_returning_transfer(
    two_state_sums: NDArray[np.float64], case: str
) -> None:
    n_states = 2
    if case == "single":
        sums = exact_sums(np.array([1 + 2j]), 0.3, np.array([1.0]), np.ones((4, 1, 1)))
    elif case == "two_as_three":
        sums, n_states = two_state_sums, 3
    elif case == "collinear":
        n_states = 3
        sums = exact_sums(
            np.array([1 + 2j, 2 + 4j, 3 + 6j]),
            0.3,
            np.array([0.6, 0.3, 0.1]),
            np.array([np.eye(3), [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.4, 0.5]]]),
        )
    elif case == "empty":
        sums = np.empty((0, 6, 6))
    elif case == "nan":
        sums = np.full((4, 6, 6), np.nan)
    else:
        sums = np.zeros((4, 6, 6))

    with pytest.raises(AnalysisFailed, match="identif|points|centers"):
        selfcal_transfer(sums, n_states)


@pytest.mark.parametrize("n_states", [2, 3])
def test_sampled_gaussian_readout_recovers_transfer_within_sampling_error(
    n_states: int,
) -> None:
    rng = np.random.default_rng(83017)
    centers = np.array([2.3 + 1.4j, 4.8 + 2.3j, 3.1 + 4j])[:n_states]
    populations = np.array([0.7, 0.3]) if n_states == 2 else np.array([0.65, 0.25, 0.1])
    transitions = rng.dirichlet(np.ones(n_states), size=(21, n_states))
    transitions = 0.5 * transitions + 0.5 * np.eye(n_states)
    sums = np.empty((len(transitions), 6, 6))
    sigma, shots = 0.55, 20_000
    for index, transition in enumerate(transitions):
        herald_state = rng.choice(n_states, shots, p=populations)
        cumulative = np.cumsum(transition[herald_state], axis=1)
        signal_state = (rng.random((shots, 1)) > cumulative).sum(axis=1)
        herald = centers[herald_state] + sigma * (
            rng.normal(size=shots) + 1j * rng.normal(size=shots)
        )
        signal = centers[signal_state] + sigma * (
            rng.normal(size=shots) + 1j * rng.normal(size=shots)
        )
        h = np.column_stack(
            [
                np.ones(shots),
                herald.real,
                herald.imag,
                herald.real**2,
                herald.real * herald.imag,
                herald.imag**2,
            ]
        )
        s = np.column_stack(
            [
                np.ones(shots),
                signal.real,
                signal.imag,
                signal.real**2,
                signal.real * signal.imag,
                signal.imag**2,
            ]
        )
        sums[index] = h.T @ s

    result = selfcal_transfer(sums, n_states)

    # 420,000 pooled shots constrain calibration; rare-state K rows remain noisier.
    np.testing.assert_allclose(result.centers, centers, atol=0.03)
    assert result.sigma == pytest.approx(sigma, abs=0.02)
    np.testing.assert_allclose(result.populations, populations, atol=0.008)
    assert np.sqrt(np.mean((result.transition_matrices - transitions) ** 2)) < 0.03
    joint = populations[None, :, None] * transitions
    assert np.sqrt(np.mean((result.joint_probabilities - joint) ** 2)) < 0.008
    change = 1 - np.trace(joint, axis1=1, axis2=2)
    assert np.sqrt(np.mean((result.change_probability - change) ** 2)) < 0.012


@pytest.mark.parametrize("n_states", [2, 3])
def test_previous_calibration_refines_to_full_search_result(n_states: int) -> None:
    centers = np.array([-0.8 + 1.1j, 1.3 + 2.2j, 0.2 - 0.9j])[:n_states]
    populations = np.array([0.7, 0.3]) if n_states == 2 else np.array([0.6, 0.3, 0.1])
    first = np.repeat(np.eye(n_states)[None], 3, axis=0)
    previous = selfcal_transfer(exact_sums(centers, 0.4, populations, first), n_states)
    # A changed pooled herald origin/scale must not change the warm-start units.
    updated_centers = np.asarray(centers * 1.04 + 0.03 - 0.02j, dtype=np.complex128)
    updated_populations = populations + np.array(
        [0.02, -0.02] if n_states == 2 else [0.02, -0.01, -0.01]
    )
    transitions = np.repeat((0.4 * np.eye(n_states) + 0.6 / n_states)[None], 4, axis=0)
    sums = exact_sums(updated_centers, 0.42, updated_populations, transitions)

    full = selfcal_transfer(sums, n_states)
    warm = selfcal_transfer(sums, n_states, previous=previous)

    np.testing.assert_allclose(warm.centers, full.centers, atol=1e-6)
    assert warm.sigma == pytest.approx(full.sigma, abs=1e-6)
    np.testing.assert_allclose(warm.populations, full.populations, atol=1e-6)
    np.testing.assert_allclose(
        warm.transition_matrices, full.transition_matrices, atol=1e-6
    )
    np.testing.assert_allclose(
        warm.change_probability, full.change_probability, atol=1e-6
    )
