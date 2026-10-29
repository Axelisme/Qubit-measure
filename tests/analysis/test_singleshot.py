"""Public IQ classification and population probability contracts."""

import numpy as np
import pytest
from zcu_tools.analysis.fitting.singleshot import transition_state_circle_probabilities
from zcu_tools.analysis.singleshot import (
    analyze_centroid3_iq,
    analyze_reference_iq,
    classify_rabi_iq,
    classify_reference_iq,
    classify_result,
    correct_populations,
    raw_population_signal,
)
from zcu_tools.utils.shot_classification import gaussian_region_probability


def test_centroid3_iq_analysis_preserves_scan_axes_and_unclipped_estimates() -> None:
    from zcu_tools.cfg_model import Centroid3Classifier

    calibration = Centroid3Classifier(
        g_center=0j,
        e_center=2 + 0j,
        f_center=4 + 0j,
        confusion_matrix=[[0.75, 0.25, 0], [0.25, 0.75, 0], [0, 0, 1]],
    )
    shots = np.array([[[0, 0, 0, 0], [0, 2, 4, 1]]], dtype=np.complex128)
    result = analyze_centroid3_iq(shots, calibration)
    np.testing.assert_allclose(result.raw_populations, [[[1, 0, 0], [0.5, 0.25, 0.25]]])
    np.testing.assert_allclose(
        result.corrected_populations, [[[1.5, -0.5, 0], [0.625, 0.125, 0.25]]]
    )
    np.testing.assert_allclose(result.confusion_matrix, calibration.confusion_matrix)
    assert result.condition_number == pytest.approx(2)
    assert result.minimum_singular_value == pytest.approx(0.5)


def test_population_signal_and_readout_correction() -> None:
    populations = np.array([[0.3, 0.5, 0.2], [0.4, 0.4, 0.2]], dtype=np.float64)
    matrix = np.array([[0.8, 0.2, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 1.0]])
    measured = populations @ matrix
    np.testing.assert_allclose(correct_populations(measured, matrix), populations)
    assert correct_populations(populations, None) is populations
    other_adc = np.array([[[0.8, 0.1], [0.9, 0.0]]], dtype=np.float64)
    np.testing.assert_array_equal(
        raw_population_signal([populations[None, ...], other_adc]), populations
    )
    with pytest.raises(np.linalg.LinAlgError):
        correct_populations(populations, np.zeros((3, 3)))


def test_correction_can_preserve_unphysical_estimates_without_changing_default() -> (
    None
):
    matrix = np.array([[0.8, 0.2, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 1.0]])
    measured = np.array([[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(
        correct_populations(measured, matrix, clip=False), [[-1 / 7, 8 / 7, 0]]
    )
    np.testing.assert_allclose(correct_populations(measured, matrix), [[0, 1, 0]])
    assert correct_populations(measured, None, clip=False) is measured


def test_reference_iq_uses_training_only_and_reports_heldout_label_fractions() -> None:
    training = np.array([0, 10, 10j])[:, None] + np.array([1, -1, 1j, -1j])
    evaluation = np.array([[0, 0, 0, 0], [10, 10, 10, 10], [10j, 10j, 10j, 0]])
    signals = np.concatenate((training, evaluation), axis=1)
    result = analyze_reference_iq(signals, np.array([0, 1, 2]))
    np.testing.assert_allclose(result.centroids, [0, 10, 10j])
    np.testing.assert_allclose(result.confusion_matrix, np.eye(3))
    np.testing.assert_allclose(
        result.raw_populations, [[1, 0, 0], [0, 1, 0], [0.25, 0, 0.75]]
    )
    np.testing.assert_allclose(result.corrected_populations, result.raw_populations)
    np.testing.assert_allclose(result.label_errors, [0, 0, 0.25])
    assert result.macro_error == pytest.approx(1 / 12)
    assert result.minimum_separation == pytest.approx(np.sqrt(150))
    assert result.condition_number == pytest.approx(1)
    assert result.minimum_singular_value == pytest.approx(1)
    assert result.training_shots == result.evaluation_shots == 4


@pytest.mark.parametrize(
    ("signals", "labels"),
    [
        (np.ones((2, 8), dtype=complex), np.array([0, 1])),
        (np.ones((3, 5), dtype=complex), np.array([0, 1, 2])),
        (np.ones((3, 8), dtype=complex), np.array([0, 0, 2])),
        (np.ones((3, 8), dtype=complex), np.array([0.0, 1.5, 2.0])),
        (np.full((3, 8), complex(np.nan, 0)), np.array([0, 1, 2])),
        (np.ones((3, 8), dtype=complex), np.array([0, 1, 2])),
    ],
)
def test_reference_iq_rejects_invalid_or_unidentifiable_references(
    signals, labels
) -> None:
    with pytest.raises(ValueError, match="reference|Reference"):
        analyze_reference_iq(signals, labels)


def test_reference_iq_sorts_labels_and_keeps_odd_remainder_for_evaluation() -> None:
    signals = np.array([0, 10, 10j])[:, None] + np.array(
        [1, -1, 1j, -1j, 0, 0, 0, 0, 0]
    )
    result = analyze_reference_iq(signals[[2, 0, 1]], np.array([2, 0, 1]))
    np.testing.assert_allclose(result.centroids, [0, 10, 10j])
    np.testing.assert_allclose(result.raw_populations, np.eye(3))
    assert result.training_shots == 4
    assert result.evaluation_shots == 5


def test_reference_correction_uses_training_rows_without_clipping() -> None:
    offsets = np.array([1, -1, 1j, -1j, 1, -1, 1j, -1j])
    training = np.array([0, 10, 10j])[:, None] + offsets
    training[2, :2] = [1, -1]
    evaluation = np.repeat(np.array([0, 10, 10j])[:, None], 8, axis=1)
    result = analyze_reference_iq(
        np.concatenate((training, evaluation), axis=1), np.array([0, 1, 2])
    )
    np.testing.assert_allclose(
        result.confusion_matrix, [[1, 0, 0], [0, 1, 0], [0.25, 0, 0.75]]
    )
    np.testing.assert_allclose(
        result.corrected_populations, [[1, 0, 0], [0, 1, 0], [-1 / 3, 0, 4 / 3]]
    )


def test_reference_iq_rejects_singular_within_cloud_covariance() -> None:
    signals = np.array([0, 10, 20])[:, None] + np.arange(8, dtype=complex)
    with pytest.raises(ValueError, match="covariance"):
        analyze_reference_iq(signals, np.array([0, 1, 2]))


def test_reference_diagnostics_do_not_depend_on_adc_unit_scale() -> None:
    signals = 1e-9 * (
        np.array([0, 10, 10j], dtype=np.complex128)[:, None]
        + np.array([1, -1, 1j, -1j, 0, 0, 0, 0], dtype=np.complex128)
    )
    result = analyze_reference_iq(signals, np.array([0, 1, 2]))
    assert result.minimum_separation == pytest.approx(np.sqrt(150))
    np.testing.assert_allclose(result.confusion_matrix, np.eye(3))


def test_reference_classifier_applies_fixed_centers_over_scan_axes() -> None:
    centers = np.array([0, 10, 10j], dtype=np.complex128)
    shots = np.array([[[0, 10, 10j, 5], [10j, 10j, 10j, 0]]], dtype=np.complex128)
    np.testing.assert_allclose(
        classify_reference_iq(shots, centers),
        [[[0.5, 0.25, 0.25], [0.25, 0, 0.75]]],
    )
    np.testing.assert_allclose(
        classify_reference_iq(shots[0, 0], centers), [0.5, 0.25, 0.25]
    )


@pytest.mark.parametrize(
    ("shots", "centers"),
    [
        (np.array(0j), np.array([0, 10, 10j])),
        (np.empty((2, 0), dtype=complex), np.array([0, 10, 10j])),
        (np.array([np.nan + 0j]), np.array([0, 10, 10j])),
        (np.array([0j]), np.array([0, 10])),
        (np.array([0j]), np.array([0, 0, 10j])),
        (np.array([0j]), np.array([0, np.inf, 10j])),
        (np.array([0j]), np.array([[0, 10, 10j]])),
    ],
)
def test_reference_classifier_rejects_incomplete_iq_or_invalid_centers(
    shots, centers
) -> None:
    with pytest.raises(ValueError, match="IQ|centroids"):
        classify_reference_iq(shots, centers)


def test_rabi_classification_reports_per_coordinate_fractions() -> None:
    signals = np.array([[-1, 1, 0], [-1, -1, 1]], dtype=np.complex128)
    np.testing.assert_allclose(
        classify_rabi_iq(signals, -1, 1, 0.5), [[1 / 3, 1 / 3], [2 / 3, 1 / 3]]
    )
    with pytest.raises(ValueError, match="shape"):
        classify_rabi_iq(signals[0], -1, 1, 0.5)


def test_boundaries_and_degenerate_centers() -> None:
    g, e, other = classify_result(np.array([-2, 0, 2], dtype=np.complex128), -1, 1, 1)
    assert other.all() and not g.any() and not e.any()
    with pytest.raises(ValueError, match="distinct"):
        classify_result(np.array([0j]), 0j, 0j, 1)


@pytest.mark.parametrize("radius", [0.7, 1.5, 2.0, 3.0])
def test_gaussian_regions_match_raw_shots(radius: float) -> None:
    rng = np.random.default_rng(42)
    sigma = 0.8
    shots = -1 + sigma * (rng.normal(size=250000) + 1j * rng.normal(size=250000))
    measured = [mask.mean() for mask in classify_result(shots, -1, 1, radius)]
    ground, excited = transition_state_circle_probabilities(
        -1, 1, sigma, 0.3, 0, radius
    )
    np.testing.assert_allclose(ground, measured, atol=0.004)
    np.testing.assert_allclose(excited, ground[[1, 0, 2]], atol=1e-10)
    assert np.all(ground >= 0)
    assert ground.sum() == pytest.approx(1)


def test_midpoint_other_mass_and_large_radius_limit() -> None:
    mass = gaussian_region_probability(np.array([1.0]), 0.5, 20, 2)
    assert mass[0] == pytest.approx(0.5, abs=1e-9)
    ground, _ = transition_state_circle_probabilities(-1, 1, 0.3, 0.2, 0.8, 2)
    assert ground.sum() == pytest.approx(1)
    assert np.all(ground >= 0)


def test_transition_regions_match_sampled_transition_distribution() -> None:
    from scipy.integrate import cumulative_trapezoid
    from zcu_tools.analysis.fitting.singleshot import calc_fc

    rng = np.random.default_rng(13)
    p_avg, length_ratio, sigma, radius = 0.3, 1.2, 0.6, 2.0
    grid = np.linspace(0, 1, 10001)
    rows = transition_state_circle_probabilities(
        -1, 1, sigma, p_avg, length_ratio, radius
    )
    for index, (start, direction, rate, back) in enumerate(
        [
            (-1, 2, p_avg * length_ratio, (1 - p_avg) * length_ratio),
            (1, -2, (1 - p_avg) * length_ratio, p_avg * length_ratio),
        ]
    ):
        density = calc_fc(grid, rate, back)
        cdf = cumulative_trapezoid(density, grid, initial=0)
        cdf /= cdf[-1]
        fractions = np.interp(rng.random(250000), cdf, grid)
        fractions[rng.random(250000) < np.exp(-rate)] = 0
        shots = (
            start
            + direction * fractions
            + sigma * (rng.normal(size=250000) + 1j * rng.normal(size=250000))
        )
        empirical = [mask.mean() for mask in classify_result(shots, -1, 1, radius)]
        np.testing.assert_allclose(rows[index], empirical, atol=0.004)
