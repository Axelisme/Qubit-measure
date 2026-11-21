import numpy as np
from zcu_tools.analysis import shot_uncertainty


def test_iq_sem_projects_cross_moment_and_uses_retained_count() -> None:
    # IQ shots (0, 0) and (3, 4) project to 0 and 5. The second point
    # retains the same two shots twice, even if both points started with 4 shots.
    moments = np.array(
        [
            [2, 3, 4, 9, 12, 16],
            [4, 6, 8, 18, 24, 32],
            [1, 3, 4, 9, 12, 16],
            [0, 0, 0, 0, 0, 0],
            [np.nan] * 6,
        ],
        dtype=np.float64,
    )
    sem = shot_uncertainty.iq_shot_sem(moments, direction=(0.6, 0.8))

    np.testing.assert_allclose(sem, [2.5, 1.4433756729740645, np.nan, np.nan, np.nan])


def test_class_counts_give_binomial_sem_and_multinomial_cross_covariance() -> None:
    binary = shot_uncertainty.class_probability_covariance(
        np.array([1, 3], dtype=np.int64)
    )
    np.testing.assert_allclose(np.sqrt(np.diag(binary)), [0.21650635094610965] * 2)

    # Identical fractions, but twice as many retained shots at the second point.
    counts = np.array([[2, 1, 1], [4, 2, 2], [0, 0, 0]], dtype=np.float64)
    covariance = shot_uncertainty.class_probability_covariance(counts)
    expected = np.array(
        [
            [0.0625, -0.03125, -0.03125],
            [-0.03125, 0.046875, -0.015625],
            [-0.03125, -0.015625, 0.046875],
        ]
    )
    np.testing.assert_allclose(covariance[0], expected)
    np.testing.assert_allclose(covariance[1], expected / 2)
    assert np.isnan(covariance[2]).all()
