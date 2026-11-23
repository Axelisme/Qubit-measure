import numpy as np
import pytest
from zcu_tools.analysis import shot_uncertainty


def test_iq_mean_covariance_keeps_cross_term_and_uses_retained_counts() -> None:
    # Paired shots (0, 0) and (3, 4), then the same two shots retained twice.
    moments = np.array(
        [
            [2, 3, 4, 9, 12, 16],
            [4, 6, 8, 18, 24, 32],
            [1, 3, 4, 9, 12, 16],
            [2, 3, 4, 9, np.nan, 16],
        ],
        dtype=np.float64,
    )

    covariance = shot_uncertainty.iq_mean_covariance(moments)

    np.testing.assert_allclose(covariance[0], [[2.25, 3], [3, 4]])
    np.testing.assert_allclose(covariance[1], [[0.75, 1], [1, 4 / 3]])
    assert np.isnan(covariance[2:]).all()


@pytest.mark.parametrize(("count", "i"), [(70, 0.3), (12_200, 1 / 3)])
def test_iq_estimators_accept_serially_accumulated_constant_shots(
    count: int, i: float
) -> None:
    # Accumulate in shot order rather than multiplying by the final count.
    shot = np.array([1, i, 0, i * i, 0, 0], dtype=np.float64)
    moments = np.zeros_like(shot)
    for _ in range(count):
        moments += shot

    covariance = shot_uncertainty.iq_mean_covariance(moments)
    np.testing.assert_array_equal(covariance, np.zeros((2, 2)))
    assert shot_uncertainty.iq_shot_sem(moments, direction=(1, 0)) == 0


def test_iq_estimators_keep_small_positive_scatter() -> None:
    # Two shots around I=1, separated by 2**-24 (SEM is half the separation).
    moments = np.array([2, 2, 0, 2 + 2.0**-49, 0, 0])
    covariance = shot_uncertainty.iq_mean_covariance(moments)
    np.testing.assert_array_equal(covariance, [[2.0**-50, 0], [0, 0]])
    assert shot_uncertainty.iq_shot_sem(moments, direction=(1, 0)) == pytest.approx(
        2.0**-25, abs=0
    )


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


def test_iq_sem_propagates_each_points_iq_weights() -> None:
    # The same two shots (0, 0) and (3, 4) at each point project to 0
    # and [5, 2, -1] with these weights. SEM is half the absolute difference.
    moments = np.tile([2.0, 3, 4, 9, 12, 16], (2, 3, 1))
    moments[1, 1] = np.nan
    sem = shot_uncertainty.iq_shot_sem(moments, direction=(np.array([3.0, 2, 1]), -1.0))
    np.testing.assert_allclose(sem, [[2.5, 1, 0.5], [2.5, np.nan, 0.5]])


def test_independent_iq_sem_combines_groups_with_pointwise_gradients() -> None:
    # Group 0 has shots (0, 0) and (2, -2); group 1 has (0, 0) and (6, 0).
    # At the first point their projections differ by 8 and 6: SEMs 4 and 3.
    moments = np.tile([[2.0, 2, -2, 4, -4, 4], [2.0, 6, 0, 36, 0, 0]], (3, 1, 1))
    moments[1, 1] = np.nan
    gradients = np.array([[2 - 2j, 1], [1, 1], [1, np.inf]], dtype=np.complex128)
    sem = shot_uncertainty.propagated_iq_shot_sem(moments, gradients=gradients)
    np.testing.assert_allclose(sem, [5.0, np.nan, np.nan])


def test_propagated_iq_sem_keeps_large_finite_projection() -> None:
    # Shots ±(sqrt(5e307), sqrt(5e307)) project to ±sqrt(2e308) with I+Q.
    # The squared projection sum overflows, but the two-shot SEM is finite.
    moments = np.array([[[2.0, 0, 0, 1e308, 1e308, 1e308]]])
    sem = shot_uncertainty.propagated_iq_shot_sem(
        moments, gradients=np.array([[1 + 1j]])
    )

    np.testing.assert_allclose(sem, [np.sqrt(2.0) * 1e154], rtol=1e-14)


def test_iq_sem_rejects_weights_with_extra_point_axes() -> None:
    moments = np.array([[2.0, 3, 4, 9, 12, 16]])
    with pytest.raises(ValueError, match="broadcast"):
        shot_uncertainty.iq_shot_sem(moments, direction=(np.ones((2, 1)), 0.0))


@pytest.mark.parametrize("count", [-1.0, 1.5, np.inf, -np.inf])
def test_iq_sem_rejects_invalid_counts_even_with_missing_sums(count: float) -> None:
    with pytest.raises(ValueError, match="count"):
        shot_uncertainty.iq_shot_sem(
            np.array([count, np.nan, 0, 0, 0, 0]), direction=(1, 0)
        )


def test_iq_sem_leaves_missing_points_undefined_but_rejects_infinite_moments() -> None:
    missing = np.array([3, 2, 0, 2, np.nan, 0])
    assert np.isnan(shot_uncertainty.iq_shot_sem(missing, direction=(1, 0)))
    with pytest.raises(ValueError, match="infinite"):
        shot_uncertainty.iq_shot_sem(
            np.array([1, 2, 0, 4, np.nan, np.inf]), direction=(1, 0)
        )


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ([70, 0, 4, 2, 0, 0], "negative centered sum of squares"),
        ([70, 0, 0, 2, 4, 2], "covariance bound"),
    ],
)
def test_iq_estimators_classify_impossible_statistics_even_off_projection(
    raw: list[float], reason: str
) -> None:
    moments = np.array(raw, dtype=np.float64)
    with pytest.raises(shot_uncertainty.IQCovarianceError, match=reason):
        shot_uncertainty.iq_mean_covariance(moments)
    with pytest.raises(shot_uncertainty.IQCovarianceError, match=reason):
        shot_uncertainty.iq_shot_sem(moments, direction=(1, 0))


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ([2, 1, 0, 1, 0], "shape"),
        ([1.5, np.nan, 0, 0, 0, 0], "count"),
        ([1, 1, 0, np.inf, 0, 0], "infinite"),
    ],
)
def test_iq_covariance_distinguishes_invalid_input_from_unavailable_scatter(
    raw: list[float], reason: str
) -> None:
    with pytest.raises(ValueError, match=reason) as failure:
        shot_uncertainty.iq_mean_covariance(np.array(raw, dtype=np.float64))
    assert not isinstance(failure.value, shot_uncertainty.IQCovarianceError)


def test_iq_sem_accepts_rounding_in_unprojected_statistics() -> None:
    # Two shots (0, -2) and (2, -2) have zero Q variance and zero covariance.
    moments = np.array(
        [2, 2, -4, 4, np.nextafter(-4.0, -np.inf), np.nextafter(8.0, 0.0)]
    )
    assert shot_uncertainty.iq_shot_sem(moments, direction=(1, 0)) == pytest.approx(1)


def test_iq_sem_keeps_large_finite_rank_one_moments_accurate() -> None:
    # Shots (4e153, 4e153) and (1e154, 1e154). The raw sum squared and the
    # product of centered diagonal sums overflow even though the SEM is finite.
    moments = np.array([2, 1.4e154, 1.4e154, 1.16e308, 1.16e308, 1.16e308])
    assert shot_uncertainty.iq_shot_sem(moments, direction=(1, 0)) == pytest.approx(
        3e153
    )


def test_class_counts_give_binomial_sem_and_multinomial_cross_covariance() -> None:
    binary = shot_uncertainty.class_probability_covariance(
        np.array([1, 3], dtype=np.int64)
    )
    # A one-hot sample with three successes and one failure has sample SD 0.5.
    np.testing.assert_allclose(np.sqrt(np.diag(binary)), [0.25, 0.25])

    # Identical fractions, but twice as many retained shots at the second point.
    counts = np.array([[2, 1, 1], [4, 2, 2], [0, 0, 0]], dtype=np.float64)
    covariance = shot_uncertainty.class_probability_covariance(counts)
    one_hot = np.eye(3)[[0, 0, 1, 2]]
    np.testing.assert_allclose(covariance[0], np.cov(one_hot, rowvar=False) / 4)
    doubled = np.repeat(one_hot, 2, axis=0)
    np.testing.assert_allclose(covariance[1], np.cov(doubled, rowvar=False) / 8)
    assert np.isnan(covariance[2]).all()


def test_class_covariance_requires_at_least_two_retained_shots() -> None:
    covariance = shot_uncertainty.class_probability_covariance(
        np.array([[1, 0], [2, 0]], dtype=np.int64)
    )
    assert np.isnan(covariance[0]).all()
    np.testing.assert_array_equal(covariance[1], np.zeros((2, 2)))


def test_corrected_population_sem_propagates_multinomial_shot_error() -> None:
    # The first two labels mix; p_g = 4 * (measured_g - 0.5) when no f
    # labels are present. The raw SEM is 0.25 for one g shot out of four.
    matrix = np.array([[0.75, 0.25, 0], [0.5, 0.5, 0], [0, 0, 1]])
    counts = np.array([[1, 3, 0], [1, 0, 0], [0, 4, 0]], dtype=np.int64)

    sem = shot_uncertainty.corrected_population_sem(counts, matrix)

    np.testing.assert_allclose(sem, [[1, 1, 0], [np.nan] * 3, [0, 0, 0]])


def test_singular_calibration_leaves_no_corrected_sem_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    counts = np.array([[1, 1], [2, 0]], dtype=np.int64)
    matrix = np.full((2, 2), 0.5)

    sem = shot_uncertainty.corrected_population_sem(counts, matrix)

    assert np.isnan(sem).all()
    assert "singular" in caplog.text


def test_nearly_singular_calibration_retains_large_finite_sem() -> None:
    contrast = 2.0**-30
    matrix = np.array([[0.5 + contrast, 0.5 - contrast], [0.5, 0.5]])
    counts = np.array([1, 1], dtype=np.int64)

    sem = shot_uncertainty.corrected_population_sem(counts, matrix)

    np.testing.assert_allclose(sem, [2.0**29, 2.0**29])


def test_corrected_population_sem_does_not_require_representable_variance() -> None:
    # One Other shot out of two gives a raw SEM of 0.5. Calibration scales
    # that coordinate by 1e160, so its SEM fits in float64 but its variance does not.
    counts = np.array([1, 0, 1], dtype=np.int64)
    matrix = np.diag([1.0, 1.0, 1e-160])

    sem = shot_uncertainty.corrected_population_sem(counts, matrix)

    np.testing.assert_allclose(sem, [0.5, 0.0, 5e159])


def test_class_covariance_rejects_nonfinite_total_of_finite_counts() -> None:
    limit = np.finfo(np.float64).max
    with pytest.raises(ValueError, match="total shot count"):
        shot_uncertainty.class_probability_covariance(np.array([limit, limit]))


@pytest.mark.parametrize("count", [-1.0, 1.5, np.inf, -np.inf])
def test_class_covariance_rejects_invalid_counts_even_with_missing_classes(
    count: float,
) -> None:
    missing = shot_uncertainty.class_probability_covariance(np.array([np.nan, 3]))
    assert np.isnan(missing).all()
    with pytest.raises(ValueError, match="count"):
        shot_uncertainty.class_probability_covariance(np.array([count, np.nan]))
