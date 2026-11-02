"""Readout QND metrics recover a two-state shot model through the public seam."""

from dataclasses import astuple, fields

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.analysis.readout_metrics import readout_metrics


@pytest.fixture(scope="module")
def model_sums() -> NDArray[np.float64]:
    """Sample all branches with independent isotropic noise and unequal counts."""
    rng = np.random.default_rng(417)
    sums = np.empty((8, 6, 6), dtype=np.float64)
    direction = np.array([0.6, 0.8])
    ground = np.array([1.2, -0.9])
    for branch in range(8):
        prepare_pi, r1_on, middle_pi = branch // 4, (branch // 2) % 2, branch % 2
        count = 80_000 + 2_000 * branch
        initial = rng.random(count) < 0.17
        initial = np.logical_xor(initial, prepare_pi)
        flip_up, flip_down, pi_miss = (
            (0.08, 0.21, 0.12) if r1_on else (0.025, 0.09, 0.035)
        )
        flips = rng.random(count) < np.where(initial, flip_down, flip_up)
        final = np.logical_xor(initial, flips)
        if middle_pi:
            final = np.logical_xor(final, rng.random(count) >= pi_miss)
        r1 = rng.normal(0, 0.7, (count, 2))
        if r1_on:
            r1 += ground + 2.8 * initial[:, None] * direction
        r2 = ground + 2.8 * final[:, None] * direction
        r2 += rng.normal(0, 0.7, (count, 2))
        first = np.column_stack(
            (np.ones(count), r1, r1[:, 0] ** 2, r1[:, 0] * r1[:, 1], r1[:, 1] ** 2)
        )
        second = np.column_stack(
            (np.ones(count), r2, r2[:, 0] ** 2, r2[:, 0] * r2[:, 1], r2[:, 1] ** 2)
        )
        sums[branch] = first.T @ second
    return sums


def test_recovers_shared_calibration_and_distinct_on_off_dynamics(
    model_sums: NDArray[np.float64],
) -> None:
    before = model_sums.copy()
    result = readout_metrics(model_sums)

    assert result.snr == pytest.approx(4.0, abs=0.05)
    assert result.f_sep == pytest.approx(0.9772498681, abs=0.002)
    assert result.thermal_population == pytest.approx(0.17, abs=0.008)
    assert result.retention_on == pytest.approx(0.71, abs=0.015)
    assert result.retention_off == pytest.approx(0.885, abs=0.015)
    assert result.qnd_on == pytest.approx(0.855, abs=0.008)
    assert result.qnd_off == pytest.approx(0.9425, abs=0.008)
    assert result.flip_up_on == pytest.approx(0.08, abs=0.012)
    assert result.flip_up_off == pytest.approx(0.025, abs=0.012)
    assert result.flip_down_on == pytest.approx(0.21, abs=0.012)
    assert result.flip_down_off == pytest.approx(0.09, abs=0.012)
    assert result.pi_miss_on == pytest.approx(0.12, abs=0.015)
    assert result.pi_miss_off == pytest.approx(0.035, abs=0.015)
    assert result.retention_from_covariance == pytest.approx(0.71, abs=0.02)
    np.testing.assert_array_equal(model_sums, before)
    # Reference from the same seed/shots using the four (1, I, Q) moment axes.
    # Compressing symmetric second moments must not change any metric.
    np.testing.assert_allclose(
        astuple(result),
        (
            4.00013653378503, 0.9772535535957193, 0.1688476829207297,
            0.7126014383147572, 0.885710551705862, 0.8563007191573786,
            0.9428552758529309, 0.07660676709616154, 0.02353653945741957,
            0.21079179458908126, 0.09075290883671844, 0.1233073527100131,
            0.03794674474085881, 0.7076600627917273,
        ),
        rtol=1e-12,
        atol=1e-12,
    )


def test_empty_point_has_no_metrics() -> None:
    result = readout_metrics(np.zeros((8, 6, 6), dtype=np.float64))
    assert np.all(np.isnan(astuple(result)))


@pytest.mark.parametrize(
    "shape",
    [(8, 6, 5), (9, 6, 6), (6, 6), (8, 6, 6, 1), (8, 3, 3, 3, 3)],
)
def test_rejects_wrong_branch_or_moment_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="shape"):
        readout_metrics(np.zeros(shape, dtype=np.float64))


@pytest.mark.parametrize("branch", range(8))
def test_empty_branch_only_invalidates_dependent_metrics(
    model_sums: NDArray[np.float64], branch: int
) -> None:
    baseline = readout_metrics(model_sums)
    sums = model_sums.copy()
    sums[branch] = 0

    result = readout_metrics(sums)

    for field in fields(result):
        if branch in (2, 6):
            missing = True
        elif branch in (0, 4):
            missing = field.name.endswith("_off")
        elif branch in (1, 5):
            missing = field.name == "pi_miss_off"
        else:
            missing = field.name == "pi_miss_on"
        actual = getattr(result, field.name)
        if missing:
            assert np.isnan(actual), field.name
        else:
            assert actual == getattr(baseline, field.name), field.name


@pytest.mark.parametrize("contrast", [1.0, 2.0])
def test_nonpositive_inferred_separation_has_no_metrics(contrast: float) -> None:
    # Valid moments of two point masses separated in I with transverse variance 1.
    # Their Delta squared is negative for contrast 1 and zero for contrast 2.
    sums = np.zeros((8, 6, 6), dtype=np.float64)
    sums[:, 0, 0] = 1
    sums[:, 5, 0] = 1
    sums[6, 1, 0] = contrast
    sums[6, 3, 0] = contrast**2

    result = readout_metrics(sums)

    assert np.all(np.isnan(astuple(result)))
