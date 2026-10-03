from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from zcu_tools.analysis.fitting.base import quadratic_fit, quadratic_fit_wo_a

type QuadraticFitter = Callable[
    [NDArray[np.float64], NDArray[np.float64]], tuple[float, ...]
]


def test_quadratic_fit_recovers_normalized_ellipse() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
    xs = 2.0 * np.cos(theta)
    ys = 3.0 * np.sin(theta)
    expected = np.array([0.25, 0.0, 1.0 / 9.0, 0.0, 0.0, -1.0])
    expected /= np.linalg.norm(expected)

    result = quadratic_fit(xs, ys)

    assert isinstance(result, tuple)
    assert len(result) == 6
    assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-12)
    assert abs(np.dot(result, expected)) == pytest.approx(1.0, abs=1e-12)


def test_quadratic_fit_without_x_squared_recovers_normalized_parabola() -> None:
    ys = np.linspace(-2.0, 3.0, 40)
    xs = 2.0 * ys**2 + 3.0 * ys + 4.0
    expected = np.array([-1.0, 2.0, 3.0, 0.0, 4.0])
    expected /= np.linalg.norm(expected)

    result = quadratic_fit_wo_a(xs, ys)

    assert isinstance(result, tuple)
    assert len(result) == 5
    assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-12)
    assert abs(np.dot(result, expected)) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("fit", [quadratic_fit, quadratic_fit_wo_a])
def test_quadratic_fit_ignores_incomplete_finite_pairs(fit: QuadraticFitter) -> None:
    xs = np.linspace(-2.0, 2.0, 20)
    ys = np.sin(xs)
    expected = np.asarray(fit(xs, ys))
    incomplete_xs = np.append(xs, [np.nan, 3.0, np.inf, 4.0])
    incomplete_ys = np.append(ys, [1.0, np.nan, 2.0, -np.inf])

    result = np.asarray(fit(incomplete_xs, incomplete_ys))

    assert np.all(np.isfinite(result))
    same_sign_error = np.linalg.norm(result - expected)
    opposite_sign_error = np.linalg.norm(result + expected)
    assert min(same_sign_error, opposite_sign_error) < 1e-12


@pytest.mark.parametrize("fit, size", [(quadratic_fit, 6), (quadratic_fit_wo_a, 5)])
def test_quadratic_fit_returns_nan_tuple_without_finite_pairs(
    fit: QuadraticFitter, size: int
) -> None:
    xs = np.array([np.nan, 1.0, np.inf])
    ys = np.array([0.0, np.inf, np.nan])

    result = fit(xs, ys)

    assert isinstance(result, tuple)
    assert len(result) == size
    assert np.all(np.isnan(result))
