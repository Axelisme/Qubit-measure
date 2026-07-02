from __future__ import annotations

import pytest
from zcu_tools.utils.math import IDWInterpolation


def test_idw_interpolation_empty_predicts_zero():
    assert IDWInterpolation().predict(3.0) == 0.0


def test_idw_interpolation_single_point_predicts_observed_value():
    interp = IDWInterpolation()
    interp.update(1.0, 4.0)

    assert interp.predict(10.0) == 4.0


def test_idw_interpolation_two_points_interpolates_and_extrapolates():
    interp = IDWInterpolation()
    interp.update(1.0, 3.0)
    interp.update(3.0, 7.0)

    assert interp.predict(2.0) == pytest.approx(5.0)
    assert interp.predict(4.0) == pytest.approx(9.0)


def test_idw_interpolation_duplicate_two_point_x_averages_values():
    interp = IDWInterpolation(epsilon=1e-6)
    interp.update(1.0, 3.0)
    interp.update(1.0, 7.0)

    assert interp.predict(2.0) == pytest.approx(5.0)


def test_idw_interpolation_move_shifts_observed_values():
    interp = IDWInterpolation()
    interp.update(1.0, 3.0)
    interp.update(3.0, 7.0)

    interp.move(2.0)

    assert interp.predict(2.0) == pytest.approx(7.0)


def test_idw_interpolation_uses_nearest_k_for_weighted_regression():
    interp = IDWInterpolation(k=3)
    interp.update(0.0, 0.0)
    interp.update(1.0, 10.0)
    interp.update(2.0, 20.0)
    interp.update(100.0, -1000.0)

    assert interp.predict(1.5) == pytest.approx(15.0)
