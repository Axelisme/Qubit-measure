"""Polyline selection behavior observed through the analysis kernel facade."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.analysis.fluxdep import (
    BrushPoint,
    apply_mask_stroke,
    points_in_normalized_brush,
    points_in_normalized_stroke,
    toggle_near_mask,
)


@pytest.mark.parametrize("select", [True, False])
def test_single_vertex_matches_existing_mask_brush(select: bool) -> None:
    xs = np.linspace(-5.0, 5.0, 41)
    ys = np.linspace(4.0, 5.0, 31)
    actual = np.full((xs.size, ys.size), not select, dtype=bool)
    expected = actual.copy()
    before_xs, before_ys = xs.copy(), ys.copy()
    apply_mask_stroke(xs, ys, actual, [BrushPoint(0, 4.5)], 0.2, select=select)
    toggle_near_mask(xs, ys, expected, 0, 4.5, 0.2, select)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(xs, before_xs)
    np.testing.assert_array_equal(ys, before_ys)


def test_single_vertex_matches_existing_point_brush_without_mutation() -> None:
    xs = np.array([-1.0, 0.0, 1.0])
    ys = np.array([4.5, 4.5, 4.5])
    before_xs, before_ys = xs.copy(), ys.copy()
    actual = points_in_normalized_stroke(
        xs,
        ys,
        stroke=[BrushPoint(0, 4.5)],
        width=0.2,
        x_bound=(-5, 5),
        y_bound=(4, 5),
    )
    expected = points_in_normalized_brush(
        xs,
        ys,
        x=0,
        y=4.5,
        width=0.2,
        x_bound=(-5, 5),
        y_bound=(4, 5),
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(xs, before_xs)
    np.testing.assert_array_equal(ys, before_ys)


def test_segment_fills_middle_and_erase_removes_same_region() -> None:
    xs = np.linspace(0, 10, 101)
    ys = np.linspace(4, 5, 101)
    mask = np.zeros((101, 101), dtype=bool)
    stroke = [BrushPoint(2, 4.5), BrushPoint(8, 4.5)]
    apply_mask_stroke(xs, ys, mask, stroke, 0.02, select=True)
    assert mask[20:81, 50].all()
    assert not mask[50, 80]
    assert not mask[0, 50]
    apply_mask_stroke(xs, ys, mask, stroke, 0.02, select=False)
    assert not mask.any()


def test_turn_and_duplicate_vertex_cover_both_segments() -> None:
    xs = np.array([2.0, 5.0, 8.0, 8.0, 5.0])
    ys = np.array([4.2, 4.2, 4.5, 4.8, 4.5])
    stroke = [
        BrushPoint(2, 4.2),
        BrushPoint(8, 4.2),
        BrushPoint(8, 4.2),
        BrushPoint(8, 4.8),
    ]
    mask = points_in_normalized_stroke(
        xs,
        ys,
        stroke=stroke,
        width=0.05,
        x_bound=(0, 10),
        y_bound=(4, 5),
    )
    np.testing.assert_array_equal(mask, [True, True, True, True, False])
    reverse = points_in_normalized_stroke(
        xs,
        ys,
        stroke=list(reversed(stroke)),
        width=0.05,
        x_bound=(10, 0),
        y_bound=(5, 4),
    )
    np.testing.assert_array_equal(reverse, mask)


def test_outside_vertices_can_cross_the_supplied_grid() -> None:
    xs = np.linspace(0, 1, 21)
    ys = np.linspace(0, 1, 21)
    mask = np.zeros((21, 21), dtype=bool)
    apply_mask_stroke(
        xs, ys, mask, [BrushPoint(-1, 0.5), BrushPoint(2, 0.5)], 0.1, select=True
    )
    assert mask[:, 10].all()
    assert not mask[:, 0].any()


@pytest.mark.parametrize("width", [-0.1, float("nan"), float("inf")])
def test_invalid_width_rejected_before_mutation(width: float) -> None:
    xs = np.linspace(0, 1, 5)
    mask = np.zeros((5, 5), dtype=bool)
    with pytest.raises(ValueError, match="width"):
        apply_mask_stroke(xs, xs, mask, [BrushPoint(0.5, 0.5)], width, select=True)
    assert not mask.any()


@pytest.mark.parametrize(
    "stroke",
    [
        [],
        [BrushPoint(0.5, 0.5), BrushPoint(float("nan"), 0.5)],
        [BrushPoint(float("inf"), 0.5)],
    ],
)
def test_invalid_stroke_rejected_before_mutation(stroke: list[BrushPoint]) -> None:
    xs = np.linspace(0, 1, 5)
    mask = np.zeros((5, 5), dtype=bool)
    with pytest.raises(ValueError, match="stroke"):
        apply_mask_stroke(xs, xs, mask, stroke, 0.1, select=True)
    assert not mask.any()


def test_zero_width_stationary_stroke_and_moving_stroke_rejection() -> None:
    xs = np.array([0.0, 0.5, 1.0])
    mask = np.zeros((3, 3), dtype=bool)
    apply_mask_stroke(xs, xs, mask, [BrushPoint(0.5, 0.5)] * 2, 0, select=True)
    expected = np.zeros((3, 3), dtype=bool)
    expected[1, 1] = True
    np.testing.assert_array_equal(mask, expected)
    with pytest.raises(ValueError, match="width"):
        apply_mask_stroke(
            xs, xs, mask, [BrushPoint(0, 0), BrushPoint(1, 1)], 0, select=False
        )
    np.testing.assert_array_equal(mask, expected)


def test_invalid_axes_and_mask_shapes_rejected_before_mutation() -> None:
    xs = np.array([0.0, 1.0])
    mask = np.zeros((2, 2), dtype=bool)
    for bad_axis in (np.array([]), np.array([1.0, 1.0]), np.array([0.0, float("nan")])):
        with pytest.raises(ValueError, match="axis|axes|shape|span|finite|dev_values"):
            apply_mask_stroke(bad_axis, xs, mask, [BrushPoint(0, 0)], 0.1, select=True)
        assert not mask.any()
    with pytest.raises(ValueError, match="shape"):
        apply_mask_stroke(
            xs, xs, np.zeros((2, 3), dtype=bool), [BrushPoint(0, 0)], 0.1, select=True
        )


def test_point_shapes_nonfinite_values_and_bounds_fail_fast() -> None:
    xs = np.array([0.0, 1.0])
    for ys in (np.array([0.0]), np.array([[0.0, 1.0]]), np.array([0.0, float("nan")])):
        with pytest.raises(ValueError, match="shape|1D|finite"):
            points_in_normalized_stroke(
                xs,
                ys,
                stroke=[BrushPoint(0, 0)],
                width=0.1,
                x_bound=(0, 1),
                y_bound=(0, 1),
            )
    for bound in ((0.0, 0.0), (0.0, float("inf"))):
        with pytest.raises(ValueError, match="x_bound"):
            points_in_normalized_stroke(
                xs,
                xs,
                stroke=[BrushPoint(0, 0)],
                width=0.1,
                x_bound=bound,
                y_bound=(0, 1),
            )


def test_empty_point_cloud_returns_empty_membership_for_valid_request() -> None:
    xs = np.array([], dtype=np.float64)
    result = points_in_normalized_stroke(
        xs,
        xs,
        stroke=[BrushPoint(0, 0)],
        width=0.1,
        x_bound=(0, 1),
        y_bound=(0, 1),
    )
    assert result.dtype == np.bool_
    assert result.shape == (0,)
