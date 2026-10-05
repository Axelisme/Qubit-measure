"""Observable joint-cloud capture, filtering and index-based projection."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.analysis.fluxdep.cross_selection import (
    CrossSelectionBackground,
    CrossSelectionChange,
    CrossSelectionInputs,
    analyze_cross_selection,
    fill_cross_selection_state,
    make_cross_selection_state,
    project_cross_selection,
    set_cross_selection_distance,
    set_cross_selection_tool,
    stroke_cross_selection_state,
)
from zcu_tools.analysis.fluxdep.processing import downsample_points
from zcu_tools.analysis.fluxdep.stroke import BrushPoint, BrushStroke, BrushTool


@pytest.fixture
def inputs():
    return CrossSelectionInputs(
        np.array([0.0, 0.01, 0.5, 1.0, 0.5]),
        np.array([4.0, 4.01, 4.5, 5.0, 4.5]),
        (),
        (0.0, 1.0),
        (4.0, 5.0),
    )


def test_capture_detaches_cloud_and_descending_nonuniform_background():
    fluxs = np.array([1.0, 0.4, 0.0])
    freqs = np.array([5.0, 4.7, 4.0])
    real = np.array([[0.0, np.nan, 1.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    background = CrossSelectionBackground("source", fluxs, freqs, real)
    cloud = np.array([0.0, 1.0])
    captured = CrossSelectionInputs(
        cloud, np.array([4.0, 5.0]), (background,), (0.0, 1.0), (4.0, 5.0)
    )
    fluxs[:] = 8.0
    freqs[:] = 8.0
    real[:] = 8.0
    cloud[:] = 8.0
    np.testing.assert_array_equal(captured.fluxs, [0.0, 1.0])
    np.testing.assert_array_equal(captured.backgrounds[0].fluxs, [1.0, 0.4, 0.0])
    assert np.isnan(captured.backgrounds[0].real_signals[0, 1])
    for array in (
        captured.fluxs,
        captured.freqs,
        captured.backgrounds[0].real_signals,
        captured.backgrounds[0].fluxs,
        captured.backgrounds[0].freqs,
    ):
        assert not array.flags.writeable


@pytest.mark.parametrize(
    "bad",
    [
        np.array([]),
        np.array([[0.0]]),
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([True]),
        np.array([1], dtype=np.int64),
    ],
)
def test_capture_rejects_invalid_cloud(bad):
    with pytest.raises(ValueError, match="cloud|fluxs|freqs"):
        CrossSelectionInputs(bad, bad.copy(), (), (0.0, 1.0), (4.0, 5.0))


@pytest.mark.parametrize(
    "axis",
    [
        np.array([0.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 1.0, 0.5]),
        np.array([0.0, np.inf]),
    ],
)
def test_background_rejects_invalid_axes(axis):
    with pytest.raises(ValueError, match="axis|flux|monotone"):
        CrossSelectionBackground(
            "source", axis, np.array([4.0, 5.0]), np.ones((axis.size, 2))
        )


def test_capture_preserves_masked_background_missing_samples():
    real = np.ma.array(np.ones((2, 2)), mask=[[True, False], [False, False]])
    background = CrossSelectionBackground(
        "masked", np.array([0.0, 1.0]), np.array([4.0, 5.0]), real
    )
    real[:] = 8.0
    assert np.isnan(background.real_signals[0, 0])
    assert background.real_signals[1, 1] == 1.0


@pytest.mark.parametrize(
    "real", [np.full((2, 2), np.inf), np.ones((2, 3)), np.ones((2, 2), dtype=int)]
)
def test_background_rejects_invalid_real_matrix(real):
    with pytest.raises(ValueError, match="real_signals"):
        CrossSelectionBackground(
            "bad", np.array([0.0, 1.0]), np.array([4.0, 5.0]), real
        )


@pytest.mark.parametrize(
    "bound", [(0.0, 0.0), (1.0, 0.0), (0.0, np.inf), (False, 1.0), (0.1, 1.0)]
)
def test_capture_rejects_nonpositive_nonfinite_or_non_enclosing_bounds(bound):
    with pytest.raises(ValueError, match="flux_bound"):
        CrossSelectionInputs(
            np.array([0.0, 1.0]), np.array([4.0, 5.0]), (), bound, (4.0, 5.0)
        )


def test_full_mask_downsampling_preserves_order_and_duplicate_identity(inputs):
    seed = make_cross_selection_state(inputs)
    state = set_cross_selection_distance(inputs, seed, 0.1)
    result = analyze_cross_selection(inputs, state)
    expected = downsample_points(inputs.fluxs, inputs.freqs - 4.0, 0.1)
    np.testing.assert_array_equal(result.selected, expected)
    np.testing.assert_array_equal(result.fluxs, inputs.fluxs)
    np.testing.assert_array_equal(result.freqs, inputs.freqs)
    assert result.selected.shape == (5,)
    assert result.selected[2] and result.selected[4]
    assert result.min_distance == 0.1
    np.testing.assert_array_equal(
        analyze_cross_selection(inputs, state).selected, result.selected
    )
    assert seed.selected.all()


def test_stroke_clear_fill_distance_and_inverse_project(inputs):
    seed = make_cross_selection_state(inputs)
    gesture = BrushStroke((BrushPoint(0.5, 4.5),), 0.0, "erase")
    erased = stroke_cross_selection_state(inputs, seed, gesture)
    np.testing.assert_array_equal(erased.selected, [True, True, False, True, False])
    assert seed.selected.all()
    view = project_cross_selection(inputs, erased)
    np.testing.assert_array_equal(view.removed_points, [[0.5, 4.5], [0.5, 4.5]])
    assert view.added_points.shape == (0, 2)
    assert view.stroke_vertices == gesture.vertices
    inverse = project_cross_selection(inputs, seed, previous=erased)
    np.testing.assert_array_equal(inverse.added_points, [[0.5, 4.5], [0.5, 4.5]])
    assert inverse.removed_points.shape == (0, 2)
    cleared = fill_cross_selection_state(inputs, erased, select=False)
    result = analyze_cross_selection(inputs, cleared)
    assert not result.selected.any()
    assert result.selected.size == 5
    restored = fill_cross_selection_state(inputs, cleared, select=True)
    assert restored.selected.all()
    changed = set_cross_selection_distance(inputs, restored, 0.1)
    assert project_cross_selection(inputs, changed).removed_points.size > 0


@pytest.mark.parametrize("mode,select", [("unknown", 1)])
def test_invalid_mode_seed_fill_and_before_image_fail_atomically(inputs, mode, select):
    seed = make_cross_selection_state(inputs)
    with pytest.raises(ValueError, match="mode"):
        make_cross_selection_state(inputs, mode=mode)
    with pytest.raises(ValueError, match="select"):
        fill_cross_selection_state(inputs, seed, select=select)
    with pytest.raises(ValueError, match="min_distance"):
        analyze_cross_selection(
            inputs,
            replace(seed, last_change=CrossSelectionChange(seed.selected.copy(), True)),
        )
    with pytest.raises(ValueError, match="mode"):
        stroke_cross_selection_state(
            inputs, seed, BrushStroke((BrushPoint(0.5, 4.5),), 0.05, mode)
        )
    assert seed.selected.all() and seed.last_change is None


def test_tool_update_keeps_before_image_and_projection_detaches(inputs):
    state = stroke_cross_selection_state(
        inputs,
        make_cross_selection_state(inputs),
        BrushStroke((BrushPoint(0.5, 4.5),), 0.05, "erase"),
    )
    updated = set_cross_selection_tool(inputs, state, BrushTool(0.08, "select"))
    np.testing.assert_array_equal(updated.selected, state.selected)
    assert updated.last_change is not None and state.last_change is not None
    np.testing.assert_array_equal(
        updated.last_change.selected, state.last_change.selected
    )
    assert updated.last_change.vertices == state.last_change.vertices
    assert (updated.width, updated.mode) == (0.08, "select")
    view = project_cross_selection(inputs, updated)
    view.state.selected[:] = False
    view.result.selected[:] = False
    assert updated.selected.any()


@pytest.mark.parametrize("value", [-0.01, 0.101, np.inf, np.nan, True, 10**400])
def test_invalid_distance_is_atomic(inputs, value):
    state = make_cross_selection_state(inputs)
    with pytest.raises(ValueError, match="min_distance"):
        set_cross_selection_distance(inputs, state, value)
    assert state.selected.all()
    assert state.last_change is None


@pytest.mark.parametrize(
    "tool",
    [
        BrushTool(),
        BrushTool(-0.01),
        BrushTool(0.11),
        BrushTool(np.nan),
        BrushTool(True),
    ],
)
def test_invalid_tool_is_atomic(inputs, tool):
    state = make_cross_selection_state(inputs)
    with pytest.raises(ValueError, match="width|tool"):
        set_cross_selection_tool(inputs, state, tool)
    assert state.selected.all()


@pytest.mark.parametrize(
    "gesture",
    [
        BrushStroke((), 0.05, "erase"),
        BrushStroke((BrushPoint(np.nan, 4.0),), 0.05, "erase"),
        BrushStroke((BrushPoint(0.0, 4.0), BrushPoint(1.0, 5.0)), 0.0, "erase"),
        BrushStroke((BrushPoint(0.0, 4.0), BrushPoint(1.0, 5.0)), 0.000001, "erase"),
    ],
)
def test_invalid_gesture_is_atomic_including_sample_budget(inputs, gesture):
    state = make_cross_selection_state(inputs)
    with pytest.raises(ValueError, match="stroke|width|vertices|sample|finite|radius"):
        stroke_cross_selection_state(inputs, state, gesture)
    assert state.selected.all()
    assert state.last_change is None


@pytest.mark.parametrize(
    "mask",
    [
        np.array([True]),
        np.ones((5, 1), dtype=bool),
        np.ones(5, dtype=float),
    ],
)
def test_analysis_rejects_invalid_complete_mask(inputs, mask):
    with pytest.raises(ValueError, match="selected|mask"):
        analyze_cross_selection(
            inputs, replace(make_cross_selection_state(inputs), selected=mask)
        )
