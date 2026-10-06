"""TwoTone actions and command callers share atomic selection and history."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
from zcu_tools.analysis.fluxdep.processing import (
    cast2real_and_norm,
    spectrum2d_findpoint,
)
from zcu_tools.analysis.fluxdep.stroke import (
    BrushMode,
    BrushPoint,
    BrushStroke,
    BrushTool,
    apply_mask_stroke,
)
from zcu_tools.analysis.fluxdep.twotone import (
    TwoToneInputs,
    TwoTonePickState,
    TwoToneSettings,
    analyze_twotone_pick,
    project_twotone_pick,
)
from zcu_tools.gui.app.fluxdep.twotone import TwoTonePickPlugin
from zcu_tools.gui.expected_error import (
    ExpectedErrorCategory,
    FailedPreconditionError,
    InvalidInputError,
)
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler


def assert_same_state(actual: TwoTonePickState, expected: TwoTonePickState) -> None:
    np.testing.assert_array_equal(actual.mask, expected.mask)
    assert (
        actual.threshold,
        actual.sigma,
        actual.smooth_method,
        actual.width,
        actual.mode,
    ) == (
        expected.threshold,
        expected.sigma,
        expected.smooth_method,
        expected.width,
        expected.mode,
    )
    if expected.last_change is None:
        assert actual.last_change is None
    else:
        assert actual.last_change is not None
        np.testing.assert_array_equal(
            actual.last_change.mask, expected.last_change.mask
        )
        assert (
            actual.last_change.threshold,
            actual.last_change.sigma,
            actual.last_change.smooth_method,
            actual.last_change.vertices,
            actual.last_change.width,
        ) == (
            expected.last_change.threshold,
            expected.last_change.sigma,
            expected.last_change.smooth_method,
            expected.last_change.vertices,
            expected.last_change.width,
        )


@pytest.mark.parametrize("mode", ["select", "erase"])
def test_command_and_typed_stroke_share_kernel_and_single_commit(
    twotone_inputs: TwoToneInputs, mode: BrushMode
) -> None:
    plugin = TwoTonePickPlugin(twotone_inputs)
    action_session = plugin.open(ManualOwnerScheduler())
    command_session = plugin.open(ManualOwnerScheduler())
    if mode == "select":
        plugin.clear.execute(action_session, None)
        plugin.clear.execute(command_session, None)
    vertices = (BrushPoint(-0.8, 4.6), BrushPoint(0.8, 5.0))
    payload = BrushStroke(vertices, 0.03, mode)
    before = action_session.snapshot()
    expected_mask = before.mask.copy()
    spectrum = twotone_inputs.spectrum
    apply_mask_stroke(
        spectrum.dev_values,
        spectrum.freqs,
        expected_mask,
        vertices,
        0.03,
        select=mode == "select",
    )
    observed: list[TwoTonePickState] = []
    action_session.subscribe(lambda: observed.append(action_session.snapshot()))
    plugin.stroke.execute(action_session, payload)
    plugin.execute_command(
        command_session,
        "stroke",
        {
            "vertices": [[point.x, point.y] for point in vertices],
            "width": 0.03,
            "mode": mode,
        },
    )
    assert len(observed) == 1
    assert_same_state(action_session.snapshot(), command_session.snapshot())
    np.testing.assert_array_equal(action_session.snapshot().mask, expected_mask)
    assert expected_mask.any()
    assert not expected_mask.all()
    result = analyze_twotone_pick(twotone_inputs, action_session.snapshot())
    other = analyze_twotone_pick(twotone_inputs, command_session.snapshot())
    np.testing.assert_array_equal(result.dev_values, other.dev_values)
    np.testing.assert_array_equal(result.freqs, other.freqs)
    assert_same_state(action_session.undo(), before)
    with pytest.raises(FailedPreconditionError):
        action_session.undo()


@pytest.mark.parametrize("tool_before_stroke", [False, True])
def test_tool_update_preserves_undo_snapshot(
    twotone_inputs: TwoToneInputs, tool_before_stroke: bool
) -> None:
    plugin = TwoTonePickPlugin(twotone_inputs)
    session = plugin.open(ManualOwnerScheduler())
    seed = session.snapshot()
    plugin.set_tool.execute(session, BrushTool(0.004, "erase"))
    assert not session.can_undo()
    plugin.stroke.execute(session, BrushStroke((BrushPoint(0.0, 4.8),), 0.02, "erase"))
    if not tool_before_stroke:
        plugin.execute_command(session, "set_tool", {"width": 0.05, "mode": "select"})
    restored = session.undo()
    np.testing.assert_array_equal(restored.mask, seed.mask)
    assert restored.width == 0.004
    assert restored.mode == "erase"
    assert not session.can_undo()
    if tool_before_stroke:
        assert session.snapshot().width == 0.004


def test_tools_without_history_and_pure_projection_never_create_history(
    twotone_inputs: TwoToneInputs,
) -> None:
    plugin = TwoTonePickPlugin(twotone_inputs)
    session = plugin.open(ManualOwnerScheduler())
    plugin.execute_command(session, "set_tool", {"width": 0.004})
    before = session.snapshot()
    view = project_twotone_pick(twotone_inputs, before)
    assert not session.can_undo()
    assert_same_state(session.snapshot(), before)
    assert view.added_points.shape == (0, 2)
    assert view.removed_points.shape == (0, 2)
    view.state.mask[:] = False
    assert session.snapshot().mask.all()


@pytest.mark.parametrize(
    "command",
    [
        {"threshold": 2.0},
        {"sigma": 2.0},
        {"sigma": 0.0, "smooth_method": "wavelet"},
        {"sigma": 0.001, "smooth_method": "gaussian"},
        {"smooth_method": "gaussian"},
        {"threshold": 1.5, "sigma": 0.5, "smooth_method": "gaussian"},
    ],
)
def test_detector_updates_use_existing_numerical_owner_and_undo(
    twotone_inputs: TwoToneInputs, command: Mapping[str, object]
) -> None:
    plugin = TwoTonePickPlugin(twotone_inputs)
    session = plugin.open(ManualOwnerScheduler())
    before = session.snapshot()
    plugin.execute_command(session, "set_settings", command)
    state = session.snapshot()
    result = analyze_twotone_pick(twotone_inputs, state)
    spectrum = twotone_inputs.spectrum
    expected_real = cast2real_and_norm(
        spectrum.signals, sigma=state.sigma, smooth_method=state.smooth_method
    )
    devs, freqs = spectrum2d_findpoint(
        spectrum.dev_values,
        spectrum.freqs,
        expected_real,
        state.threshold,
        weight=state.mask,
    )
    order = np.argsort(devs)
    np.testing.assert_allclose(result.real_signals, expected_real)
    np.testing.assert_array_equal(result.dev_values, devs[order])
    np.testing.assert_array_equal(result.freqs, freqs[order])
    assert_same_state(session.undo(), before)


def test_fill_clear_and_positional_changes_are_not_net_counts(
    twotone_inputs: TwoToneInputs,
) -> None:
    plugin = TwoTonePickPlugin(twotone_inputs)
    session = plugin.open(ManualOwnerScheduler())
    original = analyze_twotone_pick(twotone_inputs, session.snapshot())
    assert original.dev_values.size > 0
    plugin.clear.execute(session, None)
    view = project_twotone_pick(twotone_inputs, session.snapshot())
    assert view.result.dev_values.size == 0
    assert view.mask_removed == session.snapshot().mask.size
    assert view.removed_points.shape == (original.dev_values.size, 2)
    plugin.execute_command(session, "set_tool", {"mode": "select"})
    plugin.perform_on_all.execute(session, None)
    selected = project_twotone_pick(twotone_inputs, session.snapshot())
    assert selected.mask_added == session.snapshot().mask.size
    assert selected.added_points.shape == (original.dev_values.size, 2)
    np.testing.assert_array_equal(selected.result.dev_values, original.dev_values)
    np.testing.assert_array_equal(selected.result.freqs, original.freqs)
    plugin.execute_command(session, "set_tool", {"mode": "erase"})
    plugin.execute_command(session, "perform_on_all", {})
    assert not session.snapshot().mask.any()
    session.undo()
    assert session.snapshot().mask.all()


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("set_settings", {}),
        ("set_settings", {"threshold": None}),
        ("set_settings", {"threshold": True}),
        ("set_settings", {"threshold": 0}),
        ("set_settings", {"threshold": 21}),
        ("set_settings", {"threshold": float("nan")}),
        ("set_settings", {"threshold": float("inf")}),
        ("set_settings", {"threshold": 10**1000}),
        ("set_settings", {"sigma": -1}),
        ("set_settings", {"sigma": 6}),
        ("set_settings", {"sigma": 0.0, "smooth_method": "gaussian"}),
        ("set_settings", {"sigma": 1e-300, "smooth_method": "gaussian"}),
        ("set_settings", {"smooth_method": "other"}),
        ("set_settings", {"unknown": 1}),
        ("set_tool", {}),
        ("set_tool", {"width": -1}),
        ("set_tool", {"width": 0.2}),
        ("set_tool", {"width": True}),
        ("set_tool", {"mode": "other"}),
        ("stroke", {"vertices": [], "width": 0.03, "mode": "erase"}),
        ("stroke", {"vertices": [[0.0, 4.8]], "width": 0.03}),
        ("stroke", {"vertices": [[True, 4.8]], "width": 0.03, "mode": "erase"}),
        ("stroke", {"vertices": [[0.0, float("inf")]], "width": 0.03, "mode": "erase"}),
        (
            "stroke",
            {"vertices": [[-1.0, 4.8], [1.0, 4.8]], "width": 1e-9, "mode": "erase"},
        ),
        (
            "stroke",
            {"vertices": [[-1.0, 4.8], [1.0, 4.8]], "width": 0.0, "mode": "erase"},
        ),
        ("clear", {"unknown": 1}),
        ("perform_on_all", {"mode": "erase"}),
        ("unknown", {}),
    ],
)
def test_invalid_commands_preserve_full_state_and_history(
    twotone_inputs: TwoToneInputs, name: str, params: Mapping[str, object]
) -> None:
    plugin = TwoTonePickPlugin(twotone_inputs)
    session = plugin.open(ManualOwnerScheduler())
    seed = session.snapshot()
    plugin.clear.execute(session, None)
    before = session.snapshot()
    with pytest.raises(InvalidInputError) as error:
        plugin.execute_command(session, name, params)
    assert error.value.category is ExpectedErrorCategory.INVALID_INPUT
    assert_same_state(session.snapshot(), before)
    assert_same_state(session.undo(), seed)


@pytest.mark.parametrize(
    "value", [True, float("nan"), float("inf"), -1.0, 21.0, 10**100, 10**1000]
)
def test_typed_detector_action_rejects_without_commit(
    twotone_inputs: TwoToneInputs, value: float
) -> None:
    plugin = TwoTonePickPlugin(twotone_inputs)
    session = plugin.open(ManualOwnerScheduler())
    before = session.snapshot()
    with pytest.raises(InvalidInputError):
        plugin.set_settings.execute(session, TwoToneSettings(threshold=value))
    assert_same_state(session.snapshot(), before)
    assert not session.can_undo()
