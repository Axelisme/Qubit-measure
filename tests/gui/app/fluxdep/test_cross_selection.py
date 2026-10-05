"""Typed and command joint-cloud actions commit through the same Session."""

from collections.abc import Mapping

import numpy as np
import pytest
from zcu_tools.analysis.fluxdep.cross_selection import (
    CrossSelectionInputs,
    analyze_cross_selection,
    project_cross_selection,
)
from zcu_tools.analysis.fluxdep.stroke import BrushPoint, BrushStroke, BrushTool
from zcu_tools.gui.app.fluxdep.cross_selection import CrossSelectionPlugin
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler


@pytest.fixture
def plugin():
    return CrossSelectionPlugin(
        CrossSelectionInputs(
            np.array([0.0, 0.5, 1.0, 0.5]),
            np.array([4.0, 4.5, 5.0, 4.5]),
            (),
            (0.0, 1.0),
            (4.0, 5.0),
        )
    )


def test_command_and_typed_stroke_commit_once_with_single_undo(plugin):
    typed = plugin.open(ManualOwnerScheduler())
    command = plugin.open(ManualOwnerScheduler())
    notifications = []
    typed.subscribe(lambda: notifications.append(typed.snapshot()))
    plugin.stroke.execute(typed, BrushStroke((BrushPoint(0.5, 4.5),), 0.0, "erase"))
    plugin.execute_command(
        command, "stroke", {"vertices": [[0.5, 4.5]], "width": 0.0, "mode": "erase"}
    )
    assert len(notifications) == 1
    np.testing.assert_array_equal(typed.snapshot().selected, [True, False, True, False])
    np.testing.assert_array_equal(
        command.snapshot().selected, typed.snapshot().selected
    )
    assert (typed.snapshot().width, typed.snapshot().mode) == (0.0, "erase")
    assert typed.undo().selected.all()
    with pytest.raises(FailedPreconditionError):
        typed.undo()


def test_tool_preserves_undo_and_reads_do_not_consume_it(plugin):
    session = plugin.open(ManualOwnerScheduler())
    plugin.set_tool.execute(session, BrushTool(0.02, "erase"))
    assert not session.can_undo()
    plugin.clear.execute(session, None)
    plugin.execute_command(session, "set_tool", {"width": 0.09, "mode": "select"})
    view = project_cross_selection(plugin.inputs, session.snapshot())
    assert view.removed_points.shape == (4, 2)
    assert session.can_undo()
    restored = session.undo()
    assert restored.selected.all()
    assert (restored.width, restored.mode) == (0.02, "erase")
    assert not session.can_undo()


def test_distance_perform_all_and_empty_are_nonterminal(plugin):
    session = plugin.open(ManualOwnerScheduler())
    plugin.execute_command(session, "set_min_distance", {"min_distance": 0.1})
    assert session.snapshot().min_distance == 0.1
    assert session.undo().min_distance == 0.0
    plugin.execute_command(session, "set_tool", {"mode": "erase"})
    plugin.perform_on_all.execute(session, None)
    assert not analyze_cross_selection(plugin.inputs, session.snapshot()).selected.any()
    plugin.execute_command(session, "set_tool", {"mode": "select"})
    plugin.execute_command(session, "perform_on_all", {})
    assert session.snapshot().selected.all()


@pytest.mark.parametrize(
    ("command", "params"),
    [
        ("set_min_distance", {}),
        ("set_min_distance", {"min_distance": True}),
        ("set_min_distance", {"min_distance": -0.1}),
        ("set_min_distance", {"min_distance": float("nan")}),
        ("set_tool", {}),
        ("set_tool", {"width": None, "mode": None}),
        ("set_tool", {"width": 0.2}),
        ("set_tool", {"mode": "unknown"}),
        ("clear", {"unexpected": 1}),
        ("stroke", {"vertices": [], "width": 0.05, "mode": "erase"}),
        (
            "stroke",
            {"vertices": [[0.0, 4.0], [1.0, 5.0]], "width": 0.0, "mode": "erase"},
        ),
        (
            "stroke",
            {"vertices": [[0.0, 4.0], [1.0, 5.0]], "width": 0.000001, "mode": "erase"},
        ),
    ],
)
def test_invalid_command_preserves_state_and_history(
    plugin, command: str, params: Mapping[str, object]
):
    session = plugin.open(ManualOwnerScheduler())
    plugin.clear.execute(session, None)
    before = session.snapshot()
    with pytest.raises(InvalidInputError):
        plugin.execute_command(session, command, params)
    np.testing.assert_array_equal(session.snapshot().selected, before.selected)
    assert session.snapshot().min_distance == before.min_distance
    assert session.undo().selected.all()


def test_closed_input_rejects_commands_and_typed_actions(plugin):
    session = plugin.open(ManualOwnerScheduler())
    session.close_input()
    with pytest.raises(FailedPreconditionError):
        plugin.clear.execute(session, None)
    with pytest.raises(FailedPreconditionError):
        plugin.execute_command(session, "set_min_distance", {"min_distance": 0.1})
