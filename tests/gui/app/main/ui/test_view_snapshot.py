"""Tests for MainWindow.get_view_snapshot() State-SSOT invariant.

The key invariant: tab_ids in the view snapshot must equal State's list_tab_ids(),
never _tab_widgets.keys(), so ghost widget entries cannot leak into the projection
(ADR-0013: view is a reader of State, not a second source of truth).

MainWindow inherits from QMainWindow (a Qt C++ type) so object.__new__ is
forbidden. Instead we invoke ``MainWindow.get_view_snapshot`` as an unbound method
on a plain ``types.SimpleNamespace`` duck-typed shell that exposes only the
attributes the method reads. This tests the real production code path with zero Qt
widget construction.
"""

from __future__ import annotations

import types
from typing import Any
from unittest.mock import MagicMock

from zcu_tools.gui.app.main.services.remote.dialogs import DialogName
from zcu_tools.gui.app.main.ui.main_window import MainWindow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeQTabWidget:
    """Minimal stub of QTabWidget used only by get_view_snapshot."""

    def __init__(self, current: object | None = None, count: int = 0) -> None:
        self._current = current
        self._count = count

    def count(self) -> int:
        return self._count

    def currentWidget(self) -> object | None:
        return self._current


def _make_snapshot_shell(
    *,
    state_tab_ids: list[str],
    tab_widgets: dict[str, Any],
    current_widget: object | None = None,
    open_dialogs: list[DialogName] | None = None,
) -> Any:
    """Build a duck-typed shell that satisfies get_view_snapshot's attribute reads.

    Calls the real MainWindow.get_view_snapshot as an unbound method so the
    production code is exercised, not a copy. QMainWindow.__new__ is not used —
    we pass a plain SimpleNamespace instead.
    """
    ctrl = MagicMock()
    ctrl.list_tab_ids.return_value = list(state_tab_ids)

    tab_count = 1 if current_widget is not None else 0
    tabs_stub = _FakeQTabWidget(current=current_widget, count=tab_count)

    shell = types.SimpleNamespace(
        _ctrl=ctrl,
        _tab_widgets=tab_widgets,
        _tabs=tabs_stub,
        # Optional UI labels — snapshot guards with "if X" so None is safe.
        _ctx_label=None,
        _predictor_label=None,
        _status_bar=None,
        list_open_dialogs=lambda: list(open_dialogs or []),
    )
    return shell


def _snapshot(shell: Any) -> dict[str, object]:
    """Invoke the real get_view_snapshot on the duck-typed shell."""
    return MainWindow.get_view_snapshot(shell)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests: tab_ids == State SSOT
# ---------------------------------------------------------------------------


def test_tab_ids_equal_state_ssot_when_widgets_are_superset() -> None:
    """Ghost widget entry must NOT appear in tab_ids."""
    widget_a = object()
    widget_b = object()
    ghost_widget = object()  # in _tab_widgets but not in State

    shell = _make_snapshot_shell(
        state_tab_ids=["tab-A", "tab-B"],
        tab_widgets={"tab-A": widget_a, "tab-B": widget_b, "ghost-id": ghost_widget},
        current_widget=widget_a,
    )

    snap = _snapshot(shell)

    assert snap["tab_ids"] == ["tab-A", "tab-B"], (
        "ghost widget entry must be excluded from tab_ids"
    )


def test_tab_ids_empty_when_state_has_no_tabs() -> None:
    """When State has no tabs, tab_ids must be empty even if widgets exist."""
    ghost_widget = object()

    shell = _make_snapshot_shell(
        state_tab_ids=[],
        tab_widgets={"stale-id": ghost_widget},
        current_widget=ghost_widget,
    )

    snap = _snapshot(shell)

    assert snap["tab_ids"] == []


def test_tab_ids_order_follows_state() -> None:
    """tab_ids ordering must follow State.list_tab_ids, not insertion order of widgets."""
    widget_x = object()
    widget_y = object()
    # _tab_widgets inserted in reversed order relative to State
    shell = _make_snapshot_shell(
        state_tab_ids=["tab-X", "tab-Y"],
        tab_widgets={"tab-Y": widget_y, "tab-X": widget_x},
        current_widget=widget_x,
    )

    snap = _snapshot(shell)

    assert snap["tab_ids"] == ["tab-X", "tab-Y"]


# ---------------------------------------------------------------------------
# Tests: active_tab_id must be a State-known tab or None
# ---------------------------------------------------------------------------


def test_active_tab_id_is_none_when_current_widget_is_ghost() -> None:
    """If currentWidget maps to a ghost id (not in State), active_tab_id is None."""
    real_widget = object()
    ghost_widget = object()

    shell = _make_snapshot_shell(
        state_tab_ids=["tab-real"],
        # ghost-id is in _tab_widgets but NOT in State
        tab_widgets={"tab-real": real_widget, "ghost-id": ghost_widget},
        current_widget=ghost_widget,
    )

    snap = _snapshot(shell)

    assert snap["active_tab_id"] is None, (
        "active_tab_id must be None when currentWidget maps only to a ghost id"
    )


def test_active_tab_id_is_valid_when_current_widget_is_state_tab() -> None:
    """If currentWidget maps to a State-known tab, active_tab_id is that id."""
    real_widget = object()
    other_widget = object()

    shell = _make_snapshot_shell(
        state_tab_ids=["tab-A", "tab-B"],
        tab_widgets={"tab-A": real_widget, "tab-B": other_widget},
        current_widget=real_widget,
    )

    snap = _snapshot(shell)

    assert snap["active_tab_id"] == "tab-A"


def test_active_tab_id_is_none_when_no_tabs_in_widget() -> None:
    """When _tabs.count() == 0, active_tab_id must be None regardless of State."""
    shell = _make_snapshot_shell(
        state_tab_ids=["tab-A"],
        tab_widgets={"tab-A": object()},
        current_widget=None,  # triggers count==0 in _FakeQTabWidget
    )

    snap = _snapshot(shell)

    assert snap["active_tab_id"] is None


# ---------------------------------------------------------------------------
# Tests: snapshot structure completeness
# ---------------------------------------------------------------------------


def test_snapshot_contains_required_keys() -> None:
    """get_view_snapshot always returns the full key set regardless of content."""
    shell = _make_snapshot_shell(
        state_tab_ids=["tab-A"],
        tab_widgets={"tab-A": object()},
    )

    snap = _snapshot(shell)

    expected_keys = {
        "active_tab_id",
        "tab_ids",
        "context_label",
        "predictor_label",
        "status",
        "open_dialogs",
    }
    assert expected_keys == set(snap.keys())


def test_open_dialogs_come_from_named_dialog_facade() -> None:
    """Only the MainWindow named-dialog facade contributes snapshot dialog names."""
    shell = _make_snapshot_shell(
        state_tab_ids=["tab-A"],
        tab_widgets={"tab-A": object()},
        open_dialogs=[DialogName.PREDICTOR, DialogName.STARTUP],
    )

    snap = _snapshot(shell)

    assert snap["open_dialogs"] == ["predictor", "startup"]
