"""Tests for ViewQueryService — pure-read View projection delegation.

Cfg field editing is no longer here (ADR-0013 F11): agents edit a tab's cfg
through its CfgEditorService session (editor.set_field), so ViewQueryService is
snapshot/screenshot only.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from zcu_tools.gui.services.view_query import ViewQueryService


def _make_service() -> tuple[ViewQueryService, MagicMock]:
    view = MagicMock()
    svc = ViewQueryService(lambda: view)
    return svc, view


def test_snapshot_delegates_to_view():
    svc, view = _make_service()
    view.get_view_snapshot.return_value = {"active_tab_id": "t1"}

    assert svc.snapshot() == {"active_tab_id": "t1"}
    view.get_view_snapshot.assert_called_once()


def test_screenshot_delegates_with_tab_id():
    svc, view = _make_service()
    view.take_screenshot.return_value = b"PNG"

    assert svc.screenshot("t1") == b"PNG"
    view.take_screenshot.assert_called_once_with("t1")


def test_figure_screenshot_delegates():
    svc, view = _make_service()
    view.take_figure_screenshot.return_value = b"FIG"

    assert svc.figure_screenshot("t1") == b"FIG"
    view.take_figure_screenshot.assert_called_once_with("t1")
