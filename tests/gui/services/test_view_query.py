"""Tests for ViewQueryService — delegation and set_field fail-fast."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
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


def test_set_field_resolves_against_live_root():
    svc, view = _make_service()
    root = MagicMock()
    view.get_tab_live_model_root.return_value = root

    with pytest.MonkeyPatch.context() as mp:
        calls: list[tuple] = []
        mp.setattr(
            "zcu_tools.gui.services.remote.path_resolver.resolve_and_set",
            lambda r, p, v: calls.append((r, p, v)),
        )
        svc.set_field("t1", "section.field", 42)

    view.get_tab_live_model_root.assert_called_once_with("t1")
    assert calls == [(root, "section.field", 42)]


def test_set_field_propagates_form_not_populated():
    """When the form is not populated the view raises; the service must not swallow it."""
    svc, view = _make_service()
    view.get_tab_live_model_root.side_effect = RuntimeError(
        "tab 't1' cfg form has no live model yet"
    )

    with pytest.raises(RuntimeError, match="no live model yet"):
        svc.set_field("t1", "section.field", 42)
