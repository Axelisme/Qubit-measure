"""mcp_server.send_gui_rpc surfaces the envelope's gui_changes sidecar.

Drives send_gui_rpc with the socket layer mocked: _send_line synchronously
delivers a crafted reply via _deliver_reply, so we assert the surfacing logic
(reserved _gui_changes key on success; appended to the error message on a
stale-guard block) without a real GUI.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.services.remote import mcp_server


@pytest.fixture()
def wired(monkeypatch):
    """Wire send_gui_rpc to a synchronous in-memory responder."""
    monkeypatch.setattr(mcp_server, "_GUI_SOCK", MagicMock())
    reply_holder = {}

    def fake_send_line(payload):
        rid = payload["id"]
        resp = dict(reply_holder["resp"])
        resp["id"] = rid
        mcp_server._deliver_reply(resp)

    monkeypatch.setattr(mcp_server, "_send_line", fake_send_line)
    return reply_holder


def test_ok_reply_surfaces_gui_changes(wired):
    changes = [{"category": "cfg_edited", "object_id": "editor-1", "count": 2}]
    wired["resp"] = {"ok": True, "result": {"value": 42}, "gui_changes": changes}
    out = mcp_server.send_gui_rpc("state.has_soc", {})
    assert out["value"] == 42
    assert out["_gui_changes"] == changes


def test_ok_reply_without_changes_has_no_sidecar(wired):
    wired["resp"] = {"ok": True, "result": {"value": 1}}
    out = mcp_server.send_gui_rpc("state.has_soc", {})
    assert "_gui_changes" not in out


def test_changes_poll_forwards_to_changes_poll_rpc(wired):
    changes = [{"category": "tab_changed", "object_id": "tab-1", "count": 1}]
    wired["resp"] = {"ok": True, "result": {}, "gui_changes": changes}
    out = mcp_server.TOOLS["gui_changes_poll"]["handler"]({})
    assert out["_gui_changes"] == changes


def test_changes_poll_empty_when_nothing_changed(wired):
    wired["resp"] = {"ok": True, "result": {}}
    out = mcp_server.TOOLS["gui_changes_poll"]["handler"]({})
    assert "_gui_changes" not in out


def test_error_reply_appends_changes_to_message(wired):
    changes = [{"category": "cfg_edited", "object_id": "editor-1", "count": 1}]
    wired["resp"] = {
        "ok": False,
        "error": {"code": "precondition_failed", "message": "tab stale"},
        "gui_changes": changes,
    }
    with pytest.raises(RuntimeError) as ei:
        mcp_server.send_gui_rpc("run.start", {"tab_id": "tab-1"})
    msg = str(ei.value)
    assert "precondition_failed" in msg
    assert "gui_changes" in msg
    assert "editor-1" in msg
