"""mcp_server-side optimistic-concurrency bookkeeping (Phase 94).

Drives send_gui_rpc with the socket layer mocked: _send_line synchronously
delivers a per-method crafted reply via _deliver_reply, so we assert the mcp
policy (attach expected_versions for guarded ops, translate a stale rejection,
refresh _LAST_SEEN after every successful RPC) without a real GUI.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.services.remote import mcp_server


@pytest.fixture()
def wired(monkeypatch):
    """Wire send_gui_rpc to a synchronous in-memory responder keyed by method.

    Returns a dict you populate as ``{method: reply_envelope}``; ``sent`` records
    every outgoing ``(method, params)`` so tests can assert what was attached.
    """
    monkeypatch.setattr(mcp_server, "_GUI_SOCK", MagicMock())
    monkeypatch.setattr(mcp_server, "_LAST_SEEN", {}, raising=False)
    replies: Dict[str, Dict[str, Any]] = {}
    sent: list[tuple[str, Dict[str, Any]]] = []

    def fake_send_line(payload):
        method = payload["method"]
        sent.append((method, payload["params"]))
        resp = dict(replies.get(method, {"ok": True, "result": {}}))
        resp["id"] = payload["id"]
        mcp_server._deliver_reply(resp)

    monkeypatch.setattr(mcp_server, "_send_line", fake_send_line)
    replies["sent"] = sent  # type: ignore[assignment]
    return replies


def _versions_reply(table: Dict[str, int]) -> Dict[str, Any]:
    return {"ok": True, "result": {"versions": table}}


def test_guarded_op_attaches_expected_versions(wired):
    sent = wired["sent"]
    # Baseline the agent has observed.
    mcp_server._LAST_SEEN.update(
        {"tab:t:cfg": 3, "tab:t": 1, "soc": 2, "context": 4, "device:yoko": 5}
    )
    wired["run.start"] = {"ok": True, "result": {}}
    wired["resources.versions"] = _versions_reply(dict(mcp_server._LAST_SEEN))

    mcp_server.send_gui_rpc("run.start", {"tab_id": "t"})

    run_params = next(p for (m, p) in sent if m == "run.start")
    assert run_params["expected_versions"] == {
        "tab:t:cfg": 3,
        "tab:t": 1,
        "soc": 2,
        "context": 4,
        "device:yoko": 5,
    }


def test_save_depends_on_result_and_save_path_not_cfg(wired):
    sent = wired["sent"]
    mcp_server._LAST_SEEN.update(
        {"tab:t:result": 7, "tab:t:save_path": 2, "tab:t:cfg": 9}
    )
    wired["save.data"] = {"ok": True, "result": {}}
    wired["resources.versions"] = _versions_reply(dict(mcp_server._LAST_SEEN))

    mcp_server.send_gui_rpc("save.data", {"tab_id": "t"})

    params = next(p for (m, p) in sent if m == "save.data")
    assert params["expected_versions"] == {"tab:t:result": 7, "tab:t:save_path": 2}
    assert "tab:t:cfg" not in params["expected_versions"]


def test_unguarded_op_attaches_nothing(wired):
    sent = wired["sent"]
    wired["tab.snapshot"] = {"ok": True, "result": {"x": 1}}
    wired["resources.versions"] = _versions_reply({})

    mcp_server.send_gui_rpc("tab.snapshot", {"tab_id": "t"})

    params = next(p for (m, p) in sent if m == "tab.snapshot")
    assert "expected_versions" not in params


def test_stale_rejection_translated_and_refreshes(wired):
    mcp_server._LAST_SEEN.update({"tab:t:cfg": 3, "tab:t": 1, "soc": 1, "context": 1})
    wired["run.start"] = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "reason": "stale_version",
            "message": "stale",
        },
    }
    # After rejection the bridge re-reads the table; the human's edit bumped cfg.
    wired["resources.versions"] = _versions_reply({"tab:t:cfg": 4})

    with pytest.raises(RuntimeError) as ei:
        mcp_server.send_gui_rpc("run.start", {"tab_id": "t"})

    msg = str(ei.value)
    assert "PRECONDITION_FAILED" in msg
    # No raw version numbers leak into the agent-facing message.
    assert "4" not in msg and "tab:t:cfg" not in msg
    # _LAST_SEEN was resynced from the post-rejection read.
    assert mcp_server._LAST_SEEN == {"tab:t:cfg": 4}


def test_successful_rpc_refreshes_last_seen(wired):
    wired["state.has_soc"] = {"ok": True, "result": {"value": True}}
    wired["resources.versions"] = _versions_reply({"soc": 9, "context": 1})

    mcp_server.send_gui_rpc("state.has_soc", {})

    assert mcp_server._LAST_SEEN == {"soc": 9, "context": 1}


def test_device_glob_expands_to_all_device_keys(wired):
    sent = wired["sent"]
    mcp_server._LAST_SEEN.update(
        {
            "tab:t:cfg": 1,
            "tab:t": 1,
            "soc": 1,
            "context": 1,
            "device:yoko": 2,
            "device:sgs": 3,
        }
    )
    wired["run.start"] = {"ok": True, "result": {}}
    wired["resources.versions"] = _versions_reply(dict(mcp_server._LAST_SEEN))

    mcp_server.send_gui_rpc("run.start", {"tab_id": "t"})

    expected = next(p for (m, p) in sent if m == "run.start")["expected_versions"]
    assert expected["device:yoko"] == 2
    assert expected["device:sgs"] == 3
