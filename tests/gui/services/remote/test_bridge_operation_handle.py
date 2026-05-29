"""mcp_server-side async-operation handle bookkeeping (Phase 95).

Drives send_gui_rpc / the gui_device_wait_setup tool with the socket layer
mocked, asserting the mcp policy: a start op's operation_id is captured under its
semantic key (latest wins), and the semantic wait tool translates name ->
operation_id -> operation.await without the agent ever seeing the raw id.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.services.remote import mcp_server


@pytest.fixture()
def wired(monkeypatch):
    """Synchronous in-memory responder keyed by method; records what was sent."""
    monkeypatch.setattr(mcp_server, "_GUI_SOCK", MagicMock())
    monkeypatch.setattr(mcp_server, "_LAST_SEEN", {}, raising=False)
    monkeypatch.setattr(mcp_server, "_OP_BY_KEY", {}, raising=False)
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


def _versions(table: Dict[str, int]) -> Dict[str, Any]:
    return {"ok": True, "result": {"versions": table}}


def test_device_setup_captures_operation_id_and_strips_from_result(wired):
    wired["device.setup"] = {"ok": True, "result": {"operation_id": 42}}
    wired["resources.versions"] = _versions({})

    out = mcp_server.send_gui_rpc("device.setup", {"name": "flux", "updates": {}})

    assert mcp_server._OP_BY_KEY["device:flux"] == 42  # captured for await-by-name
    assert "operation_id" not in out  # never surfaced to the agent


def test_latest_setup_wins_for_same_device(wired):
    wired["resources.versions"] = _versions({})
    wired["device.setup"] = {"ok": True, "result": {"operation_id": 1}}
    mcp_server.send_gui_rpc("device.setup", {"name": "flux", "updates": {}})
    wired["device.setup"] = {"ok": True, "result": {"operation_id": 2}}
    mcp_server.send_gui_rpc("device.setup", {"name": "flux", "updates": {}})

    assert mcp_server._OP_BY_KEY["device:flux"] == 2


def test_wait_setup_translates_name_to_operation_await(wired):
    sent = wired["sent"]
    mcp_server._OP_BY_KEY["device:flux"] = 42
    wired["operation.await"] = {"ok": True, "result": {"status": "finished"}}
    wired["resources.versions"] = _versions({})

    out = mcp_server.TOOLS["gui_device_wait_setup"]["handler"]({"name": "flux"})

    await_params = next(p for (m, p) in sent if m == "operation.await")
    assert await_params["operation_id"] == 42  # name translated to id
    assert out["status"] == "finished"


def test_wait_setup_no_operation_when_key_unknown(wired):
    out = mcp_server.TOOLS["gui_device_wait_setup"]["handler"]({"name": "ghost"})
    assert out["status"] == "no_operation"


def test_wait_setup_propagates_failure(wired):
    mcp_server._OP_BY_KEY["device:flux"] = 7
    wired["operation.await"] = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "reason": "failed",
            "message": "hardware boom",
        },
    }
    wired["resources.versions"] = _versions({})

    with pytest.raises(RuntimeError, match="hardware boom"):
        mcp_server.TOOLS["gui_device_wait_setup"]["handler"]({"name": "flux"})


def test_run_start_captures_operation_id_under_tab_key(wired):
    # run.start is also a guarded op; expected_versions ride along, and the
    # returned operation_id is captured under the tab key.
    mcp_server._LAST_SEEN.update({"tab:t:cfg": 1, "tab:t": 1, "soc": 1, "context": 1})
    wired["run.start"] = {"ok": True, "result": {"operation_id": 5}}
    wired["resources.versions"] = _versions(dict(mcp_server._LAST_SEEN))

    mcp_server.send_gui_rpc("run.start", {"tab_id": "t"})

    assert mcp_server._OP_BY_KEY["tab:t"] == 5
