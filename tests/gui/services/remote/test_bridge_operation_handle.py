"""mcp_server-side async-operation handle bookkeeping.

Drives send_gui_rpc / the gui_device_wait_operation tool with the socket layer
mocked, asserting the mcp policy: a start op's operation_id is captured under its
semantic key (latest wins), and the semantic wait tool translates name ->
operation_id -> operation.await without the agent ever seeing the raw id. Also
covers the connect/disconnect/setup short-wait degrade (finished -> snapshot,
timeout -> pending handle).
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.services.remote import mcp_server


@pytest.fixture()
def wired(monkeypatch):
    """Synchronous in-memory responder keyed by method; records what was sent."""
    # The socket layer moved into mcp_server._BRIDGE (an McpBridge): the live
    # socket is _BRIDGE._sock and the send/deliver primitives are
    # _BRIDGE._send_line / _BRIDGE._deliver_reply. Patching _sock to a truthy
    # mock makes send_rpc_raw's "is None" guard pass; the fake _send_line
    # synchronously delivers a reply via the bridge's _deliver_reply.
    monkeypatch.setattr(mcp_server._BRIDGE, "_sock", MagicMock())
    monkeypatch.setattr(mcp_server, "_LAST_SEEN", {}, raising=False)
    monkeypatch.setattr(mcp_server, "_OP_BY_KEY", {}, raising=False)
    replies: Dict[str, Dict[str, Any]] = {}
    sent: list[tuple[str, Dict[str, Any]]] = []

    def fake_send_line(payload):
        method = payload["method"]
        sent.append((method, payload["params"]))
        resp = dict(replies.get(method, {"ok": True, "result": {}}))
        resp["id"] = payload["id"]
        mcp_server._BRIDGE._deliver_reply(resp)

    monkeypatch.setattr(mcp_server._BRIDGE, "_send_line", fake_send_line)
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


def test_connect_captures_operation_id_under_device_key(wired):
    wired["device.connect"] = {"ok": True, "result": {"operation_id": 9}}
    wired["operation.await"] = {"ok": True, "result": {"status": "finished"}}
    wired["device.snapshot"] = {
        "ok": True,
        "result": {"snapshot": {"name": "flux", "status": "connected"}},
    }
    wired["resources.versions"] = _versions({})

    out = mcp_server.TOOLS["gui_device_connect"]["handler"](
        {"type_name": "FakeDevice", "name": "flux", "address": "addr"}
    )

    # connect's operation_id is captured (so wait_operation can await by name) and
    # the short wait landed -> finished + snapshot returned.
    assert out["status"] == "finished"
    assert out["snapshot"]["name"] == "flux"


def test_connect_degrades_to_pending_on_short_wait_timeout(wired):
    wired["device.connect"] = {"ok": True, "result": {"operation_id": 9}}
    wired["operation.await"] = {
        "ok": False,
        "error": {"code": "timeout", "message": "did not complete within 1.0s"},
    }
    wired["resources.versions"] = _versions({})

    out = mcp_server.TOOLS["gui_device_connect"]["handler"](
        {"type_name": "FakeDevice", "name": "flux", "address": "addr"}
    )

    # Timeout is the expected degrade path -> pending handle, NOT an error.
    assert out["status"] == "pending"
    assert "flux" in out["message"]


def test_wait_operation_translates_name_to_operation_await(wired):
    sent = wired["sent"]
    mcp_server._OP_BY_KEY["device:flux"] = 42
    wired["operation.await"] = {"ok": True, "result": {"status": "finished"}}
    wired["resources.versions"] = _versions({})

    out = mcp_server.TOOLS["gui_device_wait_operation"]["handler"]({"name": "flux"})

    await_params = next(p for (m, p) in sent if m == "operation.await")
    assert await_params["operation_id"] == 42  # name translated to id
    assert out["status"] == "finished"


def test_wait_operation_no_operation_when_key_unknown(wired):
    out = mcp_server.TOOLS["gui_device_wait_operation"]["handler"]({"name": "ghost"})
    assert out["status"] == "no_operation"


def test_wait_operation_propagates_failure(wired):
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
        mcp_server.TOOLS["gui_device_wait_operation"]["handler"]({"name": "flux"})


def test_await_operation_refreshes_last_seen_via_rpc(wired):
    # Phase 120c-2: no event poll. The version baseline resync now rides every
    # send_gui_rpc round-trip — so a wait/poll (which calls operation.await)
    # resyncs the baseline, covering the async-terminal bump that an event poll
    # used to. The async run bumped tab:t:result; awaiting it picks that up.
    mcp_server._OP_BY_KEY["tab:t"] = 5
    wired["operation.await"] = {"ok": True, "result": {"status": "finished"}}
    wired["resources.versions"] = _versions({"tab:t:result": 9})

    out = mcp_server.TOOLS["gui_run_wait"]["handler"]({"tab_id": "t", "timeout": 0.1})

    assert out["status"] == "finished"
    assert "waited_seconds" in out  # how long the wait blocked (Phase 120c-4)
    assert mcp_server._LAST_SEEN.get("tab:t:result") == 9  # baseline resynced
    mcp_server._OP_BY_KEY.pop("tab:t", None)


def test_wait_timeout_returns_timed_out_not_raises(monkeypatch):
    # Phase 120c-4: a bounded wait that elapses is an expected outcome, not a
    # crash — gui_run_wait returns {status:'timed_out', waited_seconds} instead
    # of raising. Covers both timeout flavors (bridge socket TimeoutError and the
    # GUI-side "(timeout)" RuntimeError).
    mcp_server._OP_BY_KEY["tab:t"] = 5

    def bridge_timeout(method, params, timeout_seconds=30.0):
        del params, timeout_seconds
        if method == "operation.await":
            raise TimeoutError("GUI RPC 'operation.await' did not complete")
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", bridge_timeout)
    out = mcp_server.TOOLS["gui_run_wait"]["handler"]({"tab_id": "t", "timeout": 0.05})
    assert out["status"] == "timed_out"
    assert isinstance(out["waited_seconds"], float)

    def gui_timeout(method, params, timeout_seconds=30.0):
        del params, timeout_seconds
        if method == "operation.await":
            raise RuntimeError("GUI Error (timeout): not done")
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", gui_timeout)
    out2 = mcp_server.TOOLS["gui_run_wait"]["handler"]({"tab_id": "t", "timeout": 0.05})
    assert out2["status"] == "timed_out"
    mcp_server._OP_BY_KEY.pop("tab:t", None)


def test_wait_genuine_failure_still_raises(monkeypatch):
    # A failed/cancelled outcome is NOT a timeout — it must still raise so the
    # agent sees it as an error (distinct from timed_out).
    mcp_server._OP_BY_KEY["tab:t"] = 5

    def fail(method, params, timeout_seconds=30.0):
        del params, timeout_seconds
        if method == "operation.await":
            raise RuntimeError("GUI Error (precondition_failed): run blew up")
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fail)
    with pytest.raises(RuntimeError):
        mcp_server.TOOLS["gui_run_wait"]["handler"]({"tab_id": "t", "timeout": 0.05})
    mcp_server._OP_BY_KEY.pop("tab:t", None)


def test_run_start_captures_operation_id_under_tab_key(wired):
    # run.start is also a guarded op; expected_versions ride along, and the
    # returned operation_id is captured under the tab key.
    mcp_server._LAST_SEEN.update({"tab:t:cfg": 1, "tab:t": 1, "soc": 1, "context": 1})
    wired["run.start"] = {"ok": True, "result": {"operation_id": 5}}
    wired["resources.versions"] = _versions(dict(mcp_server._LAST_SEEN))

    mcp_server.send_gui_rpc("run.start", {"tab_id": "t"})

    assert mcp_server._OP_BY_KEY["tab:t"] == 5
