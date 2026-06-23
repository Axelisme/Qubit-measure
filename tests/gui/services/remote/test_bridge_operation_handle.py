"""mcp_server-side async-operation handle bookkeeping.

Drives send_gui_rpc / the generic gui_op_wait tool with the socket layer mocked,
asserting the mcp policy: a start op's operation_id is captured under its semantic
key (latest wins) AND folded into the START reply as 'handle'; the generic
gui_op_wait then drives operation.await by that handle directly (P2 / ADR-0026 §8 —
no per-op by-name wait tool, no name->id translation). Also covers the
connect/disconnect/setup short-wait degrade (finished -> snapshot, timeout ->
pending handle).
"""

from __future__ import annotations

from typing import Any

import pytest
from zcu_tools.mcp.measure import server as mcp_server

from ._helpers import FakeTransport


@pytest.fixture()
def wired(monkeypatch):
    """Inject a synchronous FakeTransport into the bridge; reset mcp policy state.

    No socket internals are patched: the bridge runs its REAL send_rpc_raw over
    the fake transport, which echoes a reply per ``replies[method]``. ``_LAST_SEEN``
    / ``_OP_BY_KEY`` are measure-gui mcp policy (reset for isolation). The returned
    dict carries ``["sent"]`` = the transport's recorded outgoing (method, params).
    """
    fake = FakeTransport()
    mcp_server._BRIDGE.set_transport(fake)
    monkeypatch.setattr(mcp_server, "_LAST_SEEN", {}, raising=False)
    monkeypatch.setattr(mcp_server, "_OP_BY_KEY", {}, raising=False)
    replies: dict[str, dict[str, Any]] = fake.replies
    replies["sent"] = fake.sent  # type: ignore[assignment]
    yield replies
    mcp_server._BRIDGE.set_transport(None)


def _versions(table: dict[str, int]) -> dict[str, Any]:
    return {"ok": True, "result": {"versions": table}}


def test_device_setup_captures_operation_id_and_renames_to_handle(wired):
    wired["device.setup"] = {"ok": True, "result": {"operation_id": 42}}
    wired["resources.versions"] = _versions({})

    out = mcp_server.send_gui_rpc("device.setup", {"name": "flux", "updates": {}})

    assert mcp_server._OP_BY_KEY["device:flux"] == 42  # captured under the device key
    # P2 (ADR-0026 §8): the raw operation_id is renamed to 'handle' and KEPT in the
    # reply (the agent drives gui_op_wait / gui_op_poll with it) — the raw id key is
    # gone.
    assert "operation_id" not in out
    assert out["handle"] == 42


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


def test_soc_connect_uses_method_spec_timeout(monkeypatch):
    calls: list[tuple[str, dict[str, Any], float]] = []

    def fake_send(method, params, timeout_seconds=30.0):
        calls.append((method, params, timeout_seconds))
        return {"soc": {"description": "mock soc", "is_mock": True}}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)

    out = mcp_server.tool_gui_soc_connect({"kind": "mock"})

    assert out["soc"]["is_mock"] is True
    method, params, timeout = calls[0]
    assert method == "soc.connect"
    assert params == {"kind": "mock"}
    assert timeout == pytest.approx(
        mcp_server.METHOD_SPECS["soc.connect"].timeout_seconds
        + mcp_server._SOC_CONNECT_TIMEOUT_SLACK
    )


def test_soc_connect_timeout_reconciles_completed_connection(monkeypatch):
    calls: list[tuple[str, float]] = []

    def fake_send(method, params, timeout_seconds=30.0):
        del params
        calls.append((method, timeout_seconds))
        if method == "soc.connect":
            raise TimeoutError("late connect reply")
        if method == "state.has_soc":
            return {"value": True}
        if method == "soc.info":
            return {"description": "connected after timeout", "is_mock": False}
        raise AssertionError(method)

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)

    out = mcp_server.tool_gui_soc_connect(
        {"kind": "remote", "ip": "192.0.2.1", "port": 8888}
    )

    assert out["soc"] == {"description": "connected after timeout", "is_mock": False}
    assert "warning" in out
    assert [method for method, _ in calls] == [
        "soc.connect",
        "state.has_soc",
        "soc.info",
    ]
    assert calls[0][1] == pytest.approx(mcp_server._soc_connect_rpc_timeout())
    assert calls[1][1] == mcp_server._SOC_CONNECT_RECONCILE_TIMEOUT
    assert calls[2][1] == mcp_server._SOC_CONNECT_RECONCILE_TIMEOUT


def test_op_wait_forwards_handle_to_operation_await(wired):
    # P2 (ADR-0026 §8): the generic gui_op_wait drives the operation by the handle
    # the START reply folded — no name->id translation, the agent passes the id
    # (operation_id) directly. The old by-name wait tool is retired.
    sent = wired["sent"]
    wired["operation.await"] = {"ok": True, "result": {"status": "finished"}}
    wired["resources.versions"] = _versions({})

    out = mcp_server.TOOLS["gui_op_wait"]["handler"]({"handle": 42})

    await_params = next(p for (m, p) in sent if m == "operation.await")
    assert await_params["operation_id"] == 42  # handle forwarded verbatim
    assert out["status"] == "finished"


def test_op_wait_propagates_failure(wired):
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
        mcp_server.TOOLS["gui_op_wait"]["handler"]({"handle": 7})


def test_op_wait_no_operation_when_handle_absent():
    # no_operation is the by-handle helper's null branch (a missing/already-reaped
    # handle). The public gui_op_wait always carries a handle, so the branch is
    # exercised directly on the helper.
    out = mcp_server._await_operation_by_handle(None, "operation", 1.0)
    assert out["status"] == "no_operation"


def test_await_operation_refreshes_last_seen_via_rpc(wired):
    # Phase 120c-2: no event poll. The version baseline resync now rides every
    # send_gui_rpc round-trip — so gui_op_wait (which calls operation.await)
    # resyncs the baseline, covering the async-terminal bump that an event poll
    # used to. The async run bumped tab:t:result; awaiting it picks that up.
    wired["operation.await"] = {"ok": True, "result": {"status": "finished"}}
    wired["resources.versions"] = _versions({"tab:t:result": 9})

    out = mcp_server.TOOLS["gui_op_wait"]["handler"]({"handle": 5, "timeout": 0.1})

    assert out["status"] == "finished"
    assert "waited_seconds" in out  # how long the wait blocked (Phase 120c-4)
    assert mcp_server._LAST_SEEN.get("tab:t:result") == 9  # baseline resynced


def test_wait_timeout_returns_timed_out_not_raises(monkeypatch):
    # Phase 120c-4: a bounded wait that elapses is an expected outcome, not a
    # crash — gui_op_wait returns {status:'timed_out', waited_seconds} instead of
    # raising. Covers both timeout flavors (bridge socket TimeoutError and the
    # GUI-side "(timeout)" RuntimeError).
    def bridge_timeout(method, params, timeout_seconds=30.0):
        del params, timeout_seconds
        if method == "operation.await":
            raise TimeoutError("GUI RPC 'operation.await' did not complete")
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", bridge_timeout)
    out = mcp_server.TOOLS["gui_op_wait"]["handler"]({"handle": 5, "timeout": 0.05})
    assert out["status"] == "timed_out"
    assert isinstance(out["waited_seconds"], float)

    def gui_timeout(method, params, timeout_seconds=30.0):
        del params, timeout_seconds
        if method == "operation.await":
            raise RuntimeError("GUI Error (timeout): not done")
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", gui_timeout)
    out2 = mcp_server.TOOLS["gui_op_wait"]["handler"]({"handle": 5, "timeout": 0.05})
    assert out2["status"] == "timed_out"


def test_wait_failed_outcome_still_raises(monkeypatch):
    # A 'failed' outcome must still raise as an error (distinct from cancelled
    # which is now a structured result, and from timed_out which is non-raising).
    def fail(method, params, timeout_seconds=30.0):
        del params, timeout_seconds
        if method == "operation.await":
            raise RuntimeError("GUI Error (precondition_failed): run blew up")
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fail)
    with pytest.raises(RuntimeError):
        mcp_server.TOOLS["gui_op_wait"]["handler"]({"handle": 5, "timeout": 0.05})


def test_wait_cancelled_returns_structured_not_raises(wired):
    # ADR-0025 §cancelled-wire: a cancelled operation returns {status:'cancelled'}
    # — NOT a raise — so the agent can read the feedback and re-plan gracefully.
    wired["operation.await"] = {
        "ok": True,
        "result": {
            "reason": "completed",
            "status": "cancelled",
            "feedback": "user pressed Stop",
        },
    }
    wired["resources.versions"] = _versions({})

    out = mcp_server.TOOLS["gui_op_wait"]["handler"]({"handle": 5, "timeout": 5.0})

    assert out["status"] == "cancelled"
    assert out["feedback"] == "user pressed Stop"
    assert "waited_seconds" in out


def test_wait_cancelled_without_feedback_no_feedback_key(wired):
    # Plain cancel (no Stop reason): status='cancelled', feedback key absent.
    wired["operation.await"] = {
        "ok": True,
        "result": {"reason": "completed", "status": "cancelled"},
    }
    wired["resources.versions"] = _versions({})

    out = mcp_server.TOOLS["gui_op_wait"]["handler"]({"handle": 5, "timeout": 5.0})

    assert out["status"] == "cancelled"
    assert "feedback" not in out


def test_run_start_captures_operation_id_under_tab_key(wired):
    # tab.run_start is also a guarded op; expected_versions ride along, and the
    # returned operation_id is captured under the tab key.
    mcp_server._LAST_SEEN.update({"tab:t:cfg": 1, "tab:t": 1, "soc": 1, "context": 1})
    wired["tab.run_start"] = {"ok": True, "result": {"operation_id": 5}}
    wired["resources.versions"] = _versions(dict(mcp_server._LAST_SEEN))

    mcp_server.send_gui_rpc("tab.run_start", {"tab_id": "t"})

    assert mcp_server._OP_BY_KEY["tab:t"] == 5
