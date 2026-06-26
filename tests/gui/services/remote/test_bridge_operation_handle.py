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
from zcu_tools.mcp.core.bridge import GuiTransportTimeoutError
from zcu_tools.mcp.measure import server as mcp_server

from ._helpers import FakeTransport


@pytest.fixture()
def wired():
    """Inject a synchronous FakeTransport into the bridge; reset mcp policy state.

    No socket internals are patched: the bridge runs its REAL send_rpc_raw over
    the fake transport, which echoes a reply per ``replies[method]``. ``_LAST_SEEN``
    / ``_OP_BY_KEY`` are measure MCP session policy (reset for isolation). The
    returned dict carries ``["sent"]`` = the transport's recorded outgoing
    (method, params).
    """
    fake = FakeTransport()
    mcp_server._BRIDGE.set_transport(fake)
    mcp_server._SESSION.clear_policy_state()
    replies: dict[str, dict[str, Any]] = fake.replies
    replies["sent"] = fake.sent  # type: ignore[assignment]
    yield replies
    mcp_server._BRIDGE.set_transport(None)
    mcp_server._SESSION.clear_policy_state()


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


def test_default_rpc_timeout_uses_method_spec(monkeypatch):
    calls: list[tuple[str, dict[str, Any], float]] = []

    def fake_send(method, params, timeout_seconds):
        calls.append((method, params, timeout_seconds))
        return {}

    monkeypatch.setattr(mcp_server._SESSION, "send_gui_rpc", fake_send)

    mcp_server.send_gui_rpc("state.has_soc", {})

    assert calls == [
        (
            "state.has_soc",
            {},
            mcp_server.METHOD_SPECS["state.has_soc"].timeout_seconds
            + mcp_server._WAIT_TRANSPORT_SLACK_SECONDS,
        )
    ]


def test_long_wait_rpc_requires_explicit_timeout():
    with pytest.raises(ValueError, match="operation.await"):
        mcp_server.send_gui_rpc("operation.await", {"operation_id": 1, "timeout": 0.0})


def test_send_gui_rpc_marks_gui_handler_timeout(wired):
    wired["state.has_soc"] = {
        "ok": False,
        "error": {
            "code": "timeout",
            "message": "handler did not complete within 5.0s",
        },
    }

    with pytest.raises(mcp_server.GuiRpcError) as exc_info:
        mcp_server.send_gui_rpc("state.has_soc", {})

    assert exc_info.value.code == "timeout"
    assert exc_info.value.reason == "gui_handler_timeout"


def test_send_gui_rpc_marks_transport_timeout(monkeypatch, wired):
    del wired

    def fail_transport(method, params, timeout_seconds):
        del params
        raise GuiTransportTimeoutError(method, timeout_seconds)

    monkeypatch.setattr(mcp_server._BRIDGE, "send_rpc_raw", fail_transport)

    with pytest.raises(mcp_server.GuiRpcError) as exc_info:
        mcp_server.send_gui_rpc("state.has_soc", {})

    assert exc_info.value.code == "timeout"
    assert exc_info.value.reason == "gui_transport_timeout"


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
    # raising. Covers both structured GuiRpcError and the legacy string form.
    def gui_handler_timeout(method, params, timeout_seconds=30.0):
        del params, timeout_seconds
        if method == "operation.await":
            raise mcp_server.GuiRpcError(
                "GUI Error (timeout): not done",
                reason="gui_handler_timeout",
                code="timeout",
            )
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", gui_handler_timeout)
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


def test_wait_transport_timeout_raises(monkeypatch):
    def transport_timeout(method, params, timeout_seconds=30.0):
        del params, timeout_seconds
        if method == "operation.await":
            raise mcp_server.GuiRpcError(
                "GUI Transport Timeout: operation.await",
                reason="gui_transport_timeout",
                code="timeout",
            )
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", transport_timeout)

    with pytest.raises(mcp_server.GuiRpcError, match="Transport Timeout"):
        mcp_server.TOOLS["gui_op_wait"]["handler"]({"handle": 5, "timeout": 0.05})


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


class _AwaitScript:
    """Stateful fake send_gui_rpc that scripts a sequence of operation.await
    replies (the poll drain loop calls await repeatedly) plus a static
    operation.progress payload.

    Each ``await`` step is one of: a dict payload (delivered as a success reply),
    or a RuntimeError instance (raised — e.g. a TIMEOUT or PRECONDITION_FAILED).
    After the scripted steps are exhausted, the last step repeats (so a terminal /
    timeout is sticky, mirroring the wire's settled re-readability).
    """

    def __init__(self, steps, progress=None):
        self._steps = list(steps)
        self._idx = 0
        self._progress = progress or {"active": True, "bars": []}
        self.await_calls = 0

    def __call__(self, method, params, timeout_seconds=30.0):
        del params, timeout_seconds
        if method == "operation.progress":
            return self._progress
        if method == "operation.await":
            self.await_calls += 1
            step = self._steps[min(self._idx, len(self._steps) - 1)]
            self._idx += 1
            if isinstance(step, BaseException):
                raise step
            return step
        return {}


def _timeout_err() -> RuntimeError:
    return RuntimeError("GUI Error (timeout): not done")


def _msg(text: str) -> dict[str, Any]:
    return {"reason": "user_feedback", "feedback": text}


def test_poll_drains_all_buffered_feedback_while_running(monkeypatch):
    # (1) N feedback Messages buffered + op running -> status 'running' AND the
    # reply carries all N messages in arrival order. Because poll DRAINS, a
    # subsequent immediate wait does NOT re-deliver them (the queue is emptied;
    # await then times out = still running).
    script = _AwaitScript(
        [_msg("first"), _msg("second"), _msg("third"), _timeout_err()],
        progress={
            "active": True,
            "bars": [{"token": "t0", "format": "50%", "percent": 50, "maximum": 100}],
        },
    )
    monkeypatch.setattr(mcp_server, "send_gui_rpc", script)

    out = mcp_server.TOOLS["gui_op_poll"]["handler"]({"handle": 5})

    assert out["status"] == "running"
    assert out["feedback"] == ["first", "second", "third"]  # FIFO order
    assert out["bars"] == [{"token": "t0", "format": "50%", "percent": 50}]
    # 3 messages + 1 timeout = 4 await calls (drained until the queue emptied).
    assert script.await_calls == 4

    # Subsequent immediate wait: queue already drained -> await times out ->
    # gui_op_wait reports 'timed_out' (still running), NOT the consumed feedback.
    follow = _AwaitScript([_timeout_err()])
    monkeypatch.setattr(mcp_server, "send_gui_rpc", follow)
    out2 = mcp_server.TOOLS["gui_op_wait"]["handler"]({"handle": 5, "timeout": 0.01})
    assert out2["status"] == "timed_out"
    assert "feedback" not in out2


def test_poll_after_send_and_stop_returns_cancelled_with_reason(monkeypatch):
    # (2) Send & Stop -> sticky cancelled terminal. poll -> status 'cancelled'
    # with the Stop reason; a subsequent wait STILL returns cancelled (the sticky
    # terminal is not destroyed by poll).
    cancelled = {
        "reason": "completed",
        "status": "cancelled",
        "feedback": "stop: recalibrate first",
    }
    script = _AwaitScript([cancelled])
    monkeypatch.setattr(mcp_server, "send_gui_rpc", script)

    out = mcp_server.TOOLS["gui_op_poll"]["handler"]({"handle": 7})

    assert out["status"] == "cancelled"
    assert out["stop_reason"] == "stop: recalibrate first"
    assert "recalibrate" in out["message"]

    # Sticky terminal: a later wait reads the SAME cancelled outcome (re-readable).
    out2 = mcp_server.TOOLS["gui_op_wait"]["handler"]({"handle": 7, "timeout": 1.0})
    assert out2["status"] == "cancelled"
    assert out2["feedback"] == "stop: recalibrate first"


def test_poll_running_no_events_folds_progress(monkeypatch):
    # (3) running op, no events -> 'running' + slimmed progress bars (the
    # unchanged behaviour; no feedback key when nothing was drained).
    script = _AwaitScript(
        [_timeout_err()],
        progress={
            "active": True,
            "bars": [
                {"token": "outer", "format": "2/10", "percent": 20, "maximum": 10}
            ],
        },
    )
    monkeypatch.setattr(mcp_server, "send_gui_rpc", script)

    out = mcp_server.TOOLS["gui_op_poll"]["handler"]({"handle": 9})

    assert out["status"] == "running"
    assert out["active"] is True
    assert out["bars"] == [{"token": "outer", "format": "2/10", "percent": 20}]
    assert "feedback" not in out


def test_poll_cleanly_finished(monkeypatch):
    # (4) cleanly finished op -> 'finished' (no feedback key).
    script = _AwaitScript([{"reason": "completed", "status": "finished"}])
    monkeypatch.setattr(mcp_server, "send_gui_rpc", script)

    out = mcp_server.TOOLS["gui_op_poll"]["handler"]({"handle": 11})

    assert out["status"] == "finished"
    assert "feedback" not in out


def test_poll_messages_then_terminal_in_one_drain(monkeypatch):
    # (5) mixed: messages THEN a terminal drained in one poll -> the reply carries
    # BOTH the feedback list AND the terminal status. Here the terminal is a clean
    # finish; the queued nudges that preceded it are still surfaced.
    script = _AwaitScript(
        [
            _msg("note A"),
            _msg("note B"),
            {"reason": "completed", "status": "finished"},
        ]
    )
    monkeypatch.setattr(mcp_server, "send_gui_rpc", script)

    out = mcp_server.TOOLS["gui_op_poll"]["handler"]({"handle": 13})

    assert out["status"] == "finished"
    assert out["feedback"] == ["note A", "note B"]
    assert script.await_calls == 3


def test_poll_messages_then_cancelled_terminal(monkeypatch):
    # (5b) mixed where the terminal is a cancel: both the drained feedback list and
    # the cancelled status (+stop_reason) ride the same reply.
    script = _AwaitScript(
        [
            _msg("watch the fridge temp"),
            {
                "reason": "completed",
                "status": "cancelled",
                "feedback": "stop: drifted",
            },
        ]
    )
    monkeypatch.setattr(mcp_server, "send_gui_rpc", script)

    out = mcp_server.TOOLS["gui_op_poll"]["handler"]({"handle": 15})

    assert out["status"] == "cancelled"
    assert out["feedback"] == ["watch the fridge temp"]
    assert out["stop_reason"] == "stop: drifted"


def test_poll_failed_outcome_reports_failed_status(monkeypatch):
    # A genuine failure surfaces as a raised RuntimeError (reason='failed'); poll
    # reports it as status 'failed' (does NOT raise, unlike the blocking wait).
    def fail(method, params, timeout_seconds=30.0):
        del params, timeout_seconds
        if method == "operation.await":
            raise RuntimeError("GUI Error (precondition_failed): hardware boom")
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fail)

    out = mcp_server.TOOLS["gui_op_poll"]["handler"]({"handle": 17})

    assert out["status"] == "failed"
    assert "hardware boom" in out["message"]


def test_poll_no_operation_when_handle_absent():
    out = mcp_server._poll_operation_by_handle(None, "operation")
    assert out["status"] == "no_operation"


def test_run_start_captures_operation_id_under_tab_key(wired):
    # tab.run_start is also a guarded op; expected_versions ride along, and the
    # returned operation_id is captured under the tab key.
    mcp_server._LAST_SEEN.update({"tab:t:cfg": 1, "tab:t": 1, "soc": 1, "context": 1})
    wired["tab.run_start"] = {"ok": True, "result": {"operation_id": 5}}
    wired["resources.versions"] = _versions(dict(mcp_server._LAST_SEEN))

    mcp_server.send_gui_rpc("tab.run_start", {"tab_id": "t"})

    assert mcp_server._OP_BY_KEY["tab:t"] == 5
