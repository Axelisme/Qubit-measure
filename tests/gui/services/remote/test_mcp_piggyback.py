"""mcp-side diagnostic-only piggyback (Phase 120c-2).

The GUI still emits its full EventBus stream on the wire, but the bridge exposes
ONLY diagnostics to the agent: resource-change events are dropped in
_deliver_event, and _drain_pending returns diagnostics that piggyback any tool
reply. These exercise that pure queue logic without a live socket.
"""

from __future__ import annotations

from unittest.mock import patch

from zcu_tools.gui.services.remote import mcp_server


def _clear() -> None:
    with mcp_server._DIAGNOSTIC_COND:
        mcp_server._DIAGNOSTIC_QUEUE.clear()


def test_deliver_drops_events_keeps_diagnostics():
    _clear()
    # A resource-change event is dropped (agent not exposed to it).
    mcp_server._deliver_event({"event": "run_finished", "payload": {}})
    # A diagnostic is queued for piggyback.
    mcp_server._deliver_event(
        {"event": "diagnostic", "payload": {"severity": "error", "title": "x"}}
    )
    assert len(mcp_server._DIAGNOSTIC_QUEUE) == 1
    assert mcp_server._DIAGNOSTIC_QUEUE[0]["event"] == "diagnostic"
    _clear()


def test_drain_pending_takes_diagnostics_and_empties():
    _clear()
    mcp_server._deliver_event({"event": "tab_added", "payload": {"tab_id": "a"}})
    mcp_server._deliver_event(
        {"event": "diagnostic", "payload": {"severity": "info", "message": "saved"}}
    )
    drained = mcp_server._drain_pending()
    # Only diagnostics surface; the event was dropped on delivery.
    assert "events" not in drained
    assert [d["payload"]["severity"] for d in drained["diagnostics"]] == ["info"]
    # Second drain is empty (one buffer, drained once).
    again = mcp_server._drain_pending()
    assert again == {"diagnostics": []}
    _clear()


# ---------------------------------------------------------------------------
# Version handshake: wire (contract) vs gui (code revision) split
# ---------------------------------------------------------------------------


def _note_with(wire_ver, gui_ver) -> str:
    resp = {"result": {"wire_version": wire_ver, "gui_version": gui_ver}}
    with patch.object(mcp_server, "_send_gui_rpc_raw", return_value=resp):
        return mcp_server._wire_version_note()


def test_note_wire_match_shows_all_three_versions():
    note = _note_with(mcp_server.MCP_WIRE_VERSION, 7)
    assert "MISMATCH" not in note
    assert f"wire v{mcp_server.MCP_WIRE_VERSION}" in note
    # gui/mcp code revisions are reported, not compared.
    assert "gui code v7" in note
    assert f"mcp code v{mcp_server.MCP_VERSION}" in note


def test_note_wire_mismatch_is_hard():
    note = _note_with(mcp_server.MCP_WIRE_VERSION + 99, 1)
    assert "WIRE VERSION MISMATCH" in note


def test_note_differing_gui_version_is_not_a_mismatch():
    # A different GUI code revision must NOT trigger a mismatch — it's reported.
    note = _note_with(mcp_server.MCP_WIRE_VERSION, 999)
    assert "MISMATCH" not in note
    assert "gui code v999" in note


# ---------------------------------------------------------------------------
# Short-wait degrade (shared by device / run / connect)
# ---------------------------------------------------------------------------


def test_short_wait_settles_returns_product():
    # operation.await returns in time -> {status:finished, **product()}.
    mcp_server._OP_BY_KEY["tab:t1"] = 42
    with patch.object(mcp_server, "_send_gui_rpc_raw", return_value={"result": {}}):
        with patch.object(
            mcp_server, "send_gui_rpc", return_value={"status": "finished"}
        ):
            out = mcp_server._start_op_with_short_wait(
                "tab:t1",
                "Run on 't1'",
                1.0,
                lambda: {"tab": {"has_run_result": True}},
                "hint",
            )
    assert out["status"] == "finished"
    assert out["tab"] == {"has_run_result": True}
    mcp_server._OP_BY_KEY.pop("tab:t1", None)


def test_short_wait_timeout_degrades_to_pending():
    mcp_server._OP_BY_KEY["tab:t2"] = 43

    def _raise_timeout(*a, **k):
        raise RuntimeError("GUI Error (timeout): still running")

    with patch.object(mcp_server, "send_gui_rpc", side_effect=_raise_timeout):
        out = mcp_server._start_op_with_short_wait(
            "tab:t2",
            "Run on 't2'",
            1.0,
            lambda: {"tab": {}},
            "await with gui_run_wait.",
        )
    assert out["status"] == "pending"
    assert "await with gui_run_wait." in out["message"]
    mcp_server._OP_BY_KEY.pop("tab:t2", None)


def test_run_and_connect_are_override_tools_with_waits():
    for t in ("gui_run_start", "gui_run_wait", "gui_connect_start", "gui_connect_wait"):
        assert t in mcp_server.TOOLS
