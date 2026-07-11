"""MCP-side diagnostic and low-frequency event piggyback queues."""

from __future__ import annotations

from unittest.mock import patch

from zcu_tools.mcp.measure import server as mcp_server


def _clear() -> None:
    mcp_server._SESSION.clear_pending()


def test_deliver_keeps_full_events_separate_from_diagnostics():
    _clear()
    event = {
        "event": "run_finished",
        "payload": {"tab_id": "t1", "outcome": "finished"},
        "seq": 7,
        "origin": {"kind": "agent", "operation_id": "3"},
    }
    mcp_server._deliver_event(event)
    # A diagnostic is queued for piggyback.
    mcp_server._deliver_event(
        {"event": "diagnostic", "payload": {"severity": "error", "title": "x"}}
    )
    drained = mcp_server._drain_pending()
    assert len(drained["diagnostics"]) == 1
    assert drained["diagnostics"][0]["event"] == "diagnostic"
    assert drained["events"] == [event]
    _clear()


def test_drain_pending_takes_diagnostics_and_empties():
    _clear()
    mcp_server._deliver_event({"event": "tab_added", "payload": {"tab_id": "a"}})
    mcp_server._deliver_event(
        {"event": "diagnostic", "payload": {"severity": "info", "message": "saved"}}
    )
    drained = mcp_server._drain_pending()
    assert [d["payload"]["severity"] for d in drained["diagnostics"]] == ["info"]
    assert drained["events"] == [{"event": "tab_added", "payload": {"tab_id": "a"}}]
    # Second drain is empty (one buffer, drained once).
    again = mcp_server._drain_pending()
    assert again == {"diagnostics": [], "events": []}
    _clear()


def test_piggyback_blocks_preserve_event_envelope_as_compact_json() -> None:
    _clear()
    event = {
        "event": "run_finished",
        "payload": {"tab_id": "t1"},
        "seq": 11,
        "origin": {"kind": "agent", "operation_id": "4"},
    }
    mcp_server._deliver_event(event)

    blocks = mcp_server._piggyback_blocks()

    assert blocks == [
        {
            "type": "text",
            "text": (
                'events since last call:\n[{"event":"run_finished",'
                '"payload":{"tab_id":"t1"},"seq":11,"origin":'
                '{"kind":"agent","operation_id":"4"}}]'
            ),
        }
    ]


def test_event_fifo_is_bounded_and_keeps_newest_entries() -> None:
    _clear()
    for seq in range(1025):
        mcp_server._deliver_event({"event": "tab_added", "seq": seq})

    events = mcp_server._drain_pending()["events"]

    assert len(events) == 1024
    assert events[0]["seq"] == 1
    assert events[-1]["seq"] == 1024


def test_disconnect_clears_diagnostic_and_event_queues() -> None:
    _clear()
    mcp_server._deliver_event({"event": "tab_added", "seq": 1})
    mcp_server._deliver_event({"event": "diagnostic", "payload": {}})

    with patch.object(mcp_server._BRIDGE, "disconnect", return_value="detached"):
        result = mcp_server.tool_gui_disconnect({})

    assert result == {"note": "detached"}
    assert mcp_server._drain_pending() == {"diagnostics": [], "events": []}


# ---------------------------------------------------------------------------
# Version handshake: wire (contract) vs gui (code revision) split
# ---------------------------------------------------------------------------


def _note_with(wire_ver, gui_ver) -> str:
    resp = {"result": {"wire_version": wire_ver, "gui_version": gui_ver}}
    # Socket-layer probe moved into the shared bridge (E4): patch its send_rpc_raw
    # and call the bridge's wire_version_note. The version-note INTENT (wire match
    # shows three versions; wire mismatch is hard; differing gui code is not a
    # mismatch) is unchanged.
    with patch.object(mcp_server._BRIDGE, "send_rpc_raw", return_value=resp):
        return mcp_server._BRIDGE.wire_version_note()


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
    with patch.object(mcp_server._BRIDGE, "send_rpc_raw", return_value={"result": {}}):
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
            "await with gui_op_wait.",
        )
    assert out["status"] == "pending"
    assert "await with gui_op_wait." in out["message"]
    mcp_server._OP_BY_KEY.pop("tab:t2", None)


def test_run_override_tools_with_waits():
    # run keeps its async START tool; the per-op wait is retired in favour of the
    # generic gui_op_wait / gui_op_poll driven by the handle the START reply folds
    # (P2 / ADR-0026 §8).
    for t in ("gui_tab_run_start", "gui_op_wait", "gui_op_poll"):
        assert t in mcp_server.TOOLS
    assert "gui_tab_run_wait" not in mcp_server.TOOLS


def test_soc_connect_is_synchronous_no_wait_or_poll():
    # soc.connect is synchronous now: the single gui_soc_connect override remains,
    # but the async _wait / _poll tools are gone.
    assert "gui_soc_connect" in mcp_server.TOOLS
    assert "gui_soc_connect_wait" not in mcp_server.TOOLS
    assert "gui_soc_connect_poll" not in mcp_server.TOOLS
