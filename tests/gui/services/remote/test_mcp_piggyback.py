"""mcp-side event/diagnostic split + piggyback drain (ADR-0013 C2a).

These exercise the bridge's pure queue logic without a live socket: the reader
routes diagnostics into their own queue, and _drain_pending takes both buffers
so any tool result can piggyback background notifications.
"""

from __future__ import annotations

from unittest.mock import patch

from zcu_tools.gui.services.remote import mcp_server


def _clear() -> None:
    with mcp_server._EVENT_COND:
        mcp_server._EVENT_QUEUE.clear()
        mcp_server._DIAGNOSTIC_QUEUE.clear()


def test_reader_routes_diagnostic_to_its_own_queue():
    _clear()
    mcp_server._deliver_event({"event": "run_finished", "payload": {}})
    mcp_server._deliver_event(
        {"event": "diagnostic", "payload": {"severity": "error", "title": "x"}}
    )
    assert len(mcp_server._EVENT_QUEUE) == 1
    assert len(mcp_server._DIAGNOSTIC_QUEUE) == 1
    assert mcp_server._EVENT_QUEUE[0]["event"] == "run_finished"
    assert mcp_server._DIAGNOSTIC_QUEUE[0]["event"] == "diagnostic"
    _clear()


def test_drain_pending_takes_both_and_empties():
    _clear()
    mcp_server._deliver_event({"event": "tab_added", "payload": {"tab_id": "a"}})
    mcp_server._deliver_event(
        {"event": "diagnostic", "payload": {"severity": "info", "message": "saved"}}
    )
    drained = mcp_server._drain_pending()
    assert [e["event"] for e in drained["events"]] == ["tab_added"]
    assert [d["payload"]["severity"] for d in drained["diagnostics"]] == ["info"]
    # Second drain is empty (single buffer, one drain each).
    again = mcp_server._drain_pending()
    assert again == {"events": [], "diagnostics": []}
    _clear()


def test_default_subscribe_set_is_experiment_lifecycle():
    assert set(mcp_server._DEFAULT_SUBSCRIBE) == {
        "run_started",
        "run_finished",
        "device_setup_changed",
        "soc_changed",
    }


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
