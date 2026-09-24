"""Measure sessions retain their own events until a reply drains them."""

from pathlib import Path

from zcu_tools.mcp.measure.session import MeasureMcpSession

from ._support import make_client


def test_events_and_diagnostics_are_separate_and_drained_once(tmp_path: Path) -> None:
    first = make_client(tmp_path)
    second = make_client(tmp_path)
    event = {
        "event": "run_finished",
        "payload": {"tab_id": "t", "outcome": "finished"},
        "seq": 7,
        "origin": {"kind": "agent", "operation_id": "3"},
    }
    diagnostic = {
        "event": "diagnostic",
        "payload": {"severity": "info", "message": "saved"},
    }
    first.context.session.deliver_event(event)
    second.context.session.deliver_event(
        {"event": "tab_added", "payload": {"tab_id": "b"}}
    )
    first.context.session.deliver_event(diagnostic)
    assert first.context.session.drain_pending() == {
        "events": [event],
        "diagnostics": [diagnostic],
    }
    assert first.context.session.drain_pending() == {"events": [], "diagnostics": []}
    assert second.context.session.drain_pending() == {
        "events": [{"event": "tab_added", "payload": {"tab_id": "b"}}],
        "diagnostics": [],
    }


def test_configured_queue_capacity_keeps_the_newest_events(tmp_path: Path) -> None:
    config = make_client(tmp_path).context.config
    session = MeasureMcpSession(
        config,
        resolve_connect_port=lambda config, requested: config.default_port,
        port_is_open=lambda port: False,
        diagnostic_queue_max=2,
        event_queue_max=3,
    )
    for seq in range(5):
        session.deliver_event({"event": "tab_added", "seq": seq})
        session.deliver_event({"event": "diagnostic", "seq": seq})
    assert session.drain_pending() == {
        "events": [{"event": "tab_added", "seq": seq} for seq in (2, 3, 4)],
        "diagnostics": [{"event": "diagnostic", "seq": seq} for seq in (3, 4)],
    }


def test_disconnect_clears_only_its_session_queues(tmp_path: Path) -> None:
    first, second = make_client(tmp_path), make_client(tmp_path)
    event = {"event": "tab_added", "seq": 1}
    diagnostic = {"event": "diagnostic", "payload": {}}
    for client in (first, second):
        client.context.session.deliver_event(event)
        client.context.session.deliver_event(diagnostic)
    first.call("gui_bridge_detach", {})
    assert not first.transport.is_open
    assert second.transport.is_open
    assert first.context.session.drain_pending() == {"events": [], "diagnostics": []}
    assert second.context.session.drain_pending() == {
        "events": [event],
        "diagnostics": [diagnostic],
    }
