"""The shipped entrypoint binds tool replies and exit cleanup to its session."""

import io
import json
import sys
from typing import Any

import pytest
from zcu_tools.gui import logging_setup
from zcu_tools.mcp.core.bridge import McpBridge
from zcu_tools.mcp.measure import server

from ._support import WireTransport


class TextStream(io.StringIO):
    def reconfigure(self, *, encoding: str) -> None:
        pass


def test_stdio_only_drains_events_on_success_and_preserves_wire_envelopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = {
        "event": "run_finished",
        "payload": {"tab_id": "t"},
        "seq": 11,
        "origin": {"kind": "agent", "operation_id": "4"},
    }
    transport = WireTransport()
    transport.replies.update(
        {
            "events.list": {"ok": True, "result": {"events": ["run_finished"]}},
            "events.subscribe": {"ok": True, "result": {}},
        }
    )

    def failed_load(params: dict[str, Any]) -> dict[str, Any]:
        if transport.deliver_event is None:
            raise AssertionError("Missing event sink")
        transport.deliver_event(event)
        transport.deliver_event(
            {
                "event": "diagnostic",
                "payload": {
                    "severity": "warning",
                    "title": "Load",
                    "message": "try again",
                },
            }
        )
        return {
            "ok": False,
            "error": {
                "code": "precondition_failed",
                "reason": "invalid_data_file",
                "message": "bad input",
            },
        }

    transport.replies["tab.load_data"] = failed_load
    transport.replies["tab.get_cfg"] = {"ok": True, "result": {"tree": {"gain": 0.5}}}

    def connect(bridge: McpBridge, port: int, token: str | None = None) -> str:
        bridge.set_transport(transport)
        return "connected"

    monkeypatch.setattr(McpBridge, "connect", connect)
    monkeypatch.setattr(logging_setup, "setup_gui_logging", lambda **kwargs: None)
    # Only the lifecycle read fan-out is scripted; operation behavior uses real handlers.
    transport.responder = lambda method, params: {
        "state.has_project": {"value": False},
        "state.has_context": {"value": False},
        "state.has_active_context": {"value": False},
        "state.has_soc": {"value": False},
        "tab.snapshot": {"tabs": []},
        "context.active": {"label": None},
        "state.hardware_gate": {},
        "run.running_tab": {"tab_id": None},
        "view.snapshot": {"active_tab_id": None},
    }[method]
    calls = [
        ("gui_bridge_connect", {"port": 9911}),
        ("gui_tab_load_data", {"tab_id": "t", "data_path": "bad.h5"}),
        ("gui_tab_get_cfg", {"tab_id": "t"}),
        ("gui_tab_get_cfg", {"tab_id": "t"}),
    ]
    stdin = TextStream(
        "".join(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": index,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                }
            )
            + "\n"
            for index, (name, arguments) in enumerate(calls)
        )
    )
    stdout = TextStream()
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)
    server.main()
    replies = [json.loads(line)["result"] for line in stdout.getvalue().splitlines()]
    assert replies[1]["isError"] is True
    assert len(replies[1]["content"]) == 1
    assert "reason: invalid_data_file" in replies[1]["content"][0]["text"]
    assert replies[2]["content"] == [
        {"type": "text", "text": '{"tree":{"gain":0.5}}'},
        {
            "type": "text",
            "text": "notifications since last call:\nwarning: Load — try again",
        },
        {
            "type": "text",
            "text": 'events since last call:\n[{"event":"run_finished","payload":{"tab_id":"t"},"seq":11,"origin":{"kind":"agent","operation_id":"4"}}]',
        },
    ]
    assert replies[3]["content"] == [{"type": "text", "text": '{"tree":{"gain":0.5}}'}]


@pytest.mark.parametrize("launched", [False, True])
def test_stdio_eof_only_stops_a_gui_owned_by_this_bridge(
    monkeypatch: pytest.MonkeyPatch,
    launched: bool,
) -> None:
    stops: list[dict[str, Any]] = []

    def stop(
        bridge: McpBridge, *, timeout: float, timeout_kill: bool, shutdown_rpc: str
    ) -> dict[str, Any]:
        stops.append({"timeout_kill": timeout_kill, "shutdown_rpc": shutdown_rpc})
        return {"exited": True, "note": "stopped"}

    monkeypatch.setattr(McpBridge, "launched_gui", property(lambda self: launched))
    monkeypatch.setattr(McpBridge, "stop", stop)
    monkeypatch.setattr(logging_setup, "setup_gui_logging", lambda **kwargs: None)
    monkeypatch.setattr(sys, "stdin", TextStream())
    monkeypatch.setattr(sys, "stdout", TextStream())
    server.main()
    assert stops == (
        [{"timeout_kill": True, "shutdown_rpc": "app.shutdown"}] if launched else []
    )
