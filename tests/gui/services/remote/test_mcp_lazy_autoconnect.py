"""Lazy auto-connect on the first guarded gui_* RPC (measure MCP server).

send_gui_rpc must attach to the running GUI by itself the first time it is
called with no live socket, resolving the port via session discovery (the same
path gui_connect takes). These pin: (a) it auto-connects then forwards when a
GUI is discoverable, (b) it raises a clear no-GUI error when the port resolves
to nothing listening, (c) it never reconnects when already connected.

Mocks the bridge's connect/port-resolution rather than opening a real socket;
an already-connected state is modelled with the synchronous FakeTransport.
"""

from __future__ import annotations

from typing import Any

import pytest
from zcu_tools.mcp.measure import server as mcp_server

from ._helpers import FakeTransport


@pytest.fixture()
def disconnected():
    """Bridge with no live socket + a clean guard baseline."""
    mcp_server._BRIDGE.set_transport(None)
    mcp_server._SESSION.clear_policy_state()
    yield
    mcp_server._BRIDGE.set_transport(None)
    mcp_server._SESSION.clear_policy_state()


def _versions_reply(table: dict[str, int]) -> dict[str, Any]:
    return {"ok": True, "result": {"versions": table}}


def _event_replies(fake: FakeTransport) -> None:
    fake.replies["events.list"] = {
        "ok": True,
        "result": {"events": ["tab_added", "run_finished"]},
    }
    fake.replies["events.subscribe"] = {
        "ok": True,
        "result": {"subscribed": ["run_finished", "tab_added"]},
    }


def test_auto_connects_then_forwards_when_gui_discoverable(disconnected, monkeypatch):
    # Session discovery resolves a port; connect() attaches a FakeTransport so the
    # subsequent send_rpc_raw round-trip completes synchronously.
    fake = FakeTransport()
    fake.replies["state.has_soc"] = {"ok": True, "result": {"value": True}}
    fake.replies["resources.versions"] = _versions_reply({})
    _event_replies(fake)

    monkeypatch.setattr(mcp_server, "resolve_connect_port", lambda _cfg, _req: 9911)
    # The pre-connect listener probe (cold-start fast-fail) must see the
    # discovered port as open so auto-connect proceeds to the mocked connect.
    monkeypatch.setattr(mcp_server, "_port_is_open", lambda _port: True)

    connected: dict[str, Any] = {}

    def fake_connect(port, token=None):
        connected["port"] = port
        connected["token"] = token
        mcp_server._BRIDGE.set_transport(fake)
        return "connected"

    monkeypatch.setattr(mcp_server._BRIDGE, "connect", fake_connect)

    result = mcp_server.send_gui_rpc("state.has_soc", {})

    # Resolved the discovered port and attached the control channel (no token).
    assert connected == {"port": 9911, "token": None}
    # And the RPC actually went out over the freshly-attached transport.
    assert ("state.has_soc", {}) in fake.sent
    assert fake.sent[:2] == [
        ("events.list", {}),
        (
            "events.subscribe",
            {"events": ["tab_added", "run_finished"]},
        ),
    ]
    assert result == {"value": True}


def test_does_not_auto_start_soc(disconnected, monkeypatch):
    # Auto-connect attaches the control channel only — it must never trigger a
    # soc.connect (SoC choice is the user's, not an attach side effect).
    fake = FakeTransport()
    fake.replies["resources.versions"] = _versions_reply({})
    _event_replies(fake)
    monkeypatch.setattr(mcp_server, "resolve_connect_port", lambda _cfg, _req: 9911)
    monkeypatch.setattr(mcp_server, "_port_is_open", lambda _port: True)
    monkeypatch.setattr(
        mcp_server._BRIDGE,
        "connect",
        lambda port, token=None: mcp_server._BRIDGE.set_transport(fake),
    )

    mcp_server.send_gui_rpc("tab.snapshot", {"tab_id": "t"})

    sent_methods = [m for (m, _p) in fake.sent]
    assert "soc.connect" not in sent_methods


def test_event_subscription_failure_disconnects_new_transport(
    disconnected, monkeypatch
) -> None:
    fake = FakeTransport()
    fake.replies["events.list"] = {"ok": True, "result": {"events": "invalid"}}
    monkeypatch.setattr(mcp_server, "resolve_connect_port", lambda _cfg, _req: 9911)
    monkeypatch.setattr(mcp_server, "_port_is_open", lambda _port: True)
    monkeypatch.setattr(
        mcp_server._BRIDGE,
        "connect",
        lambda _port, _token=None: mcp_server._BRIDGE.set_transport(fake),
    )

    with pytest.raises(RuntimeError, match="invalid catalog"):
        mcp_server.send_gui_rpc("state.has_soc", {})

    assert mcp_server._BRIDGE.is_connected is False
    assert mcp_server._drain_pending() == {"diagnostics": [], "events": []}


def test_raises_clear_error_when_no_gui(disconnected, monkeypatch):
    # Port resolves (to the convention default) but nothing is listening: the
    # bridge's connect raises, and we surface the no-GUI message — no "call
    # gui_connect first" since attaching is automatic now.
    monkeypatch.setattr(mcp_server, "resolve_connect_port", lambda _cfg, _req: 8765)

    def fake_connect(port, token=None):
        raise RuntimeError(f"No GUI is listening on 127.0.0.1:{port}")

    monkeypatch.setattr(mcp_server._BRIDGE, "connect", fake_connect)

    with pytest.raises(RuntimeError) as ei:
        mcp_server.send_gui_rpc("state.has_soc", {})

    msg = str(ei.value)
    assert "no running measure-gui found to attach to" in msg
    assert "gui_launch" in msg
    assert "gui_connect first" not in msg


def test_does_not_reconnect_when_already_connected(monkeypatch):
    # A live socket → send_gui_rpc must forward straight through without calling
    # resolve_connect_port / connect again.
    fake = FakeTransport()
    fake.replies["state.has_soc"] = {"ok": True, "result": {"value": True}}
    fake.replies["resources.versions"] = _versions_reply({})
    mcp_server._BRIDGE.set_transport(fake)
    mcp_server._SESSION.clear_policy_state()

    def _boom(*_a, **_k):
        raise AssertionError("must not re-attach when already connected")

    monkeypatch.setattr(mcp_server, "resolve_connect_port", _boom)
    monkeypatch.setattr(mcp_server._BRIDGE, "connect", _boom)

    try:
        result = mcp_server.send_gui_rpc("state.has_soc", {})
    finally:
        mcp_server._BRIDGE.set_transport(None)

    assert result == {"value": True}
    assert ("state.has_soc", {}) in fake.sent
