"""Tests for the AgentChatService activity tap in RemoteControlAdapter.

Covers:
  - A successful command method call → entries grows by one activity entry.
  - A query method call → entries unchanged (filtered out).
  - A listener that raises an exception → the RPC reply still succeeds (tap is
    best-effort and must never propagate back to the dispatch path).
"""

from __future__ import annotations

import socket
import time

import pytest
from qtpy.QtCore import QCoreApplication
from zcu_tools.gui.app.main.services.agent_chat import AgentChatService

from ._helpers import (
    Fixture,
    call,
    open_client,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pump(n: int = 10, delay: float = 0.01) -> None:
    """Give the Qt event loop time to process marshalled main-thread calls."""
    app = QCoreApplication.instance()
    assert app is not None
    for _ in range(n):
        app.processEvents()
        time.sleep(delay)


def _start_fixture() -> tuple[Fixture, socket.socket, int]:
    fx = Fixture()
    port = fx.start()
    sock = open_client(port)
    return fx, sock, port


# ---------------------------------------------------------------------------
# activity tap — command method recorded
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_command_method_creates_activity_entry():
    fx, sock, _ = _start_fixture()
    try:
        svc: AgentChatService = fx.ctrl.get_agent_chat()
        before = len(svc.entries())

        # gui_tab_new → tab.new (a clear command); the fixture has a fake adapter.
        resp = call(sock, "tab.new", {"adapter_name": "fake"}, rid="cmd1")
        assert resp.get("ok") is True, f"unexpected reply: {resp}"

        # Give the main thread time to process the marshalled record_activity call.
        _pump(20, delay=0.015)

        after = len(svc.entries())
        assert after > before, "expected at least one new activity entry after tab.new"
        last = svc.entries()[-1]
        assert last.kind == "activity"
        assert "tab.new" in last.text
    finally:
        fx.stop()
        sock.close()


# ---------------------------------------------------------------------------
# activity tap — query method NOT recorded
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_query_method_does_not_create_entry():
    fx, sock, _ = _start_fixture()
    try:
        svc: AgentChatService = fx.ctrl.get_agent_chat()
        before = len(svc.entries())

        # tab.list is a pure query — must not appear in the transcript.
        resp = call(sock, "tab.list", {}, rid="q1")
        assert resp.get("ok") is True, f"unexpected reply: {resp}"

        _pump(20, delay=0.015)

        assert len(svc.entries()) == before, "query method must not add activity entry"
    finally:
        fx.stop()
        sock.close()


# ---------------------------------------------------------------------------
# tap exception must not affect RPC reply
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_tap_exception_does_not_break_rpc_reply():
    """An exploding AgentChatService.record_activity must not crash the dispatch."""
    fx, sock, _ = _start_fixture()
    try:
        svc: AgentChatService = fx.ctrl.get_agent_chat()

        # Inject a listener that raises to simulate a broken observer.
        def _boom():
            raise RuntimeError("injected boom")

        svc.add_listener(_boom)

        # tab.new is a command → will try to record_activity → listener raises.
        # The RPC reply must still be ok=True.
        resp = call(sock, "tab.new", {"adapter_name": "fake"}, rid="boom1")
        assert resp.get("ok") is True, (
            f"expected ok reply even with broken listener: {resp}"
        )
    finally:
        fx.stop()
        sock.close()
