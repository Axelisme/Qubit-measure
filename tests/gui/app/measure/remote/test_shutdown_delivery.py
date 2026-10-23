"""Shutdown admission waits for terminal delivery of its correlated reply."""

from __future__ import annotations

import socket
from threading import Event

import pytest
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler

from tests.gui._qt_waits import qt_until
from tests.gui.app.measure.remote._helpers import (
    Fixture,
    open_client,
    recv_response,
    send,
)

pytestmark = pytest.mark.requires_loopback


@pytest.mark.parametrize("outcome", ["sent", "send_failure", "disconnect"])
def test_shutdown_waits_for_reply_send_terminal_before_requesting_close(
    qapp, tmp_path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    fixture = Fixture(project_root=str(tmp_path), empty_project=True)
    port = fixture.start()
    owner = QtOwnerScheduler()
    entered = Event()
    release = Event()
    shutdown_on_owner: list[bool] = []
    fixture.view.request_shutdown.side_effect = lambda: shutdown_on_owner.append(
        owner.is_owner_thread()
    )
    original_send = socket.socket.send

    def controlled_send(sock: socket.socket, data: bytes, flags: int = 0) -> int:
        if b'"shutting_down"' in bytes(data):
            entered.set()
            if not release.wait(timeout=30):
                raise TimeoutError("test did not release the controlled writer")
            if outcome == "send_failure":
                raise OSError("controlled shutdown reply send failure")
        return original_send(sock, data, flags)

    monkeypatch.setattr(socket.socket, "send", controlled_send)
    client = open_client(port)
    try:
        send(
            client,
            {
                "id": "shutdown",
                "method": "app.shutdown",
                "params": {"discard_unsaved": True},
            },
        )
        qt_until(entered.is_set, label="shutdown reply reached socket writer")
        # The writer has reached send but cannot finish. Owner-loop turns cannot
        # establish delivery or authorize teardown of this accepted RPC.
        assert shutdown_on_owner == [], (
            "shutdown ran while its accepted reply was still blocked"
        )
        if outcome == "disconnect":
            client.close()
            qt_until(
                lambda: bool(shutdown_on_owner), label="disconnected accepted shutdown"
            )
        release.set()
        if outcome == "sent":
            reply = recv_response(client, "shutdown")
            assert reply["ok"] is True
            assert reply["result"]["shutting_down"] is True
        qt_until(
            lambda: bool(shutdown_on_owner), label="terminal shutdown continuation"
        )
        assert shutdown_on_owner == [True]
    finally:
        release.set()
        client.close()
        fixture.stop()
        owner.deleteLater()
        qapp.processEvents()
