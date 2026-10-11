"""Shared recording GUI and socket helpers for remote contract tests."""

from __future__ import annotations

import json
import socket
import time
from typing import Any
from unittest.mock import MagicMock

from qtpy.QtCore import QCoreApplication

from ._helpers import Fixture


def make_view() -> MagicMock:
    view = MagicMock()
    view.show_status_message = MagicMock()
    view.make_run_container = MagicMock(return_value=None)
    # tab.list_all / overview read active_tab_id off the render view; return a real
    # (JSON-serializable) snapshot so the wire reply encodes cleanly.
    view.get_view_snapshot = MagicMock(
        return_value={"active_tab_id": None, "tab_ids": []}
    )
    return view


class RemoteCoreFixture(Fixture):
    """Use the same real entry/point composition as the remote contract suite."""


def send(sock: socket.socket, obj: dict[str, Any]) -> None:
    sock.sendall((json.dumps(obj) + "\n").encode("utf-8"))


def recv_response(sock: socket.socket, timeout_s: float = 3.0) -> dict[str, Any]:
    """Wait for one NDJSON response, pumping the Qt event loop in between."""
    app = QCoreApplication.instance()
    assert app is not None
    deadline = time.monotonic() + timeout_s
    buf = bytearray()
    sock.setblocking(False)
    while time.monotonic() < deadline:
        try:
            chunk = sock.recv(4096)
            if chunk:
                buf.extend(chunk)
                if b"\n" in buf:
                    line, _, _ = bytes(buf).partition(b"\n")
                    return json.loads(line.decode("utf-8"))
                    # rest discarded — single-response helper
            else:
                # peer closed cleanly mid-recv → return ""
                if buf and b"\n" in buf:
                    line, _, _ = bytes(buf).partition(b"\n")
                    return json.loads(line.decode("utf-8"))
                raise AssertionError("peer closed without a response line")
        except BlockingIOError:
            pass
        app.processEvents()
        time.sleep(0.005)
    raise AssertionError(f"no response within {timeout_s}s")


def open_client(port: int) -> socket.socket:
    return socket.create_connection(("127.0.0.1", port), timeout=1.0)
