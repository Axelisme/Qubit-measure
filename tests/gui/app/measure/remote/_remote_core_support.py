"""Shared recording GUI and socket helpers for remote contract tests."""

from __future__ import annotations

import json
import socket
from typing import Any
from unittest.mock import MagicMock

from ._helpers import Fixture, recv_until
from ._helpers import open_client as open_owner_client


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


def recv_response(sock: socket.socket) -> dict[str, Any]:
    """Read the next complete reply with real Qt owner delivery and framing."""
    return recv_until(sock, lambda msg: "id" in msg)


def open_client(port: int) -> socket.socket:
    return open_owner_client(port)
