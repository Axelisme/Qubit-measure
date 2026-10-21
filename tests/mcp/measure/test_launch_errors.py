"""Occupied-port launch rejection preserves actionable connection diagnostics."""

import os
import socket
from pathlib import Path
from typing import NoReturn

import psutil
import pytest
from zcu_tools.mcp.measure.session import GuiRpcError

from ._support import make_client

# A real listener verifies owner discovery at the OS socket/process boundary.
pytestmark = [pytest.mark.requires_loopback, pytest.mark.uses_wall_clock]


def test_connect_busy_port_identifies_listener_and_port_parameter(
    tmp_path: Path,
) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        client = make_client(tmp_path, port_is_open=lambda _: True)
        try:
            with pytest.raises(GuiRpcError) as error:
                client.call("connect", {"launch": "new", "port": port})
            assert error.value.reason == "port_in_use"
            assert f"pid={os.getpid()}" in str(error.value)
            assert "port parameter" in str(error.value)
            assert str(port) in str(error.value)
            assert client.transport.sent == []
        finally:
            client.context.session.close()


def test_connect_busy_port_retains_actionable_error_when_owner_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(kind: str) -> NoReturn:
        raise psutil.AccessDenied()

    monkeypatch.setattr(psutil, "net_connections", denied)
    client = make_client(tmp_path, port_is_open=lambda _: True)
    try:
        with pytest.raises(GuiRpcError) as error:
            client.call("connect", {"launch": "new"})
        assert error.value.reason == "port_in_use"
        assert "owner unknown" in str(error.value)
        assert "port parameter" in str(error.value)
        assert client.transport.sent == []
    finally:
        client.context.session.close()
