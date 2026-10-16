"""Notebook utility error contracts at their public entry points."""

import fcntl
import platform
import socket
from unittest.mock import Mock

import pytest
from zcu_tools.notebook.utils import get_ip_address


def test_get_ip_address_preserves_ioctl_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    original = OSError("synthetic ioctl failure")
    fake_socket = Mock(spec=socket.socket)
    fake_socket.fileno.return_value = 123
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: fake_socket)

    def fail_ioctl(fd: int, request: int, argument: bytes) -> bytes:
        raise original

    monkeypatch.setattr(fcntl, "ioctl", fail_ioctl)

    with pytest.raises(OSError) as caught:
        get_ip_address("synthetic0")

    assert type(caught.value) is OSError
    assert str(caught.value) == "Interface synthetic0 not found or has no IPv4 address."
    assert caught.value.__cause__ is original
