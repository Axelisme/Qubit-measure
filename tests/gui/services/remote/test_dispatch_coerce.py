"""Unit tests for the typed-request coercion helpers in dispatch.py.

These ``coerce_*`` builders turn a raw RPC params mapping into a typed
connection/device domain request. They live in ``dispatch.py`` (beside their
only callers) rather than in the transport-pure ``wire.py``; the field-level
``_require_*``/``_optional_*`` primitives they compose still live in wire.
"""

from __future__ import annotations

import pytest
from zcu_tools.gui.app.main.services.remote.dispatch import (
    coerce_connect_device_request,
    coerce_connect_request,
    coerce_disconnect_device_request,
)
from zcu_tools.gui.app.main.services.remote.errors import RemoteError

# ---------------------------------------------------------------------------
# coerce_connect_request
# ---------------------------------------------------------------------------


def test_coerce_connect_mock():
    req = coerce_connect_request({"kind": "mock"})
    from zcu_tools.gui.app.main.services.connection import ConnectMockRequest

    assert isinstance(req, ConnectMockRequest)


def test_coerce_connect_remote():
    req = coerce_connect_request({"kind": "remote", "ip": "192.168.1.1", "port": 8080})
    from zcu_tools.gui.app.main.services.connection import ConnectRemoteRequest

    assert isinstance(req, ConnectRemoteRequest)
    assert req.ip == "192.168.1.1"
    assert req.port == 8080


def test_coerce_connect_unknown_kind():
    with pytest.raises(RemoteError):
        coerce_connect_request({"kind": "ssh"})


def test_coerce_connect_remote_missing_port():
    with pytest.raises(RemoteError):
        coerce_connect_request({"kind": "remote", "ip": "1.2.3.4"})


# ---------------------------------------------------------------------------
# coerce_connect_device_request
# ---------------------------------------------------------------------------


def test_coerce_connect_device_request_ok():
    req = coerce_connect_device_request(
        {"type_name": "SGS100A", "name": "lo", "address": "TCPIP::1.2.3.4"}
    )
    assert req.type_name == "SGS100A"
    assert req.name == "lo"
    assert req.address == "TCPIP::1.2.3.4"
    assert req.remember is True  # default


def test_coerce_connect_device_request_remember_false():
    req = coerce_connect_device_request(
        {
            "type_name": "SGS100A",
            "name": "lo",
            "address": "TCPIP::1.2.3.4",
            "remember": False,
        }
    )
    assert req.remember is False


# ---------------------------------------------------------------------------
# coerce_disconnect_device_request
# ---------------------------------------------------------------------------


def test_coerce_disconnect_device_request_ok():
    req = coerce_disconnect_device_request({"name": "lo"})
    assert req.name == "lo"
    assert req.remember is True
