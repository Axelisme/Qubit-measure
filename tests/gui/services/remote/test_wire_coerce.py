"""Unit tests for wire.py coercion helpers and typed-request builders.

Exercises the strict field validators (_require_*, _optional_*, coerce_*)
that guard the raw RPC params dict before any controller is touched.
"""

from __future__ import annotations

import pytest
from zcu_tools.gui.services.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.services.remote.wire import (
    Response,
    _optional_bool,
    _require_int,
    _require_str,
    coerce_connect_device_request,
    coerce_connect_request,
    coerce_disconnect_device_request,
    parse_request,
    require_json_safe,
    require_object,
)

# ---------------------------------------------------------------------------
# _require_str
# ---------------------------------------------------------------------------


def test_require_str_ok():
    assert _require_str({"k": "hello"}, "k") == "hello"


def test_require_str_missing():
    with pytest.raises(RemoteError) as exc_info:
        _require_str({}, "k")
    assert exc_info.value.code == ErrorCode.INVALID_PARAMS


def test_require_str_not_string():
    with pytest.raises(RemoteError):
        _require_str({"k": 123}, "k")


def test_require_str_empty():
    with pytest.raises(RemoteError):
        _require_str({"k": ""}, "k")


# ---------------------------------------------------------------------------
# _require_int
# ---------------------------------------------------------------------------


def test_require_int_ok():
    assert _require_int({"k": 42}, "k") == 42


def test_require_int_rejects_bool():
    # bool is a subclass of int — must be explicitly rejected
    with pytest.raises(RemoteError):
        _require_int({"k": True}, "k")


def test_require_int_rejects_float():
    with pytest.raises(RemoteError):
        _require_int({"k": 3.14}, "k")


def test_require_int_rejects_none():
    with pytest.raises(RemoteError):
        _require_int({}, "k")


# ---------------------------------------------------------------------------
# _optional_bool
# ---------------------------------------------------------------------------


def test_optional_bool_default():
    assert _optional_bool({}, "k", True) is True
    assert _optional_bool({}, "k", False) is False


def test_optional_bool_ok():
    assert _optional_bool({"k": False}, "k", True) is False


def test_optional_bool_not_bool():
    with pytest.raises(RemoteError):
        _optional_bool({"k": "yes"}, "k", True)


# ---------------------------------------------------------------------------
# require_object
# ---------------------------------------------------------------------------


def test_require_object_ok():
    assert require_object({"k": {"a": 1}}, "k") == {"a": 1}


def test_require_object_missing():
    with pytest.raises(RemoteError):
        require_object({}, "k")


def test_require_object_not_dict():
    with pytest.raises(RemoteError):
        require_object({"k": [1, 2]}, "k")


# ---------------------------------------------------------------------------
# require_json_safe
# ---------------------------------------------------------------------------


def test_require_json_safe_ok():
    assert require_json_safe({"k": {"a": [1, 2, None]}}, "k") == {"a": [1, 2, None]}


def test_require_json_safe_missing():
    with pytest.raises(RemoteError):
        require_json_safe({}, "k")


def test_require_json_safe_not_serializable():
    with pytest.raises(RemoteError):
        require_json_safe({"k": object()}, "k")


# ---------------------------------------------------------------------------
# Response.to_wire
# ---------------------------------------------------------------------------


def test_response_to_wire_ok():
    r = Response(id="1", ok=True, result={"val": 42})
    wire = r.to_wire()
    assert wire["ok"] is True
    assert wire["result"] == {"val": 42}
    assert "error" not in wire


def test_response_to_wire_error():
    from zcu_tools.gui.services.remote.errors import ErrorEnvelope

    env = ErrorEnvelope(code=ErrorCode.INVALID_PARAMS, message="bad")
    r = Response(id="2", ok=False, error=env)
    wire = r.to_wire()
    assert wire["ok"] is False
    assert "error" in wire


# ---------------------------------------------------------------------------
# parse_request
# ---------------------------------------------------------------------------


def test_parse_request_params_not_dict():
    with pytest.raises(RemoteError):
        parse_request({"id": "1", "method": "foo", "params": [1, 2]})


def test_parse_request_default_empty_params():
    req = parse_request({"id": "1", "method": "foo"})
    assert req.params == {}


# ---------------------------------------------------------------------------
# coerce_connect_request
# ---------------------------------------------------------------------------


def test_coerce_connect_mock():
    req = coerce_connect_request({"kind": "mock"})
    from zcu_tools.gui.services.connection import ConnectMockRequest

    assert isinstance(req, ConnectMockRequest)


def test_coerce_connect_remote():
    req = coerce_connect_request({"kind": "remote", "ip": "192.168.1.1", "port": 8080})
    from zcu_tools.gui.services.connection import ConnectRemoteRequest

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
