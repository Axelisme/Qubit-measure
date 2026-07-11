"""Tests for the fluxdep remote wire envelope + field-validation primitives."""

from __future__ import annotations

import pytest
from zcu_tools.gui.app.fluxdep.services.remote.wire_version import (
    GUI_VERSION,
    WIRE_VERSION,
)
from zcu_tools.gui.remote.errors import (
    ErrorCode,
    ErrorEnvelope,
    RemoteError,
)
from zcu_tools.gui.remote.wire import (
    Response,
    parse_request,
    require_int,
    require_str,
)


def test_versions_start_at_one():
    assert WIRE_VERSION == 2
    assert GUI_VERSION == 3


def test_parse_request_ok():
    req = parse_request({"id": "r1", "method": "spectrum.list", "params": {"a": 1}})
    assert req.id == "r1"
    assert req.method == "spectrum.list"
    assert req.params == {"a": 1}


def test_parse_request_defaults_empty_params():
    req = parse_request({"id": "r1", "method": "spectrum.list"})
    assert req.params == {}


def test_parse_request_rejects_non_object_params():
    with pytest.raises(RemoteError) as exc:
        parse_request({"id": "r1", "method": "m", "params": [1, 2]})
    assert exc.value.code is ErrorCode.INVALID_PARAMS


def test_require_str_rejects_missing_and_empty():
    with pytest.raises(RemoteError):
        require_str({}, "k")
    with pytest.raises(RemoteError):
        require_str({"k": ""}, "k")
    assert require_str({"k": "v"}, "k") == "v"


def test_require_int_rejects_bool():
    with pytest.raises(RemoteError):
        require_int({"k": True}, "k")
    assert require_int({"k": 3}, "k") == 3


def test_response_to_wire_ok_and_error():
    ok = Response(id="r1", ok=True, result={"x": 1}).to_wire()
    assert ok == {"id": "r1", "ok": True, "result": {"x": 1}}

    env = ErrorEnvelope(code=ErrorCode.INVALID_PARAMS.value, message="bad")
    err = Response(id="r1", ok=False, error=env).to_wire()
    assert err["ok"] is False
    error = err["error"]
    assert isinstance(error, dict)
    assert error["code"] == "invalid_params"
