"""Unit tests for wire.py field validators and request envelope parsing.

Exercises the strict field validators (require_*, optional_*) and
parse_request that guard the raw RPC params dict before any controller is
touched. The domain ``coerce_*`` builders that compose these primitives live in
dispatch.py and are tested in ``test_dispatch_coerce.py``.
"""

from __future__ import annotations

import pytest
from zcu_tools.gui.services.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.services.remote.wire import (
    Response,
    optional_bool,
    parse_request,
    require_int,
    require_json_safe,
    require_object,
    require_str,
)

# ---------------------------------------------------------------------------
# require_str
# ---------------------------------------------------------------------------


def testrequire_str_ok():
    assert require_str({"k": "hello"}, "k") == "hello"


def testrequire_str_missing():
    with pytest.raises(RemoteError) as exc_info:
        require_str({}, "k")
    assert exc_info.value.code == ErrorCode.INVALID_PARAMS


def testrequire_str_not_string():
    with pytest.raises(RemoteError):
        require_str({"k": 123}, "k")


def testrequire_str_empty():
    with pytest.raises(RemoteError):
        require_str({"k": ""}, "k")


# ---------------------------------------------------------------------------
# require_int
# ---------------------------------------------------------------------------


def testrequire_int_ok():
    assert require_int({"k": 42}, "k") == 42


def testrequire_int_rejects_bool():
    # bool is a subclass of int — must be explicitly rejected
    with pytest.raises(RemoteError):
        require_int({"k": True}, "k")


def testrequire_int_rejects_float():
    with pytest.raises(RemoteError):
        require_int({"k": 3.14}, "k")


def testrequire_int_rejects_none():
    with pytest.raises(RemoteError):
        require_int({}, "k")


# ---------------------------------------------------------------------------
# optional_bool
# ---------------------------------------------------------------------------


def testoptional_bool_default():
    assert optional_bool({}, "k", True) is True
    assert optional_bool({}, "k", False) is False


def testoptional_bool_ok():
    assert optional_bool({"k": False}, "k", True) is False


def testoptional_bool_not_bool():
    with pytest.raises(RemoteError):
        optional_bool({"k": "yes"}, "k", True)


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
