"""Unit tests for the typed-request coercion helpers in connection_device handlers.

These ``coerce_*`` builders turn a raw RPC params mapping into a typed
connection/device domain request. They live beside their only callers rather
than in the transport-pure ``wire.py``; the field-level
``_require_*``/``_optional_*`` primitives they compose still live in wire.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import create_model
from zcu_tools.device import BaseDeviceInfo
from zcu_tools.gui.app.measure.remote.handlers.connection_device import (
    coerce_connect_device_request,
    coerce_connect_request,
    coerce_disconnect_device_request,
    h_device_setup_spec,
)
from zcu_tools.gui.remote.errors import RemoteError

# ---------------------------------------------------------------------------
# coerce_connect_request
# ---------------------------------------------------------------------------


def test_coerce_connect_mock_rejected():
    with pytest.raises(
        RemoteError,
        match="requires kind='remote'; use a trusted startup script",
    ):
        coerce_connect_request({"kind": "mock"})


def test_coerce_connect_remote():
    req = coerce_connect_request({"kind": "remote", "ip": "192.168.1.1", "port": 8080})
    from zcu_tools.gui.session.services.connection import ConnectRemoteRequest

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


# ---------------------------------------------------------------------------
# method_specs ParamSpec validation for device.connect / device.disconnect
# (Fix 2 regression: validate_params must reject wrong types before handler)
# ---------------------------------------------------------------------------


def test_device_connect_spec_rejects_non_string_type_name():
    """type_name declared as STRING — an integer must be rejected with INVALID_PARAMS."""
    from zcu_tools.gui.app.measure.remote.method_specs import METHOD_SPECS
    from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
    from zcu_tools.gui.remote.param_spec import validate_params

    spec = METHOD_SPECS["device.connect"]
    with pytest.raises(RemoteError) as exc_info:
        validate_params(
            spec.params,
            {"type_name": 42, "name": "flux", "address": "GPIB::1"},
        )
    assert exc_info.value.code is ErrorCode.INVALID_PARAMS


def test_device_connect_spec_remember_defaults_to_true():
    """Omitting 'remember' must yield the default True (not None / missing)."""
    from zcu_tools.gui.app.measure.remote.method_specs import METHOD_SPECS
    from zcu_tools.gui.remote.param_spec import validate_params

    spec = METHOD_SPECS["device.connect"]
    result = validate_params(
        spec.params,
        {"type_name": "FakeDevice", "name": "flux", "address": "fake://"},
    )
    assert result["remember"] is True


def test_device_connect_spec_remember_explicit_false():
    """Explicitly passing remember=False is valid and preserved."""
    from zcu_tools.gui.app.measure.remote.method_specs import METHOD_SPECS
    from zcu_tools.gui.remote.param_spec import validate_params

    spec = METHOD_SPECS["device.connect"]
    result = validate_params(
        spec.params,
        {
            "type_name": "FakeDevice",
            "name": "flux",
            "address": "fake://",
            "remember": False,
        },
    )
    assert result["remember"] is False


def test_device_disconnect_spec_requires_name():
    """Omitting the required 'name' must be rejected with INVALID_PARAMS."""
    from zcu_tools.gui.app.measure.remote.method_specs import METHOD_SPECS
    from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
    from zcu_tools.gui.remote.param_spec import validate_params

    spec = METHOD_SPECS["device.disconnect"]
    with pytest.raises(RemoteError) as exc_info:
        validate_params(spec.params, {})
    assert exc_info.value.code is ErrorCode.INVALID_PARAMS


def test_device_disconnect_spec_remember_defaults_to_true():
    """Omitting 'remember' in disconnect must also default to True."""
    from zcu_tools.gui.app.measure.remote.method_specs import METHOD_SPECS
    from zcu_tools.gui.remote.param_spec import validate_params

    spec = METHOD_SPECS["device.disconnect"]
    result = validate_params(spec.params, {"name": "flux"})
    assert result["remember"] is True


# ---------------------------------------------------------------------------
# _field_type_and_choices — PEP 604 (X | None) union compatibility
# ---------------------------------------------------------------------------
# These tests verify that _field_type_and_choices handles PEP 604 Optional
# (types.UnionType) the same way it handles typing.Optional/typing.Union.
# See Item 1 of Phase 5 Step 0: `origin is typing.Union` misses UnionType.


def _setup_field_result(annotation: object) -> tuple[object, object]:
    """Project a synthetic device field with the supplied runtime annotation.

    Return its public wire type and choices, or None when choices are omitted.
    Raise AssertionError if fields or entries have the wrong shape or value is missing.
    """
    info_model = create_model(
        "__synthetic_device_info", __base__=BaseDeviceInfo, value=(annotation, None)
    )
    info = info_model(address="synthetic-address", type="synthetic-type")
    adapter = MagicMock()
    adapter.device_control.get_device_info.return_value = info
    result = h_device_setup_spec(adapter, {"name": "synthetic"})
    fields = result["fields"]
    assert isinstance(fields, list)
    for entry in fields:
        assert isinstance(entry, dict)
        if entry["name"] == "value":
            return entry["type"], entry.get("choices")
    raise AssertionError("Device setup fields are missing the synthetic value field")


def test_field_type_choices_typing_optional_unwraps():
    """Baseline: typing.Optional[float] is unwrapped to ('float', None)."""
    from typing import Optional

    # tolerated: Pass legacy typing.Optional to the public handler via a synthetic model.
    result = _setup_field_result(Optional[float])  # noqa: UP045
    assert result == ("float", None)


def test_field_type_choices_pep604_optional_unwraps():
    """PEP 604: float | None must also be unwrapped to ('float', None)."""
    # Construct the PEP 604 union at runtime (not via annotation string eval).
    annotation = float | None
    result = _setup_field_result(annotation)
    assert result == ("float", None)


def test_field_type_choices_pep604_optional_int_unwraps():
    """PEP 604: int | None must also be unwrapped to ('int', None)."""
    annotation = int | None
    result = _setup_field_result(annotation)
    assert result == ("int", None)
