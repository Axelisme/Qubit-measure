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
    coerce_connect_request,
    h_device_snapshot,
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
    adapter.device_control.get_device_snapshot.return_value.info = info
    result = h_device_snapshot(adapter, {"name": "synthetic"})
    snapshot = result["snapshot"]
    if not isinstance(snapshot, dict):
        raise AssertionError("Device snapshot is not an object")
    fields = snapshot["fields"]
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
