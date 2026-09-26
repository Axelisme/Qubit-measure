from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.value_lookup import (
    MissingValue,
    ProviderError,
    UnavailableValue,
    ValueInfo,
    ValueTypeError,
)

from ._helpers import dispatch_handler as _dispatch


def test_value_list_projects_registered_info() -> None:
    ctrl = MagicMock()
    ctrl.list_value_sources.return_value = (
        ValueInfo(
            "device.flux.value",
            float,
            "device:flux",
            "Named device cached value.",
        ),
    )

    res = _dispatch(ctrl, "value.list", {})

    assert res["values"] == [
        {
            "key": "device.flux.value",
            "type": "float",
            "owner": "device:flux",
            "description": "Named device cached value.",
        }
    ]


def test_value_read_returns_info_and_resolved_value() -> None:
    ctrl = MagicMock()
    ctrl.read_value_source.return_value = (
        ValueInfo("predictor.loaded", bool, "predictor", "Whether a predictor exists."),
        True,
    )

    res = _dispatch(ctrl, "value.read", {"key": "predictor.loaded", "type": "bool"})

    ctrl.read_value_source.assert_called_once_with("predictor.loaded", "bool")
    assert res == {
        "key": "predictor.loaded",
        "type": "bool",
        "owner": "predictor",
        "description": "Whether a predictor exists.",
        "value": True,
    }


def test_value_read_missing_source_is_invalid_params() -> None:
    ctrl = MagicMock()
    ctrl.read_value_source.side_effect = MissingValue("ghost", "missing")

    with pytest.raises(RemoteError) as exc:
        _dispatch(ctrl, "value.read", {"key": "ghost"})

    assert exc.value.code is ErrorCode.INVALID_PARAMS


def test_value_read_type_mismatch_is_invalid_params() -> None:
    ctrl = MagicMock()
    ctrl.read_value_source.side_effect = ValueTypeError("x", "wrong type")

    with pytest.raises(RemoteError) as exc:
        _dispatch(ctrl, "value.read", {"key": "x", "type": "str"})

    assert exc.value.code is ErrorCode.INVALID_PARAMS


def test_value_read_unavailable_source_is_precondition_failed() -> None:
    ctrl = MagicMock()
    ctrl.read_value_source.side_effect = UnavailableValue("x", "not available now")

    with pytest.raises(RemoteError) as exc:
        _dispatch(ctrl, "value.read", {"key": "x"})

    assert exc.value.code is ErrorCode.PRECONDITION_FAILED


def test_value_read_provider_failure_is_controller_error() -> None:
    ctrl = MagicMock()
    error = ProviderError("x", "owner", RuntimeError("bad"))
    ctrl.read_value_source.side_effect = error

    with pytest.raises(ProviderError) as exc:
        _dispatch(ctrl, "value.read", {"key": "x"})

    assert exc.value is error
