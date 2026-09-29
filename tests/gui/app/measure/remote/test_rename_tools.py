"""context.ml_rename_module / rename_ml_waveform dispatch handlers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

from ._helpers import dispatch_handler as _dispatch  # noqa: E402


def test_rename_module_drives_controller():
    ctrl = MagicMock()
    _dispatch(ctrl, "context.ml_rename_module", {"old": "a", "new": "b"})
    ctrl.rename_ml_module.assert_called_once_with("a", "b")


def test_rename_waveform_drives_controller():
    ctrl = MagicMock()
    _dispatch(ctrl, "context.ml_rename_waveform", {"old": "w1", "new": "w2"})
    ctrl.rename_ml_waveform.assert_called_once_with("w1", "w2")


def test_rename_clash_is_precondition_failed():
    ctrl = MagicMock()
    ctrl.rename_ml_module.side_effect = FailedPreconditionError(
        "A module named 'b' already exists."
    )
    with pytest.raises(RemoteError) as exc:
        _dispatch(ctrl, "context.ml_rename_module", {"old": "a", "new": "b"})
    assert exc.value.code is ErrorCode.PRECONDITION_FAILED


def test_rename_missing_is_precondition_failed():
    ctrl = MagicMock()
    ctrl.rename_ml_module.side_effect = FailedPreconditionError("No module named 'a'.")
    with pytest.raises(RemoteError) as exc:
        _dispatch(ctrl, "context.ml_rename_module", {"old": "a", "new": "b"})
    assert exc.value.code is ErrorCode.PRECONDITION_FAILED
