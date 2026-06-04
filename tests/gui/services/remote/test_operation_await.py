"""operation.await dispatch handler.

The handler is off_main_thread (blocks the IO worker). It calls
ctrl.await_operation(operation_id, timeout) and shapes the OperationOutcome into
a wire result, turning failed/cancelled/timeout into RemoteError.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.services.operation_gate import OperationOutcome
from zcu_tools.gui.app.main.services.remote.dispatch import METHOD_REGISTRY
from zcu_tools.gui.app.main.services.remote.errors import ErrorCode, RemoteError

from ._helpers import dispatch_handler


def _HANDLER(ctrl, params):
    # Handlers receive the adapter (ADR-0013); wrap ctrl in an adapter stub.
    return dispatch_handler(ctrl, "operation.await", params)


def _ctrl(outcome) -> MagicMock:
    ctrl = MagicMock()
    ctrl.await_operation.return_value = outcome
    return ctrl


def test_off_main_thread_flag_set():
    assert METHOD_REGISTRY["operation.await"].off_main_thread is True


def test_finished_returns_status():
    ctrl = _ctrl(OperationOutcome("finished"))
    out = _HANDLER(ctrl, {"operation_id": 7, "timeout": 5.0})
    assert out == {"status": "finished"}
    ctrl.await_operation.assert_called_once_with(7, 5.0)


def test_failed_raises_precondition():
    ctrl = _ctrl(OperationOutcome("failed", "hardware boom"))
    with pytest.raises(RemoteError) as ei:
        _HANDLER(ctrl, {"operation_id": 7, "timeout": 5.0})
    assert ei.value.code == ErrorCode.PRECONDITION_FAILED
    assert ei.value.reason == "failed"
    assert "hardware boom" in ei.value.message


def test_cancelled_raises_precondition():
    ctrl = _ctrl(OperationOutcome("cancelled", "user cancelled"))
    with pytest.raises(RemoteError) as ei:
        _HANDLER(ctrl, {"operation_id": 7, "timeout": 5.0})
    assert ei.value.reason == "cancelled"


def test_timeout_raises_timeout():
    ctrl = _ctrl(None)  # await_outcome returns None on timeout
    with pytest.raises(RemoteError) as ei:
        _HANDLER(ctrl, {"operation_id": 7, "timeout": 0.1})
    assert ei.value.code == ErrorCode.TIMEOUT
