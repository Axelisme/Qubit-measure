"""operation.await dispatch handler.

The handler is off_main_thread (blocks the IO worker). It calls
ctrl.await_operation(operation_id, timeout) and shapes the AwaitResult into
a wire result, turning failed/cancelled/timeout into RemoteError (ADR-0019).

ADR-0023 extension: when the ctrl returns reason='user_feedback', the handler
returns {"reason": "user_feedback", "feedback": <str>} to the caller instead
of raising. reason='completed' with a success outcome returns
{"reason": "completed", "status": <str>}.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.services.remote.dispatch import METHOD_REGISTRY
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.operation_handles import AwaitResult, OperationOutcome

from ._helpers import dispatch_handler


def _HANDLER(ctrl, params):
    # Handlers receive the adapter (ADR-0013); wrap ctrl in an adapter stub.
    return dispatch_handler(ctrl, "operation.await", params)


def _ctrl(result: AwaitResult | None) -> MagicMock:
    ctrl = MagicMock()
    ctrl.await_operation.return_value = result
    return ctrl


def test_off_main_thread_flag_set():
    assert METHOD_REGISTRY["operation.await"].off_main_thread is True


# ---------------------------------------------------------------------------
# completed path
# ---------------------------------------------------------------------------


def test_finished_returns_reason_and_status():
    ctrl = _ctrl(AwaitResult(reason="completed", outcome=OperationOutcome("finished")))
    out = _HANDLER(ctrl, {"operation_id": 7, "timeout": 5.0})
    assert out == {"reason": "completed", "status": "finished"}
    ctrl.await_operation.assert_called_once_with(7, 5.0)


def test_failed_raises_precondition():
    ctrl = _ctrl(
        AwaitResult(
            reason="completed", outcome=OperationOutcome("failed", "hardware boom")
        )
    )
    with pytest.raises(RemoteError) as ei:
        _HANDLER(ctrl, {"operation_id": 7, "timeout": 5.0})
    assert ei.value.code == ErrorCode.PRECONDITION_FAILED
    assert ei.value.reason == "failed"
    assert "hardware boom" in ei.value.message


def test_cancelled_raises_precondition():
    ctrl = _ctrl(
        AwaitResult(
            reason="completed", outcome=OperationOutcome("cancelled", "user cancelled")
        )
    )
    with pytest.raises(RemoteError) as ei:
        _HANDLER(ctrl, {"operation_id": 7, "timeout": 5.0})
    assert ei.value.reason == "cancelled"


# ---------------------------------------------------------------------------
# timeout path
# ---------------------------------------------------------------------------


def test_timeout_raises_timeout():
    ctrl = _ctrl(AwaitResult(reason="timeout"))
    with pytest.raises(RemoteError) as ei:
        _HANDLER(ctrl, {"operation_id": 7, "timeout": 0.1})
    assert ei.value.code == ErrorCode.TIMEOUT


# ---------------------------------------------------------------------------
# user_feedback path (ADR-0023)
# ---------------------------------------------------------------------------


def test_user_feedback_returns_feedback_payload():
    ctrl = _ctrl(AwaitResult(reason="user_feedback", feedback="recalibrate"))
    out = _HANDLER(ctrl, {"operation_id": 7, "timeout": 5.0})
    assert out["reason"] == "user_feedback"
    assert "recalibrate" in str(out["feedback"])


def test_user_feedback_multiple_messages_forwarded():
    ctrl = _ctrl(AwaitResult(reason="user_feedback", feedback="line 1\nline 2"))
    out = _HANDLER(ctrl, {"operation_id": 7, "timeout": 5.0})
    assert out["reason"] == "user_feedback"
    assert "line 1" in str(out["feedback"])
    assert "line 2" in str(out["feedback"])
