"""operation.await dispatch handler.

The handler is off_main_thread (blocks the IO worker). It calls
operation_control.await_operation(operation_id, timeout) and shapes the AwaitResult into
a wire result (ADR-0025 §cancelled-wire):
  - completed/cancelled → structured {reason:'completed', status:'cancelled',
    feedback?} (NOT a raise; feedback present only when a Stop reason was latched).
  - completed/failed → RemoteError(PRECONDITION_FAILED, reason='failed').
  - timeout → RemoteError(TIMEOUT).
  - user_feedback → {reason:'user_feedback', feedback:<str>} (non-terminal).
  - completed/finished → {reason:'completed', status:'finished'}.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.services.remote.dispatch import METHOD_REGISTRY
from zcu_tools.gui.app.main.services.remote.handlers.operation import (
    _h_operation_progress,
)
from zcu_tools.gui.app.main.services.remote.service import RemoteControlAdapter
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.operation_handles import AwaitResult, OperationOutcome


def _HANDLER(ctrl, params):
    # Generic operation handlers must not require the giant ctrl surface.
    adapter = cast(Any, SimpleNamespace(operation_control=ctrl))
    return METHOD_REGISTRY["operation.await"].handler(adapter, params)


def _ctrl(result: AwaitResult | None) -> MagicMock:
    ctrl = MagicMock()
    ctrl.await_operation.return_value = result
    return ctrl


def test_off_main_thread_flag_set():
    assert METHOD_REGISTRY["operation.await"].off_main_thread is True


def test_progress_uses_operation_control_without_ctrl():
    ctrl = MagicMock()
    ctrl.get_operation_progress.return_value = ()
    adapter = cast(RemoteControlAdapter, SimpleNamespace(operation_control=ctrl))

    out = _h_operation_progress(adapter, {"operation_id": 7})

    assert out == {"active": False, "bars": []}
    ctrl.get_operation_progress.assert_called_once_with(7)


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


# ---------------------------------------------------------------------------
# cancelled path (ADR-0025 §cancelled-wire) — structured result, NOT a raise
# ---------------------------------------------------------------------------


def test_cancelled_with_feedback_returns_structured():
    # Settled-cancelled with a Stop reason (Send & Stop scenario):
    # the feedback is folded by _make_completed and must reach the wire payload.
    ctrl = _ctrl(
        AwaitResult(
            reason="completed",
            outcome=OperationOutcome("cancelled"),
            feedback="stop reason from user",
        )
    )
    out = _HANDLER(ctrl, {"operation_id": 7, "timeout": 5.0})
    assert out["reason"] == "completed"
    assert out["status"] == "cancelled"
    assert out["feedback"] == "stop reason from user"


def test_cancelled_without_feedback_no_raise():
    # Plain cancel (no Stop reason): status='cancelled', no feedback key.
    ctrl = _ctrl(
        AwaitResult(
            reason="completed",
            outcome=OperationOutcome("cancelled"),
            feedback=None,
        )
    )
    out = _HANDLER(ctrl, {"operation_id": 7, "timeout": 5.0})
    assert out["reason"] == "completed"
    assert out["status"] == "cancelled"
    assert "feedback" not in out


def test_cancelled_does_not_raise():
    # Regression guard: a cancelled outcome must never raise (pre-fix behavior).
    ctrl = _ctrl(
        AwaitResult(
            reason="completed",
            outcome=OperationOutcome("cancelled"),
            feedback=None,
        )
    )
    out = _HANDLER(ctrl, {"operation_id": 7, "timeout": 5.0})  # must not raise
    assert out["status"] == "cancelled"


# ---------------------------------------------------------------------------
# timeout path
# ---------------------------------------------------------------------------


def test_timeout_raises_timeout():
    ctrl = _ctrl(AwaitResult(reason="timeout"))
    with pytest.raises(RemoteError) as ei:
        _HANDLER(ctrl, {"operation_id": 7, "timeout": 0.1})
    assert ei.value.code == ErrorCode.TIMEOUT


# ---------------------------------------------------------------------------
# user_feedback path (ADR-0025)
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


# ---------------------------------------------------------------------------
# non-regression: finished / failed still work as before
# ---------------------------------------------------------------------------


def test_finished_not_affected_by_cancelled_change():
    ctrl = _ctrl(AwaitResult(reason="completed", outcome=OperationOutcome("finished")))
    out = _HANDLER(ctrl, {"operation_id": 42, "timeout": 1.0})
    assert out["status"] == "finished"
    assert "feedback" not in out


def test_failed_still_raises_not_structured():
    ctrl = _ctrl(
        AwaitResult(reason="completed", outcome=OperationOutcome("failed", "boom"))
    )
    with pytest.raises(RemoteError):
        _HANDLER(ctrl, {"operation_id": 42, "timeout": 1.0})
