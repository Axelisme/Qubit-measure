"""operation.await dispatch handler.

The handler is off_main_thread (blocks the IO worker). It calls
operation_control.await_operation(operation_id, timeout) and shapes the AwaitResult into
a wire result (ADR-0066):
  - completed/cancelled → structured {reason:'completed', status:'cancelled',
    feedback?} (NOT a raise; feedback present only when a Stop reason was latched).
  - completed/failed → structured failed/error, not a failed tool call.
  - timeout → structured timeout/running signal.
  - user_feedback → {reason:'user_feedback', feedback:<str>} (non-terminal).
  - completed/finished → {reason:'completed', status:'finished'}.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.measure.remote.dispatch import METHOD_REGISTRY
from zcu_tools.gui.app.measure.remote.method_entries import METHOD_ENTRIES
from zcu_tools.gui.app.measure.remote.method_entries._registry import (
    build_dispatch_registry,
)
from zcu_tools.gui.app.measure.services.operation_control import OperationControlFacet
from zcu_tools.gui.event_bus import EventOrigin
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.operation_handles import (
    AwaitResult,
    OperationHandles,
    OperationOutcome,
)

from tests.gui.app.measure.services._operation_owner_fakes import (
    DeviceOperationOwner,
    SaveOperationOwner,
    TabOperationOwner,
    UnusedProgress,
)


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


@pytest.mark.parametrize("method", ["tab.get_cfg", "tab.load_data"])
def test_remote_registry_rejects_off_main_reveal_or_guard(method: str) -> None:
    entry = next(item for item in METHOD_ENTRIES if item.method == method)
    # Suppress the old write-receipt restriction so this tests the guard/reveal
    # declaration rather than a separate reason to reject off-main writes.
    candidate = replace(
        entry,
        spec=replace(entry.spec, off_main_thread=True),
        agent=replace(entry.agent, refresh_after_write=False),
    )
    with pytest.raises(ValueError, match="owner thread"):
        build_dispatch_registry((candidate,))


# ---------------------------------------------------------------------------
# completed path
# ---------------------------------------------------------------------------


def test_unknown_and_evicted_handle_do_not_become_finished():
    handles = OperationHandles()
    control = OperationControlFacet(
        save=SaveOperationOwner(),
        handles=handles,
        progress=UnusedProgress(),
        run_analyze=TabOperationOwner(),
        device=DeviceOperationOwner(),
    )
    for op in (777,):
        with pytest.raises(RemoteError) as exc_info:
            _HANDLER(control, {"operation_id": op, "timeout": 0})
        assert exc_info.value.code == ErrorCode.INVALID_PARAMS
        assert exc_info.value.reason == "unknown_op"

    first = handles.create(origin=EventOrigin(kind="user"))
    handles.settle(first, OperationOutcome("failed", "original failure"))
    for _ in range(40):
        op = handles.create(origin=EventOrigin(kind="user"))
        handles.settle(op, OperationOutcome("finished"))
    with pytest.raises(RemoteError) as exc_info:
        _HANDLER(control, {"operation_id": first, "timeout": 0})
    assert exc_info.value.reason == "unknown_op"


def test_invalid_timeout_fails_before_await():
    with pytest.raises(RemoteError) as exc_info:
        _HANDLER(
            _ctrl(AwaitResult(reason="timeout")), {"operation_id": 1, "timeout": 301}
        )
    assert exc_info.value.reason == "invalid_timeout"


def test_finished_returns_reason_and_status():
    ctrl = _ctrl(AwaitResult(reason="completed", outcome=OperationOutcome("finished")))
    out = _HANDLER(ctrl, {"operation_id": 7, "timeout": 5.0})
    assert out == {"reason": "completed", "status": "finished"}
    ctrl.await_operation.assert_called_once_with(7, 5.0)


def test_failed_returns_status_and_error():
    ctrl = _ctrl(
        AwaitResult(
            reason="completed", outcome=OperationOutcome("failed", "hardware boom")
        )
    )
    assert _HANDLER(ctrl, {"operation_id": 7, "timeout": 5.0}) == {
        "reason": "completed",
        "status": "failed",
        "error": {"reason": "failed", "message": "hardware boom"},
    }


# ---------------------------------------------------------------------------
# cancelled path (ADR-0066) — structured result, NOT a raise
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


def test_timeout_returns_running_signal():
    ctrl = _ctrl(AwaitResult(reason="timeout"))
    assert _HANDLER(ctrl, {"operation_id": 7, "timeout": 0.1}) == {"reason": "timeout"}


# ---------------------------------------------------------------------------
# user_feedback path (ADR-0066)
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


def test_failed_after_another_completed_read_stays_structured():
    ctrl = _ctrl(
        AwaitResult(reason="completed", outcome=OperationOutcome("failed", "boom"))
    )
    assert _HANDLER(ctrl, {"operation_id": 42, "timeout": 1.0})["status"] == "failed"
