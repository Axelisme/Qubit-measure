"""Operation remote handlers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter

from ._common import Handler

logger = logging.getLogger(__name__)


def _progress_bars_wire(bars) -> Mapping[str, object]:
    """Shared run/device progress projection from live (token, ProgressBarModel)
    pairs — derived fields computed live at this read (the SSOT is the model)."""
    if not bars:
        return {"active": False, "bars": []}
    return {
        "active": True,
        "bars": [
            {
                "token": token,
                "format": m.format(),
                "maximum": m.qt_maximum(),
                "value": m.qt_value(),
                "percent": m.percent(),
                "n": m.n,
                "total": m.total,
            }
            for token, m in bars
        ],
    }


def _h_operation_await(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # off_main_thread handler: blocks the IO worker thread on the handle's
    # thread-safe registry (never touches main-thread-owned state). Returns a
    # structured payload with reason in {'completed', 'user_feedback', 'timeout'}
    # (ADR-0025). 'cancelled' is returned as structured data (status='cancelled',
    # optional feedback from the Stop reason); 'failed' is still raised as
    # PRECONDITION_FAILED so the agent sees it as an error.
    operation_id = int(params["operation_id"])  # type: ignore[arg-type]
    timeout = float(params["timeout"])  # type: ignore[arg-type]
    result = adapter.ctrl.await_operation(operation_id, timeout)
    if result is None:
        # Should not happen with the new API, but guard for forward-compat.
        raise RemoteError(
            ErrorCode.TIMEOUT,
            f"operation {operation_id} did not complete within {timeout}s",
        )
    if result.reason == "timeout":
        raise RemoteError(
            ErrorCode.TIMEOUT,
            f"operation {operation_id} did not complete within {timeout}s",
        )
    if result.reason == "user_feedback":
        # Non-terminal: operation still running; feedback delivered to the agent.
        return {
            "reason": "user_feedback",
            "feedback": result.feedback,
        }
    # reason == 'completed'
    outcome = result.outcome
    assert outcome is not None  # invariant: completed always has outcome
    if outcome.status == "cancelled":
        # Structured cancellation: return status + optional Stop reason so the
        # agent gets the full picture in one reply (ADR-0025 §cancelled-wire).
        # The feedback field is only present when a Stop reason was latched
        # (i.e. "Send & Stop" was used); a plain cancel has no feedback.
        payload: dict[str, object] = {"reason": "completed", "status": "cancelled"}
        if result.feedback:
            payload["feedback"] = result.feedback
        return payload
    if outcome.status == "failed":
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            outcome.error or "operation failed",
            reason="failed",
        )
    return {"reason": "completed", "status": outcome.status}


def _h_operation_progress(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # Live (token, ProgressBarModel) pairs for one operation (run or device
    # setup alike, keyed by operation_id — the SSOT); _progress_bars_wire reads
    # their methods at this point. The mcp poll folds this into its reply.
    operation_id = int(params["operation_id"])  # type: ignore[arg-type]
    return _progress_bars_wire(adapter.ctrl.get_operation_progress(operation_id))


HANDLERS: dict[str, Handler] = {
    "operation.await": _h_operation_await,
    "operation.progress": _h_operation_progress,
}
