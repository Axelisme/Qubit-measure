"""Operation remote handlers."""

# Method entries resolve these handlers by string reference at runtime.

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING

from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

from ._wire_values import cancel_scope_arguments

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


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
                "eta_s": m.remaining(),
                "n": m.n,
                "total": m.total,
            }
            for token, m in bars
        ],
    }


def h_operation_active(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {
        "operations": [
            {"op": op.op, "tab": op.tab, "kind": op.kind}
            for op in adapter.operation_control.active_operations()
        ]
    }


def h_operation_cancel(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    operation_id = params["operation_id"]
    if not isinstance(operation_id, int) or isinstance(operation_id, bool):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "operation_id must be an integer")
    try:
        status = adapter.operation_control.cancel_operation(operation_id)
    except KeyError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, str(exc), reason="unknown_op"
        ) from exc
    return {"status": status}


def h_operation_cancel_scope(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Match one scope in an exact GUI Run; do not use the generic cancel hook."""
    operation_id = params["operation_id"]
    if not isinstance(operation_id, int) or isinstance(operation_id, bool):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "operation_id must be an integer")
    scope, expected = cancel_scope_arguments(params)
    return asdict(
        adapter.run_analyze_control.cancel_scope(operation_id, scope, expected=expected)
    )


def h_operation_status_scope(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Read only the tree owned by the exact active GUI Run handle."""
    operation_id = params["operation_id"]
    if not isinstance(operation_id, int) or isinstance(operation_id, bool):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "operation_id must be an integer")
    return {"active_path": adapter.run_analyze_control.status_scope(operation_id)}


def h_operation_await(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # Off-main: only the thread-safe handle channel, never owner-thread state.
    # tolerated: The required_integer method parameter validates operation_id before dispatch.
    operation_id = int(params["operation_id"])  # pyright: ignore[reportArgumentType]
    # tolerated: The default_number method parameter validates timeout before dispatch.
    timeout = float(params["timeout"])  # pyright: ignore[reportArgumentType]
    if not 0 <= timeout <= 300:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            "timeout must be between 0 and 300 seconds",
            reason="invalid_timeout",
        )
    try:
        result = adapter.operation_control.await_operation(operation_id, timeout)
    except KeyError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, str(exc), reason="unknown_op"
        ) from exc
    if result.reason == "timeout":
        return {"reason": "timeout"}
    outcome = result.outcome
    if outcome is None:
        raise RuntimeError("completed operation is missing its outcome")
    if outcome.status == "cancelled":
        # Structured cancellation: return status + optional Stop reason so the
        # agent gets the full picture in one reply (ADR-0066).
        # The feedback field is only present when a Stop reason was latched
        # (i.e. stop(reason) was used); a plain cancel has no feedback.
        payload: dict[str, object] = {"reason": "completed", "status": "cancelled"}
        if result.feedback:
            payload["feedback"] = result.feedback
        return payload
    if outcome.status == "failed":
        return {
            "reason": "completed",
            "status": "failed",
            "error": {
                "reason": "failed",
                "message": outcome.error or "operation failed",
            },
        }
    return {"reason": "completed", "status": outcome.status}


def h_operation_progress(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # Live (token, ProgressBarModel) pairs for one operation (run or device
    # setup alike, keyed by operation_id — the SSOT); _progress_bars_wire reads
    # their methods at this point. The mcp poll folds this into its reply.
    # tolerated: The required_integer method parameter validates operation_id before dispatch.
    operation_id = int(params["operation_id"])  # pyright: ignore[reportArgumentType]
    result = dict(
        _progress_bars_wire(
            adapter.operation_control.get_operation_progress(operation_id)
        )
    )
    result["elapsed_s"] = adapter.operation_control.elapsed_seconds(operation_id)
    return result
