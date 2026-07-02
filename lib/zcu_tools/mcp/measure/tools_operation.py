"""Measure MCP tools-operation override tools."""

from __future__ import annotations

import time
from typing import Any

from zcu_tools.mcp.measure.tool_context import (
    _WAIT_TRANSPORT_SLACK_SECONDS,
    MeasureToolContext,
    _is_timeout_error,
    bind_context,
    send_gui_rpc,
)


def _await_operation_by_handle(
    operation_id: int | None, what: str, timeout: float
) -> dict[str, Any]:
    """Block on a wire ``operation_id`` until it settles, or ``timeout`` s elapse;
    semantic result. The op-agnostic core of the generic gui_op_wait (ADR-0026 §8).

    Returns ``{status, waited_seconds[, message[, feedback]]}``:
    - 'finished': settled OK.
    - 'cancelled': user/agent cancelled the op. ``feedback`` carries the Stop
      reason when "Send & Stop" was used; absent on a plain cancel. NOT a raise
      (ADR-0025 §cancelled-wire — cancelled is a normal terminal outcome, not a
      crash; the agent reads feedback and re-plans).
    - 'user_feedback': a user-feedback string arrived before the op settled
      (ADR-0025). ``feedback`` carries the text; ``reason`` is 'user_feedback'.
      The operation is still running; the agent holds the handle and can re-await
      or cancel via the op-specific cancel tool.
    - 'timed_out': still running after the bounded wait — NOT a crash, no raise.
    - 'no_operation': no handle supplied / nothing tracked.
    ``waited_seconds`` is how long the wait actually blocked. A genuine
    ``failed`` outcome still raises (the agent must see it as an error).
    """
    if operation_id is None:
        return {
            "status": "no_operation",
            "message": f"No in-flight operation for {what}.",
        }
    start = time.monotonic()
    try:
        # Allow the bridge RPC a little slack beyond the op timeout so the
        # GUI-side timeout (a clean 'still running' signal) is what fires first,
        # not the socket round-trip ceiling.
        res = send_gui_rpc(
            "operation.await",
            {"operation_id": operation_id, "timeout": timeout},
            timeout + _WAIT_TRANSPORT_SLACK_SECONDS,
        )
    except RuntimeError as exc:
        if _is_timeout_error(exc):
            return {
                "status": "timed_out",
                "waited_seconds": round(time.monotonic() - start, 3),
                "message": f"{what} still in progress after {timeout}s.",
            }
        raise  # genuine failure — surfaces to the agent as an error
    # Unwrap the structured reason from the wire payload (ADR-0025).
    reason = res.get("reason", "completed")
    waited = round(time.monotonic() - start, 3)
    if reason == "user_feedback":
        feedback = res.get("feedback") or ""
        return {
            "status": "user_feedback",
            "reason": "user_feedback",
            "feedback": feedback,
            "waited_seconds": waited,
            "message": (
                f"User sent feedback while {what} was running. "
                "Treat this as a high-priority instruction and re-plan. "
                "The operation is still running — you may gui_tab_run_cancel or re-await."
            ),
        }
    status = res.get("status", "finished")
    if status == "cancelled":
        # Structured cancellation: return status + optional Stop reason. Not a
        # raise — cancelled is a normal terminal outcome (ADR-0025 §cancelled-wire).
        out: dict[str, Any] = {
            "status": "cancelled",
            "waited_seconds": waited,
            "message": f"{what} was cancelled.",
        }
        feedback = res.get("feedback")
        if feedback:
            out["feedback"] = feedback
        return out
    return {
        "status": status,
        "waited_seconds": waited,
        "message": f"{what} completed.",
    }


def _slim_progress(progress: dict[str, Any]) -> dict[str, Any]:
    """Project the wire progress payload down to the fields an agent acts on.

    The wire ``operation.progress`` carries Qt-scaled counters (maximum / value /
    n / total) that the GUI's progress widget needs but the agent does not — it
    reasons over the human-readable ``format`` line and the ``percent`` only.
    Keep ``{token, format, percent}`` per bar; the precision/wire layer is left
    untouched (this folding is mcp-side policy).
    """
    bars = progress.get("bars", [])
    return {
        "active": progress.get("active", False),
        "bars": [
            {
                "token": b.get("token"),
                "format": b.get("format"),
                "percent": b.get("percent"),
            }
            for b in bars
        ],
    }


_POLL_DRAIN_CAP = 4096


def _poll_operation_by_handle(operation_id: int | None, what: str) -> dict[str, Any]:
    """Non-blocking status of a wire ``operation_id`` (no event needed). The
    op-agnostic core of the generic gui_op_poll (ADR-0026 §8).

    DRAINS every currently-buffered user-feedback Message (zero-timeout awaits in
    a loop) and returns them as a ``feedback`` list, then maps the FINAL outcome
    onto a plain status: 'running' (still in flight — even right after draining
    feedback), 'finished' (genuinely settled OK), 'cancelled'/'failed' (terminal —
    poll reports as a status, it does NOT raise like the blocking wait), or
    'no_operation' (no handle supplied / already reaped). Lets an agent that
    started a slow op go do other work, see every queued nudge, and check back
    without blocking.

    INVARIANT: NEVER returns 'finished' unless the op is genuinely settled-finished.
    A live op (the zero-timeout await raised TIMEOUT) is 'running', even if feedback
    was just drained from it. user-feedback Messages are one-shot FIFO — consuming
    them here is intended; a later gui_op_wait will not re-deliver them, though the
    sticky terminal outcome remains re-readable by wait.
    """
    if operation_id is None:
        return {"status": "no_operation", "message": f"No operation for {what}."}

    drained: list[str] = []
    for _ in range(_POLL_DRAIN_CAP):
        try:
            res = send_gui_rpc(
                "operation.await",
                {"operation_id": operation_id, "timeout": 0.0},
                _WAIT_TRANSPORT_SLACK_SECONDS,
            )
        except RuntimeError as exc:
            if _is_timeout_error(exc):
                # Queue drained and op NOT settled — still running. Fold the live
                # progress bars into the reply (slimmed to {token, format, percent};
                # Qt-scaled counters dropped here) so the agent watches progress
                # without a separate tool call. Mirror of today's running branch.
                progress = send_gui_rpc(
                    "operation.progress", {"operation_id": operation_id}
                )
                return _with_feedback(
                    {
                        "status": "running",
                        "message": f"{what} still in progress.",
                        **_slim_progress(progress),
                    },
                    drained,
                )
            # terminal error — report as status rather than raising (poll is a
            # query, not an await). A user-initiated cancel is a distinct,
            # non-failure outcome: surface it as 'cancelled' so the agent need not
            # parse the message to tell "it crashed" from "I cancelled it" (the
            # wire carries reason='cancelled', read structurally via reason attr).
            reason = getattr(exc, "reason", None)
            if reason == "cancelled":
                return _with_feedback(
                    {"status": "cancelled", "message": f"{what} was cancelled."},
                    drained,
                )
            return _with_feedback(
                {"status": "failed", "message": f"{what}: {exc}"}, drained
            )

        # Successful payload: either a buffered user-feedback Message (consumed —
        # keep draining) or the sticky terminal outcome (stop). Mirror
        # _await_operation_by_handle's classification of the SAME payload shape.
        reason = res.get("reason", "completed")
        if reason == "user_feedback":
            text = res.get("feedback") or ""
            drained.append(text)
            continue
        status = res.get("status", "finished")
        if status == "cancelled":
            # Structured cancellation: status + optional Stop reason. The Stop
            # reason rides the dedicated message key; the drained-feedback list is
            # separate (queued nudges that preceded the terminal).
            out: dict[str, Any] = {
                "status": "cancelled",
                "message": f"{what} was cancelled.",
            }
            cancel_reason = res.get("feedback")
            if cancel_reason:
                out["message"] = f"{what} was cancelled: {cancel_reason}"
                out["stop_reason"] = cancel_reason
            return _with_feedback(out, drained)
        if status == "failed":
            # Failure surfaced as a structured payload (rare — usually a raise).
            return _with_feedback(
                {"status": "failed", "message": f"{what}: {res.get('error')}"},
                drained,
            )
        return _with_feedback(
            {"status": status, "message": f"{what} completed."}, drained
        )

    # Drain cap hit: a pathological feedback producer. Do NOT claim finished — the
    # op may well still be live. Report running and let the agent re-poll.
    return _with_feedback(
        {
            "status": "running",
            "message": (
                f"{what} still in progress (drained {_POLL_DRAIN_CAP} feedback "
                "messages without reaching a terminal — re-poll)."
            ),
        },
        drained,
    )


def _with_feedback(reply: dict[str, Any], drained: list[str]) -> dict[str, Any]:
    """Attach the drained user-feedback list to a poll reply when non-empty.

    Queued nudges may precede a terminal, so a finished/cancelled/running reply can
    all carry drained messages. The list is in strict arrival (FIFO) order.
    """
    if drained:
        reply["feedback"] = list(drained)
    return reply


def tool_gui_op_poll(arguments: dict[str, Any]) -> dict[str, Any]:
    """Non-blocking status of any in-flight operation, by ``handle`` (ADR-0026 §8).

    ``handle`` is the opaque token a START tool (gui_tab_run_start /
    gui_tab_analyze_start / gui_tab_post_analyze_start / gui_device_*) returned in
    its reply. NEVER raises. Reports the TRUE status: running (still live — even
    right after feedback was drained) | finished (genuinely settled OK) | cancelled
    (with the Stop reason in 'stop_reason' / message when "Send & Stop" was used) |
    failed | no_operation. While 'running' the reply folds the live progress bars
    (active, bars[token/format/percent]).

    DRAINS all currently-buffered user feedback in one call and returns it as a
    ``feedback`` LIST (every queued nudge, in arrival order) so the agent sees every
    message — present on ANY status (a nudge may precede a terminal). Draining
    CONSUMES those messages: a later gui_op_wait will NOT re-deliver them, though
    the sticky terminal outcome stays re-readable by gui_op_wait. Returns only the
    status (+progress +feedback): the op's product (figure/summary/snapshot) is read
    from the START finished reply or the matching typed getter.
    """
    handle = int(arguments["handle"])
    return _poll_operation_by_handle(handle, "operation")


def tool_gui_op_wait(arguments: dict[str, Any]) -> dict[str, Any]:
    """Block until any in-flight operation settles, by ``handle`` (ADR-0026 §8).

    ``handle`` is the opaque token a START tool returned. Blocks up to ``timeout``
    seconds. Returns {status, waited_seconds[, feedback]}: finished | cancelled
    (read optional 'feedback' for the Stop reason — NOT a raise) | user_feedback
    (op still running, agent re-plans then re-waits or cancels) | timed_out (still
    running — re-wait or gui_op_poll) | no_operation. RAISES only on a genuine
    failure. Returns only the status: the op's product (figure/summary/snapshot)
    is read from the START finished reply or the matching typed getter.
    """
    handle = int(arguments["handle"])
    timeout = float(arguments.get("timeout", 120.0))
    return _await_operation_by_handle(handle, "operation", timeout)


NON_GENERATED_METHODS = frozenset(
    {
        "operation.await",
        "operation.progress",
    }
)


OVERRIDE_TOOLS: dict[str, dict[str, Any]] = {
    "gui_op_poll": {
        "handler": tool_gui_op_poll,
        "description": (
            "Non-blocking status of ANY in-flight operation, by 'handle' (the "
            "opaque token a START tool — gui_tab_run_start / gui_tab_analyze_start "
            "/ gui_tab_post_analyze_start / gui_device_* — returned in its reply). "
            "NEVER raises; returns {status, ...} with status in "
            "finished|running|cancelled|failed|no_operation. While 'running' the "
            "reply folds the live progress bars (active, bars[token/format/percent]) "
            "— no separate progress tool. Reports only the status: the op's product "
            "(figure/summary/snapshot) comes from the START finished reply or the "
            "matching typed getter (e.g. gui_tab_get_current_figure, "
            "gui_tab_get_analyze_result, gui_device_snapshot)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "handle": {
                    "type": "integer",
                    "description": "Operation handle from a START tool's reply",
                }
            },
            "required": ["handle"],
        },
    },
    "gui_op_wait": {
        "handler": tool_gui_op_wait,
        "description": (
            "Block until ANY in-flight operation settles, by 'handle' (the opaque "
            "token a START tool returned). Blocks up to 'timeout' seconds and holds "
            "your turn; for a long op prefer gui_op_poll (non-blocking) or run this "
            "from a background agent. Returns {status, waited_seconds, ...} with "
            "status in finished|cancelled|user_feedback|timed_out|no_operation: "
            "'cancelled' (read optional 'feedback' for the Stop reason — NOT a "
            "raise), 'user_feedback' (op STILL running — treat 'feedback' as a "
            "high-priority instruction, re-plan, then re-wait or cancel via the "
            "op-specific cancel tool), 'timed_out' (still running — re-wait or "
            "gui_op_poll). RAISES only on a genuine failure. Reports only the "
            "status: read the op's product via the matching typed getter."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "handle": {
                    "type": "integer",
                    "description": "Operation handle from a START tool's reply",
                },
                "timeout": {
                    "type": "number",
                    "description": "Seconds to wait (default 120)",
                },
            },
            "required": ["handle"],
        },
    },
}


def build_override_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    bind_context(ctx)
    return OVERRIDE_TOOLS
