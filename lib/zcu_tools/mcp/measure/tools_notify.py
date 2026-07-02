"""Measure MCP tools-notify override tools."""

from __future__ import annotations

from typing import Any

from zcu_tools.mcp.measure.tool_context import (
    _WAIT_TRANSPORT_SLACK_SECONDS,
    MeasureToolContext,
    bind_context,
    send_gui_rpc,
)

_NOTIFY_CONSUMER_SLACK: float = 10.0


def tool_gui_prompt_user(arguments: dict[str, Any]) -> dict[str, Any]:
    """Blocking request-reply prompt to the user. BLOCKS the turn until they respond.

    Serially composes notify.open (main thread: mint token + open dialog) and
    notify.await (off-main: block until Reply/Dismiss/QTimer). Neither method is a
    session-tracked start-op, so no operation_id is captured. There is NO poll or
    cancel: this is a single blocking request-reply, not an async handle.

    Returns {reason: 'reply'|'dismiss'|'timeout', reply?}: 'reply' carries the
    user's answer (a possibly-empty string); 'dismiss'/'timeout' carry no reply.
    Never raises on timeout or dismiss — those are expected outcomes (ADR-0025 §6).
    """
    message = str(arguments["message"])
    # Clamp to a sane minimum so the dialog always arms its QTimer (timeout<=0
    # would leave a never-auto-closing dialog with a fast consumer timeout — a
    # lost-reply window).
    timeout = max(float(arguments.get("timeout", 600.0)), 1.0)
    # Step 1: main-thread open — mints token + opens dialog (QTimer fires at
    # `timeout`; the dialog is the timeout SSOT, ADR-0025).
    open_result = send_gui_rpc(
        "notify.open", {"message": message, "timeout": timeout}, 30.0
    )
    token = int(open_result["token"])
    # Step 2: off-main await — blocks the IO worker until the dialog settles.
    # The consumer backstop MUST outlast the dialog's QTimer (timeout + slack) so
    # the dialog fires first and enqueues Timeout; a reply landing in the gap
    # would otherwise be lost.
    await_timeout = timeout + _NOTIFY_CONSUMER_SLACK
    await_result = send_gui_rpc(
        "notify.await",
        {"token": token, "timeout": await_timeout},
        await_timeout + _WAIT_TRANSPORT_SLACK_SECONDS,
    )
    # Forward the structured reason (reply/dismiss/timeout) verbatim; never raise.
    out: dict[str, Any] = {"reason": await_result.get("reason", "timeout")}
    reply = await_result.get("reply")
    if reply is not None:
        out["reply"] = reply
    return out


NON_GENERATED_METHODS = frozenset(
    {
        "notify.open",
        "notify.await",
    }
)


OVERRIDE_TOOLS: dict[str, dict[str, Any]] = {
    "gui_prompt_user": {
        "handler": tool_gui_prompt_user,
        "description": (
            "A BLOCKING request-reply prompt: ask the user a question and BLOCK "
            "your entire turn until they respond (or the timeout expires). Opens a "
            "non-modal prompt dialog in the GUI showing 'message'. There is NO poll "
            "or cancel — it is one blocking request-reply, not an async handle. "
            "Returns {reason, reply?} — switch on 'reason':\n"
            "  - reason='reply': user answered; 'reply' is present (a string, "
            "possibly empty) — read and act on it.\n"
            "  - reason='dismiss': user explicitly closed the prompt; NO 'reply' "
            "key — do NOT ask again immediately; respect the user's choice.\n"
            "  - reason='timeout': no one responded within 'timeout' seconds; NO "
            "'reply' key — the user is probably not watching; do NOT keep blocking, "
            "continue or poll.\n"
            "timeout (default 600s) is the dialog auto-close timer; a value <= 0 is "
            "clamped to 1.0s so the dialog always arms its auto-close. The RPC call "
            "blocks the whole turn for up to timeout+15s; use only when you genuinely "
            "need a human decision before continuing."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Question or message to display to the user",
                },
                "timeout": {
                    "type": "number",
                    "default": 600,
                    "description": (
                        "Seconds before the dialog auto-closes with reason='timeout' "
                        "(default 600; a value <= 0 is clamped to 1.0). Set shorter "
                        "for time-sensitive prompts."
                    ),
                },
            },
            "required": ["message"],
        },
    },
}


def build_override_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    bind_context(ctx)
    return OVERRIDE_TOOLS
