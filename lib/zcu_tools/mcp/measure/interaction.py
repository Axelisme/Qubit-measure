"""GUI interaction delivery shared by tools and recipe continuations."""

from __future__ import annotations

import base64
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from zcu_tools.mcp.core.reply import PngImage, ToolReply
from zcu_tools.mcp.measure.images import validated_png
from zcu_tools.mcp.measure.session import GuiRpcError

if TYPE_CHECKING:
    from zcu_tools.mcp.measure.analysis_execution import AnalysisExecution
    from zcu_tools.mcp.measure.tool_context import MeasureToolContext


def handoff_interaction(
    ctx: MeasureToolContext,
    execution: AnalysisExecution,
    *,
    before_send: Callable[[], None] | None = None,
) -> None:
    """Read an initial handoff without hiding its accepted analysis receipt."""
    snapshot = execution.snapshot()
    if snapshot.status != "interactive":
        return
    try:
        interact(
            ctx,
            {"tab_id": snapshot.tab},
            expected_op=snapshot.op,
            before_send=before_send,
        )
    except (GuiRpcError, ValueError, OSError) as exc:
        execution.observe_interaction(
            ToolReply({"figure": None, "delivery_error": str(exc)}, is_error=True)
        )


def interact(
    ctx: MeasureToolContext,
    params: dict[str, Any],
    *,
    expected_op: int | None = None,
    before_send: Callable[[], None] | None = None,
) -> ToolReply:
    """Deliver one committed interaction on the caller's fixed GUI binding.

    Recipes supply their original analysis handle and phase admission. A reply
    for another operation cannot replace that execution's handoff. Done joins
    the existing completion owner; PNG failure preserves the accepted receipt.
    """
    done = params.get("payload", {}).get("command") == "done"
    if done:
        params = {**params, "include_figure": False}
    reply = dict(ctx.gui.send_gui_rpc("tab.interact", params, before_send=before_send))
    if expected_op is not None and reply["handle"] != expected_op:
        raise GuiRpcError(
            "Interactive analysis has been replaced", reason="result_superseded"
        )
    execution = ctx.session.executions.for_op(reply["handle"])
    if done:
        if execution is None:
            execution = ctx.session.executions.start(
                ctx.gui,
                params["tab_id"],
                "primary",
                {
                    "handle": reply["handle"],
                    "params": None,
                    "invalidated_on_success": None,
                },
                interaction=reply,
            )
        execution.observe_interaction(ToolReply(reply), done=True)
        return execution.wait(2.0)
    figure = reply["figure"]
    images: tuple[PngImage, ...] = ()
    delivery_error = False
    try:
        if figure is not None:
            image = validated_png(base64.b64decode(figure["png_b64"], validate=True))
            path = ctx.session.write_png(image.data)
            reply["figure"] = str(path)
            images = (image,)
    except (ValueError, OSError) as exc:
        if execution is None:
            raise
        reply["figure"] = None
        reply["delivery_error"] = str(exc)
        delivery_error = True
    if execution is not None:
        reply["execution"] = execution.snapshot().execution
        execution.observe_interaction(ToolReply(reply, images, is_error=delivery_error))
    return ToolReply(reply, images, is_error=delivery_error)
