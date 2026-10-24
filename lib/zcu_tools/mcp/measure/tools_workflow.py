"""Handwritten workflow tools using the measure session's live wire policy."""

import base64
from functools import partial

from zcu_tools.mcp.core.images import validated_png
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.mcp.measure.tool_context import MeasureToolContext

_STRING = {"type": "string"}
_FORM = {"type": "string", "enum": ["plan", "tunables"]}
_PATH = {"type": "array", "items": {"type": ["string", "integer"]}}
_BATCH = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"path": _PATH, "value": {}},
        "required": ["path", "value"],
        "additionalProperties": False,
    },
}


def _operation(ctx: MeasureToolContext, reply: dict[str, object]) -> dict[str, object]:
    result = dict(reply)
    if "operation_id" in result:
        operation_id = result.pop("operation_id")
        result["handle"] = (
            None if operation_id is None else ctx.gui.expose_operation(operation_id)
        )
    return result


def _call(
    ctx: MeasureToolContext, method: str, arguments: dict[str, object]
) -> dict[str, object] | ToolReply:
    ctx = ctx.bound()
    reply = ctx.send_gui_rpc(method, arguments)
    if method == "workflow.figure":
        png = reply.pop("png_b64")
        if not isinstance(png, str):
            raise ValueError("Invalid PNG image encoding")
        image = validated_png(base64.b64decode(png, validate=True))
        return ToolReply(reply, (image,))
    result = _operation(ctx, reply)
    if isinstance(reply.get("tab"), dict):
        result["tab"] = _operation(ctx, reply["tab"])
    if isinstance(reply.get("tabs"), list):
        result["tabs"] = [_operation(ctx, tab) for tab in reply["tabs"]]
    return result


def cancel_workflow(
    ctx: MeasureToolContext, arguments: dict[str, object]
) -> dict[str, object] | ToolReply:
    """Route the shared cancel tool through workflow wire and handle projection."""
    return _call(ctx, "workflow.cancel", arguments)


def _tool(
    ctx: MeasureToolContext,
    method: str,
    description: str,
    properties: dict[str, object],
    required: tuple[str, ...],
) -> dict[str, object]:
    return {
        "handler": partial(_call, ctx, method),
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": list(required),
        },
    }


def build_workflow_tools(ctx: MeasureToolContext) -> ToolTable:
    """Bind fixed commands; GUI catalog owns timeout, holder and operation policy.

    No workflow catalog generates tools. Handlers forward once, without filling
    defaults or maintaining run/draft/permission state. Replies preserve native
    facts, map operation IDs through the existing session and validate PNG bytes.
    Owner errors propagate unchanged, including reason/data; no retry/rollback.
    """
    return {
        "workflow_open": _tool(
            ctx,
            "workflow.open",
            "Open independent drafts for an explicit workflow name; returns tab_id/name/tab snapshot. No GUI mode or selection prerequisite; a mounted GUI selects the new tab and switches to Workflow mode.",
            {"name": _STRING},
            ("name",),
        ),
        "workflow_close": _tool(
            ctx,
            "workflow.close",
            "Close an idle workflow tab. Busy returns closed=false; does not Stop or disconnect hardware.",
            {"tab_id": _STRING},
            ("tab_id",),
        ),
        "workflow_status": _tool(
            ctx,
            "workflow.status",
            "Read native status by run_id; omit/null for catalog names and all opened tabs. No rendering.",
            {"run_id": {"type": ["string", "null"]}},
            (),
        ),
        "workflow_start": _tool(
            ctx,
            "workflow.start",
            "Submit workflow under the shared RUN lease. run_id is required and must be 32 lowercase hexadecimal characters (e.g. uuid4().hex); local GUI Start generates it. Invalid IDs are rejected before operation admission or hardware access. Omit/null plan/tunables to use shared GUI drafts. Explicit batches, including [], use fresh captured defaults. Returns pending handle, not acceptance; wait(op=handle) observes terminal delivery.",
            {
                "tab_id": _STRING,
                "run_id": _STRING,
                "plan": {"anyOf": [_BATCH, {"type": "null"}]},
                "tunables": {"anyOf": [_BATCH, {"type": "null"}]},
            },
            ("tab_id", "run_id"),
        ),
        "workflow_pause": _tool(
            ctx,
            "workflow.pause",
            "Request cooperative Pause by native run_id, retaining the operation and RUN lease.",
            {"run_id": _STRING},
            ("run_id",),
        ),
        "workflow_resume": _tool(
            ctx,
            "workflow.resume",
            "Resume the same native run, operation and lease in the worker.",
            {"run_id": _STRING},
            ("run_id",),
        ),
        "workflow_stop": _tool(
            ctx,
            "workflow.stop",
            "Request Stop; lease release awaits terminal delivery. Does not retry.",
            {"run_id": _STRING},
            ("run_id",),
        ),
        "workflow_form": _tool(
            ctx,
            "workflow.form",
            "Read authored input, ordinary values/errors. Optional batch previews current draft only; never commits.",
            {
                "tab_id": _STRING,
                "form": _FORM,
                "batch": {"anyOf": [_BATCH, {"type": "null"}]},
            },
            ("tab_id", "form"),
        ),
        "workflow_describe_form": _tool(
            ctx,
            "workflow.describe_form",
            "Observe the same candidate plus one layer of metadata at typed path; [] is root, not a whole schema.",
            {
                "tab_id": _STRING,
                "form": _FORM,
                "batch": {"anyOf": [_BATCH, {"type": "null"}]},
                "path": _PATH,
            },
            ("tab_id", "form", "path"),
        ),
        "workflow_edit_form": _tool(
            ctx,
            "workflow.edit_form",
            "Edit the shared owner draft and return input/values/errors. Owner enforces busy and Form rules.",
            {"tab_id": _STRING, "form": _FORM, "batch": _BATCH},
            ("tab_id", "form", "batch"),
        ),
        "workflow_tunables": _tool(
            ctx,
            "workflow.tunables",
            "Apply edits to the run's shared tunables draft; [] submits current draft, not defaults. Returns native revision/values.",
            {"run_id": _STRING, "batch": _BATCH},
            ("run_id", "batch"),
        ),
        "workflow_records": _tool(
            ctx,
            "workflow.records",
            "Read native commits strictly after since (default 0), including after terminal delivery.",
            {"run_id": _STRING, "since": {"type": "integer", "minimum": 0}},
            ("run_id",),
        ),
        "workflow_figures": _tool(
            ctx,
            "workflow.figures",
            "List retained native figure names without rendering.",
            {"run_id": _STRING},
            ("run_id",),
        ),
        "workflow_figure": _tool(
            ctx,
            "workflow.figure",
            "Read an exact retained figure as validated PNG image content, not base64 text.",
            {"run_id": _STRING, "name": _STRING},
            ("run_id", "name"),
        ),
        "workflow_answer": _tool(
            ctx,
            "workflow.answer",
            "Answer the exact pending_question.request_id from workflow_status using its answer_schema. Returns the original pending handle; execution stays on the same worker operation/RUN lease until terminal delivery. Invalid answers leave the question paused for retry.",
            {"run_id": _STRING, "request_id": _STRING, "answer": {}},
            ("run_id", "request_id", "answer"),
        ),
    }
