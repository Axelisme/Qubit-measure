"""J3 sampling decisions compose native named drafts without retained state."""

from dataclasses import replace
from typing import NotRequired, TypedDict

from zcu_tools.gui.app.fluxdep.remote.command_specs import (
    SET_SETTINGS_PARAMS,
    SET_THRESHOLD_PARAMS,
    STROKE_PARAMS,
)
from zcu_tools.gui.app.fluxdep.remote.dto import InteractiveContextReply
from zcu_tools.gui.remote.param_spec import schema_property
from zcu_tools.mcp.core.bridge import McpBridge
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.rpc import GuiRpcCall, call_gui
from zcu_tools.mcp.core.stdio_server import ToolTable

from .projection import point_summary, with_png


class SampleArguments(TypedDict):
    """Sampling intent; GUI owns spectrum type, detector domains and admission.

    name is the literal loaded spectrum name. command edits or finishes its
    sampling draft. params contains native command arguments. image requests
    native draft evidence; finish always returns published spectrum evidence.
    """

    name: str
    command: str
    params: NotRequired[dict[str, object]]
    image: NotRequired[bool]


def build_sample_tools(bridge: McpBridge) -> ToolTable:
    """Build the stateless J3 tool over GUI captures and native commands.

    Open/reuse the named sampling context, letting GUI choose its native type,
    then command. finish reads published points and renders native evidence.
    GUI owns prerequisites, context lifetime and positional point differences.
    Invocation errors propagate without retry or rollback. No active selection,
    draft, detector settings or spectrum data is retained between calls.
    """

    one_threshold = SET_THRESHOLD_PARAMS[0]
    settings = {spec.name: spec for spec in SET_SETTINGS_PARAMS}
    stroke = {spec.name: spec for spec in STROKE_PARAMS}
    two_threshold = settings["threshold"]
    # The union envelope is only a schema; the GUI validates the native domain.
    threshold = replace(
        one_threshold,
        minimum=(
            min(one_threshold.minimum, two_threshold.minimum)
            if one_threshold.minimum is not None and two_threshold.minimum is not None
            else None
        ),
        maximum=(
            max(one_threshold.maximum, two_threshold.maximum)
            if one_threshold.maximum is not None and two_threshold.maximum is not None
            else None
        ),
        description=(
            f"OneTone: {one_threshold.description} TwoTone: {two_threshold.description}"
        ),
    )
    params_schema = {
        spec.name: schema_property(spec)
        for spec in (*SET_SETTINGS_PARAMS, *STROKE_PARAMS)
    }
    params_schema["threshold"] = schema_property(threshold)

    def sample(arguments: SampleArguments) -> ToolReply:
        name, command = arguments["name"], arguments["command"]
        target: dict[str, object] = {"name": name}
        opened = call_gui(
            bridge,
            GuiRpcCall("spectrum.interactive.open", {**target, "kind": "sample"}, 31.0),
        ).data
        before: InteractiveContextReply = opened["context"]
        native = call_gui(
            bridge,
            GuiRpcCall(
                "spectrum.interactive.command",
                {
                    **target,
                    "context_id": before["context_id"],
                    "command": command,
                    "params": dict(arguments.get("params", {})),
                },
                31.0,
            ),
        ).data
        after: InteractiveContextReply = native["context"]
        summary: dict[str, object] = {
            "name": name,
            "stage": "sampled" if command == "finish" else "sample",
            "pending": not after["closed"],
        }
        if "discarded" in opened:
            summary["discarded"] = opened["discarded"]
        if command == "finish":
            published_points = call_gui(
                bridge, GuiRpcCall("spectrum.snapshot", target, 6.0)
            ).data["points"]
            summary["points"] = point_summary(published_points)
            if "invalidated" in native:
                summary["invalidated"] = native["invalidated"]
            return with_png(
                ToolReply(summary),
                call_gui(bridge, GuiRpcCall("spectrum.render", target, 31.0)),
            )
        changes = native["effect"]["changes"]
        if before["kind"] == "onetone" and after["kind"] == "onetone":
            settings: dict[str, object] = {"threshold": after["state"]["threshold"]}
            changed: dict[str, object] = {}
            if before["state"]["threshold"] != after["state"]["threshold"]:
                changed["threshold"] = [
                    before["state"]["threshold"],
                    after["state"]["threshold"],
                ]
            summary["settings"] = settings
            if changed:
                summary["changes"] = changed
            summary["points"] = {
                "count": len(after["state"]["peak_indices"]),
                "changes": [changes["peaks_added"], changes["peaks_removed"]],
            }
        elif before["kind"] == "twotone" and after["kind"] == "twotone":
            settings = {
                key: after["state"][key]
                for key in ("threshold", "sigma", "smooth_method")
            }
            changed = {
                key: [before["state"][key], after["state"][key]]
                for key in settings
                if before["state"][key] != after["state"][key]
            }
            summary["settings"] = settings
            if changed:
                summary["changes"] = changed
            summary["points"] = {
                "count": after["state"]["point_count"],
                "changes": [changes["points_added"], changes["points_removed"]],
            }
            summary["mask"] = {
                "pixels": after["state"]["masked_count"],
                "changes": [changes["mask_added"], changes["mask_removed"]],
            }
            summary["can_undo"] = after["can_undo"]
        else:
            raise RuntimeError(
                "Sampling command returned a different native context kind"
            )
        reply = ToolReply(summary)
        if arguments.get("image", False):
            return with_png(
                reply,
                call_gui(
                    bridge,
                    GuiRpcCall(
                        "interactive.render", {**target, "target": "spectrum"}, 31.0
                    ),
                ),
            )
        return reply

    return {
        bridge.config.tool_prefix + "sample": {
            "handler": sample,
            "description": (
                "Sample a named aligned spectrum (J3). GUI selects its native picker "
                "from the published spectrum type. OneTone: set_threshold; "
                f"threshold: {one_threshold.description} finish. "
                "TwoTone: stroke(vertices=device/GHz pairs, mode=select|erase); "
                f"width: {stroke['width'].description} "
                "set_settings accepts threshold, sigma, smooth_method=wavelet|gaussian; "
                f"threshold: {settings['threshold'].description} "
                f"sigma: {settings['sigma'].description} "
                "set_settings requires at least one value. undo restores the previous "
                "edit. finish always returns published point ranges and native evidence; "
                "image=true requests draft evidence for other commands. Initial and "
                "reopened TwoTone sampling start empty. Replies summarize settings "
                "before/after, detected points and mask pixels separately. GUI owns "
                "prerequisites and parameter validation; accepted edits are not retried "
                "or rolled back."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "command": {
                        "type": "string",
                        "enum": [
                            "set_threshold",
                            "stroke",
                            "set_settings",
                            "undo",
                            "finish",
                        ],
                    },
                    "params": {
                        "type": "object",
                        "properties": params_schema,
                    },
                    "image": {"type": "boolean", "default": False},
                },
                "required": ["name", "command"],
            },
        }
    }
