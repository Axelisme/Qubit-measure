"""J3 sampling decisions compose native named drafts without retained state."""

from typing import NotRequired, TypedDict

from zcu_tools.gui.app.fluxdep.remote.dto import InteractiveContextReply
from zcu_tools.mcp.core.bridge import McpBridge
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.rpc import GuiRpcCall, call_gui
from zcu_tools.mcp.core.stdio_server import ToolTable


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

    Read published type, open/reuse its named sampling context, then command.
    GUI owns prerequisites, context lifetime and positional point differences.
    Invocation errors propagate without retry or rollback. No active selection,
    draft, detector settings or spectrum data is retained between calls.
    """

    def sample(arguments: SampleArguments) -> ToolReply:
        name, command = arguments["name"], arguments["command"]
        target: dict[str, object] = {"name": name}
        published = call_gui(bridge, GuiRpcCall("spectrum.snapshot", target, 6.0)).data
        kind = {"OneTone": "onetone", "TwoTone": "twotone"}[published["spec_type"]]
        opened = call_gui(
            bridge,
            GuiRpcCall("spectrum.interactive.open", {**target, "kind": kind}, 31.0),
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
        changes = native["effect"]["changes"]
        summary: dict[str, object] = {
            "name": name,
            "stage": "sample",
            "pending": not after["closed"],
        }
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
        if "discarded" in opened:
            summary["discarded"] = opened["discarded"]
        return ToolReply(summary)

    return {
        bridge.config.tool_prefix + "sample": {
            "handler": sample,
            "description": "Sample a named aligned spectrum (J3), using its published GUI type. OneTone: set_threshold(threshold in [0,5]), finish. TwoTone: stroke(vertices=device/GHz pairs, width=normalized radius [0,0.1], mode=select|erase), set_settings(threshold in [1,20], sigma in [0,5], smooth_method=wavelet|gaussian; gaussian sigma >=0.001), undo, finish. Settings are dimensionless; set_settings requires at least one value. Initial and reopened TwoTone sampling start empty. Replies summarize settings before/after, detected points and mask pixels separately. GUI owns prerequisites and parameter validation; accepted edits are not retried or rolled back.",
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
                        "properties": {
                            "threshold": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 20,
                            },
                            "sigma": {"type": "number", "minimum": 0, "maximum": 5},
                            "smooth_method": {
                                "type": "string",
                                "enum": ["wavelet", "gaussian"],
                            },
                            "vertices": {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 2,
                                    "maxItems": 2,
                                },
                            },
                            "width": {"type": "number", "minimum": 0, "maximum": 0.1},
                            "mode": {"type": "string", "enum": ["select", "erase"]},
                        },
                    },
                    "image": {"type": "boolean", "default": False},
                },
                "required": ["name", "command"],
            },
        }
    }
