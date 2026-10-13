"""J6 Filter decisions compose the global GUI context and native evidence."""

from typing import NotRequired, TypedDict

from zcu_tools.gui.app.fluxdep.remote.command_specs import (
    SET_MIN_DISTANCE_PARAMS,
    STROKE_PARAMS,
)
from zcu_tools.gui.app.fluxdep.remote.dto import SelectionContextReply
from zcu_tools.gui.remote.param_spec import schema_property
from zcu_tools.mcp.core.bridge import McpBridge
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.rpc import GuiRpcCall, call_gui
from zcu_tools.mcp.core.stdio_server import ToolTable

from .projection import with_png


class FilterArguments(TypedDict):
    """Global Filter intent; params carry native stroke/distance inputs."""

    command: str
    params: NotRequired[dict[str, object]]
    image: NotRequired[bool]


def build_filter_tools(bridge: McpBridge) -> ToolTable:
    """Build stateless Filter: open/reuse, command, summarize and native render.

    GUI owns global context identity, validation, downsampling and per-spectrum
    counts. Finish publishes and closes once; evidence failure does not roll
    back accepted work. No identity, mask or numerical state survives this call.
    """

    declarations = {
        spec.name: spec for spec in (*STROKE_PARAMS, *SET_MIN_DISTANCE_PARAMS)
    }
    params_schema = {name: schema_property(spec) for name, spec in declarations.items()}

    def filter_points(arguments: FilterArguments) -> ToolReply:
        command = arguments["command"]
        opened = call_gui(bridge, GuiRpcCall("selection.interactive.open", {}, 31.0))
        before: SelectionContextReply = opened.data["context"]
        native = call_gui(
            bridge,
            GuiRpcCall(
                "selection.interactive.command",
                {
                    "context_id": before["context_id"],
                    "command": command,
                    "params": arguments.get("params", {}),
                },
                31.0,
            ),
        ).data
        context: SelectionContextReply = native["context"]
        state = context["state"]
        selected: dict[str, object] = {
            "count": state["selected_count"],
            "total": len(state["selected"]),
        }
        summary: dict[str, object] = {
            "stage": "filtered" if command == "finish" else "filter",
            "pending": not context["closed"],
            "min_distance": state["min_distance"],
            "selected": selected,
            "per_spectrum": state["per_spectrum"],
        }
        if before["state"]["min_distance"] != state["min_distance"]:
            summary["changes"] = {
                "min_distance": [before["state"]["min_distance"], state["min_distance"]]
            }
        if command != "finish":
            changes = native["effect"]["changes"]
            selected["changes"] = [changes["points_added"], changes["points_removed"]]
            summary["can_undo"] = context["can_undo"]
        if "invalidated" in native:
            summary["invalidated"] = native["invalidated"]
        reply = ToolReply(summary)
        if command == "finish":
            return with_png(
                reply, call_gui(bridge, GuiRpcCall("selection.render", {}, 31.0))
            )
        if arguments.get("image", False):
            return with_png(
                reply,
                call_gui(
                    bridge,
                    GuiRpcCall("interactive.render", {"target": "selection"}, 31.0),
                ),
            )
        return reply

    return {
        bridge.config.tool_prefix + "filter": {
            "handler": filter_points,
            "description": (
                "Filter the joint cloud (J6). A new draft starts from the published "
                "masks and min_distance (every point without a published mask). "
                "stroke uses flux/GHz vertices and explicit "
                f"select/erase mode; width: {declarations['width'].description} "
                "set_min_distance sets min_distance: "
                f"{declarations['min_distance'].description} "
                "undo restores the previous edit. finish publishes and closes the "
                "global draft and returns published native evidence. image=true returns "
                "draft evidence for unfinished edits. Counts come from GUI; image "
                "errors do not roll back publication."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["stroke", "set_min_distance", "undo", "finish"],
                    },
                    "params": {
                        "type": "object",
                        "properties": params_schema,
                    },
                    "image": {"type": "boolean", "default": False},
                },
                "required": ["command"],
            },
        }
    }
