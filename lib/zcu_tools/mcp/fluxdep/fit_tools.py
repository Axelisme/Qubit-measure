"""Handwritten fit decision tools over authoritative GUI RPCs."""

from typing import NotRequired, TypedDict

from zcu_tools.gui.app.fluxdep.remote.dto import TransitionWire
from zcu_tools.gui.app.fluxdep.remote.method_specs import METHOD_SPECS
from zcu_tools.gui.remote.param_spec import schema_property
from zcu_tools.mcp.core.bridge import McpBridge
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.rpc import GuiRpcCall, call_gui
from zcu_tools.mcp.core.stdio_server import ToolTable


class DatabaseArguments(TypedDict):
    """path names a GUI-local search database to inspect, without loading a fit."""

    path: str


class FitArguments(TypedDict):
    """Optional overrides of current GUI inputs; omitted keys retain GUI values.

    search_db is the GUI-local DB file. EJb/ECb/ELb are GHz bounds.
    transitions maps native categories to index pairs. r_f/sample_f are optional
    GHz frequencies; explicit None clears them rather than retaining their value.
    GUI owns all business validation.
    """

    search_db: NotRequired[str]
    EJb: NotRequired[list[float]]
    ECb: NotRequired[list[float]]
    ELb: NotRequired[list[float]]
    transitions: NotRequired[TransitionWire]
    r_f: NotRequired[float | None]
    sample_f: NotRequired[float | None]


def build_fit_tools(bridge: McpBridge) -> ToolTable:
    """Build fit/search and database tools without retaining fit state.

    Fit reads current inputs, overlays provided keys, sends complete replacement,
    starts search and waits in 30-second slices until terminal, then rereads result.
    Failed/cancelled outcomes are data, not invocation failures. GUI validates
    inputs and owns operations/I/O. Errors propagate without retry or rollback.
    """

    def database_info(arguments: DatabaseArguments) -> ToolReply:
        return call_gui(bridge, GuiRpcCall("fit.database_info", dict(arguments), 31.0))

    def fit(arguments: FitArguments) -> ToolReply:
        current = call_gui(bridge, GuiRpcCall("fit.result", {}, 6.0)).data
        replacement = {
            spec.name: current[spec.name]
            for spec in METHOD_SPECS["fit.set_params"].params
        }
        replacement.update(arguments)
        call_gui(bridge, GuiRpcCall("fit.set_params", replacement, 6.0))
        started = call_gui(bridge, GuiRpcCall("fit.search", {}, 6.0)).data
        while True:
            waited = call_gui(
                bridge,
                GuiRpcCall(
                    "operation.await", {"token": started["token"], "timeout": 30}, 35.0
                ),
            ).data
            if waited["reason"] == "completed":
                break
        call_gui(bridge, GuiRpcCall("fit.result", {}, 6.0))
        return ToolReply(dict(waited["outcome"]))

    # Reuse native declarations, changing only the agent patch/clear interface.
    properties: dict[str, object] = {
        spec.name: schema_property(spec)
        for spec in METHOD_SPECS["fit.set_params"].params
    }
    for name in ("r_f", "sample_f"):
        properties[name] = {"type": ["number", "null"]}
    return {
        bridge.config.tool_prefix + "fit": {
            "handler": fit,
            "description": "Fit current Filter points: read current GUI inputs, overlay provided fields, replace inputs, search and wait until terminal. Omitted parameters retain current GUI values; explicit null clears r_f/sample_f. No retry or rollback.",
            "inputSchema": {"type": "object", "properties": properties},
        },
        bridge.config.tool_prefix + "database_info": {
            "handler": database_info,
            "description": "Inspect a search database's parameter/flux ranges, entry count and energy levels without running search.",
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
