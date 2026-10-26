"""Handwritten fit decision tools over authoritative GUI RPCs."""

from typing import NotRequired, TypedDict

from zcu_tools.gui.app.fluxdep.remote.dto import TransitionWire
from zcu_tools.gui.app.fluxdep.remote.method_specs import METHOD_SPECS
from zcu_tools.gui.remote.param_spec import schema_property
from zcu_tools.mcp.core.bridge import McpBridge
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.rpc import GuiRpcCall, call_gui
from zcu_tools.mcp.core.stdio_server import ToolTable

from .projection import search_summary, with_fit_png


class DatabaseArguments(TypedDict):
    """path names a GUI-local search database to inspect, without loading a fit."""

    path: str


class FitResultArguments(TypedDict):
    """image requests J5 evidence when the native result is fresh and has quality.

    Omission defaults to True. GUI supplies fit availability and freshness; this
    query never changes inputs, starts search or infers dependencies.
    """

    image: NotRequired[bool]


class FitExportArguments(TypedDict):
    """No overrides: export to the opened entry and selected component."""


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


def _read_fit_result(bridge: McpBridge, arguments: FitResultArguments) -> ToolReply:
    """Read native observation data and request evidence only when it permits it."""
    recorded = call_gui(bridge, GuiRpcCall("fit.result", {}, 6.0))
    data = recorded.data
    if not data["has_result"]:
        return ToolReply(
            {
                "status": "no_result",
                "params": data["params"],
                "stale": data["stale"],
                "stale_reasons": data["stale_reasons"],
            }
        )
    if data["quality"] is None:
        summary = ToolReply(
            {
                "status": "finished",
                "params": data["params"],
                "inputs": {
                    **{
                        name: data[name]
                        for name in (
                            "search_db",
                            "EJb",
                            "ECb",
                            "ELb",
                            "r_f",
                            "sample_f",
                        )
                    },
                    "points": None,
                },
                "residual": None,
                "per_transition": None,
            }
        )
    else:
        summary = search_summary(recorded)
    stale = data["stale"]
    summary.data.update(
        {
            "stale": stale,
            "stale_reasons": recorded.data["stale_reasons"],
        }
    )
    if stale:
        summary.data["note"] = "rerun fluxdep_fit to refresh"
    if data["quality"] is not None and not stale and arguments.get("image", True):
        return with_fit_png(
            summary, call_gui(bridge, GuiRpcCall("fit.render", {}, 31.0))
        )
    return summary


def build_fit_tools(bridge: McpBridge) -> ToolTable:
    """Build fit/search and database tools without retaining fit state.

    Fit reads current inputs, overlays provided keys, sends complete replacement,
    starts search and waits in 30-second slices until terminal, then rereads result.
    Finished fits return recorded quality and native PNG; failed/cancelled outcomes
    are data, not invocation failures. GUI validates inputs and owns operations/I/O.
    Missing quality or invocation/image errors propagate without retry or rollback.
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
        recorded = call_gui(bridge, GuiRpcCall("fit.result", {}, 6.0))
        outcome = waited["outcome"]
        if outcome["status"] != "finished":
            return ToolReply(dict(outcome))
        summary = search_summary(recorded)
        rendered = call_gui(bridge, GuiRpcCall("fit.render", {}, 31.0))
        return with_fit_png(summary, rendered)

    def fit_result(arguments: FitResultArguments) -> ToolReply:
        return _read_fit_result(bridge, arguments)

    def export_params(arguments: FitExportArguments) -> ToolReply:
        receipt = call_gui(
            bridge, GuiRpcCall("fit.export_params", dict(arguments), 31.0)
        )
        return ToolReply(
            {
                name: receipt.data[name]
                for name in (
                    "savepath",
                    "params",
                    "alignment_from",
                    "component",
                    "analysis_id",
                    "flux_writeback",
                )
            }
        )

    # Reuse native declarations, changing only the agent patch/clear interface.
    properties: dict[str, object] = {
        spec.name: schema_property(spec)
        for spec in METHOD_SPECS["fit.set_params"].params
    }
    for spec in METHOD_SPECS["fit.set_params"].params:
        if spec.name in ("r_f", "sample_f"):
            properties[spec.name] = {
                **schema_property(spec),
                "type": ["number", "null"],
            }
    return {
        bridge.config.tool_prefix + "fit": {
            "handler": fit,
            "description": (
                "Fit current Filter points: read GUI inputs, overlay provided fields, "
                "replace inputs, search and wait until terminal. Omitted fields retain "
                "current GUI values; null clears r_f/sample_f. No local business "
                "validation, retry or rollback. search_db is the search DB, not data_root. "
                "EJb/ECb/ELb are finite [lower, upper] GHz bounds; inspect database_info "
                "for available ranges and M levels. transitions maps category names "
                "to [i,j] pairs with indices 0..M-1. For delta=E_j-E_i (GHz), "
                "categories/formulas are: transitions=|delta|; blue side=|delta+r_f|; "
                "red side=|r_f-delta|; mirror=|sample_f-delta|; "
                "mirror blue=|sample_f-r_f-delta|; mirror red=|delta+sample_f-r_f|; "
                "transitions{n}=|delta/n|; mirror{n}=|sample_f-delta/n|, "
                "integer n=2..M-1 without leading zeroes. r_f is the GHz sideband "
                "reference frequency; sample_f is the GHz sampling/alias frequency "
                "used by mirror categories. Nonempty sidebands need r_f; nonempty "
                "mirrors need sample_f, and mirror blue/red need both."
            ),
            "inputSchema": {"type": "object", "properties": properties},
        },
        bridge.config.tool_prefix + "fit_result": {
            "handler": fit_result,
            "description": "Read the recorded fit and native freshness. Fresh quality-bearing results attach J5 evidence by default; stale results never attach an image. image=false suppresses evidence without changing the fit.",
            "inputSchema": {
                "type": "object",
                "properties": {"image": {"type": "boolean", "default": True}},
            },
        },
        bridge.config.tool_prefix + "fit_export_params": {
            "handler": export_params,
            "description": (
                "Write predictors/<component>.json and declared setup fields in "
                "the opened entry, then append an analysis ledger record. "
                "Return the GUI's successful path, full-precision EJ/EC/EL GHz "
                "and actual calibration source. Stale fits are refused by GUI "
                "with fit_stale. Native readiness, control and I/O failures "
                "propagate without retry, fallback or rollback."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
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
