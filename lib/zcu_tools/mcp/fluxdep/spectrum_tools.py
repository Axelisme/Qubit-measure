"""Handwritten spectrum decision tools compose the GUI's native RPCs."""

from typing import NotRequired, TypedDict

from zcu_tools.mcp.core.bridge import McpBridge
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.rpc import GuiRpcCall, call_gui
from zcu_tools.mcp.core.stdio_server import ToolTable

from .projection import fit_summary, spectrum_summaries, spectrum_summary, with_png


class SpectrumNameArguments(TypedDict):
    """Opaque loaded name used by snapshot and remove."""

    name: str


class SpectrumLoadArguments(TypedDict):
    """Native filepath/type and optional inheritance/axis transpose inputs."""

    filepath: str
    spec_type: str
    inherit_from: NotRequired[str | None]
    transpose_axes: NotRequired[bool]
    variable: NotRequired[str]
    device_axis: NotRequired[str]
    device: NotRequired[str]


class CheckpointExportArguments(TypedDict):
    """Optional native checkpoint destination and explicit replacement choice."""

    filepath: NotRequired[str]
    overwrite: NotRequired[bool]


class CheckpointRestoreArguments(TypedDict):
    """Required path of a sampled checkpoint to restore without Filter masks."""

    filepath: str


def build_spectrum_tools(bridge: McpBridge) -> ToolTable:
    """Build list/load/snapshot/remove with bounded summaries and J1 evidence.

    GUI validates inputs and owns all state. Reads are composed in the declared
    order; invocation/image failures propagate without retry or rollback. This
    table retains no per-call numerical data after returning its receipt.
    """

    def snapshot(arguments: SpectrumNameArguments) -> ToolReply:
        params: dict[str, object] = {"name": arguments["name"]}
        native = call_gui(bridge, GuiRpcCall("spectrum.snapshot", params, 6.0))
        summary = spectrum_summary(native)
        if not native.data["exists"]:
            return summary
        rendered = call_gui(bridge, GuiRpcCall("spectrum.render", params, 31.0))
        return with_png(summary, rendered)

    def load(arguments: SpectrumLoadArguments) -> ToolReply:
        receipt = call_gui(bridge, GuiRpcCall("spectrum.load", dict(arguments), 31.0))
        return snapshot({"name": receipt.data["name"]})

    def remove(arguments: SpectrumNameArguments) -> ToolReply:
        return call_gui(bridge, GuiRpcCall("spectrum.remove", dict(arguments), 6.0))

    def export_spectrums(arguments: CheckpointExportArguments) -> ToolReply:
        receipt = call_gui(
            bridge, GuiRpcCall("export.spectrums", dict(arguments), 31.0)
        )
        rows = spectrum_summaries(
            call_gui(bridge, GuiRpcCall("spectrum.list", {}, 6.0))
        )
        return ToolReply(
            {
                "filepath": receipt.data["filepath"],
                "spectrums": [
                    {
                        "name": row["name"],
                        "stage": row["stage"],
                        "point_count": row["point_count"],
                    }
                    for row in rows
                ],
                "note": "filter mask not saved",
            }
        )

    def restore(arguments: CheckpointRestoreArguments) -> ToolReply:
        call_gui(bridge, GuiRpcCall("spectrum.load_processed", dict(arguments), 31.0))
        return ToolReply(
            {
                "spectrums": spectrum_summaries(
                    call_gui(bridge, GuiRpcCall("spectrum.list", {}, 6.0))
                )
            }
        )

    def list_spectrums(_arguments: dict[str, object]) -> ToolReply:
        spectrums = spectrum_summaries(
            call_gui(bridge, GuiRpcCall("spectrum.list", {}, 6.0))
        )
        selection = call_gui(bridge, GuiRpcCall("selection.snapshot", {}, 6.0)).data
        fit = fit_summary(call_gui(bridge, GuiRpcCall("fit.result", {}, 6.0)))
        total = sum(row["point_count"] for row in spectrums)
        return ToolReply(
            {
                "spectrums": spectrums,
                "filter": {
                    "selected": total
                    if selection["selected"] is None
                    else sum(selection["selected"]),
                    "total": total,
                    "pending": selection["pending"],
                },
                "fit": fit,
            }
        )

    name_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    prefix = bridge.config.tool_prefix
    return {
        prefix + "spectrum_list": {
            "handler": list_spectrums,
            "description": "Read published spectrum stages, unfinished drafts, Filter counts and fit freshness.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        prefix + "spectrum_snapshot": {
            "handler": snapshot,
            "description": "Revisit a spectrum: rounded axes/alignment/point ranges plus native evidence PNG.",
            "inputSchema": name_schema,
        },
        prefix + "spectrum_load": {
            "handler": load,
            "description": "Load raw data and inspect J1 evidence: rounded spectrum summary plus PNG. inherit_from seeds alignment from another loaded spectrum.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"},
                    "spec_type": {"type": "string", "enum": ["OneTone", "TwoTone"]},
                    "inherit_from": {"type": "string"},
                    "transpose_axes": {"type": "boolean", "default": False},
                    "variable": {
                        "type": "string",
                        "description": "Required native Container variable name.",
                    },
                    "device_axis": {
                        "type": "string",
                        "description": "Required native variable device-axis name.",
                    },
                    "device": {
                        "type": "string",
                        "description": "Explicit instrument from the selected component; unnamed_device without target.",
                    },
                },
                "required": ["filepath", "spec_type"],
            },
        },
        prefix + "export_spectrums": {
            "handler": export_spectrums,
            "description": "Save a sampled checkpoint with native alignment and all sampled points, including empty completed spectra. Filter masks are not saved. Defaults to create-only; overwrite=true explicitly replaces. Return the resolved GUI path and bounded published spectrum summaries.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"},
                    "overwrite": {"type": "boolean", "default": False},
                },
            },
        },
        prefix + "spectrum_load_processed": {
            "handler": restore,
            "description": "Restore a sampled checkpoint and return published spectrum summaries. GUI rejects the entire restore if any name already exists, listing conflicts. Filter masks are not restored.",
            "inputSchema": {
                "type": "object",
                "properties": {"filepath": {"type": "string"}},
                "required": ["filepath"],
            },
        },
        prefix + "spectrum_remove": {
            "handler": remove,
            "description": "Remove one loaded spectrum and return its native removal receipt.",
            "inputSchema": name_schema,
        },
    }
