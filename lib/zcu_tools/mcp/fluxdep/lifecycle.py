"""Fluxdep lifecycle initializes identity, then summarizes current GUI facts."""

from pathlib import Path
from typing import NotRequired, TypedDict

from zcu_tools.mcp.core.bridge import McpBridge, MCPBridgeConfig, resolve_connect_port
from zcu_tools.mcp.core.control import ControlConnection, probe_control_wire
from zcu_tools.mcp.core.lifecycle import build_lifecycle_tools
from zcu_tools.mcp.core.rpc import GuiRpcCall, call_gui
from zcu_tools.mcp.core.stdio_server import ToolTable

from .projection import fit_summary, spectrum_summaries


class ProjectIdentity(TypedDict):
    """Only chip/qubit names; the GUI derives its output paths."""

    chip_name: str
    qub_name: str


class ConnectionArguments(TypedDict):
    """Optional lifecycle settings and project identity for attach/launch."""

    port: NotRequired[int]
    token: NotRequired[str]
    auto_connect: NotRequired[bool]
    label: NotRequired[str]
    mode: NotRequired[str]
    project: NotRequired[ProjectIdentity]


def build_control_lifecycle_tools(
    config: MCPBridgeConfig, bridge: McpBridge, repo_root: Path
) -> tuple[ToolTable, frozenset[str]]:
    """Attach fixed identity and current analysis summary to connect/launch.

    label/mode retain the selected values on reconnect. Wire compatibility is
    checked before identity initialization; handshake failure disconnects. GUI
    lock/other holder still permits connection. An optional project supplies only
    identity; GUI computes paths. Read-only connect without project never mutates
    business state. auto_connect=false launches only, returning its native note.
    After a successful handshake, business/read failures propagate without
    rolling back the connection or project. Revisions are not returned to agents.
    """
    tools, names = build_lifecycle_tools(config, bridge, repo_root, "fluxdep-gui")
    identity = ControlConnection()
    for suffix in ("connect", "launch"):
        name = config.tool_prefix + suffix
        native_handler = tools[name]["handler"]

        def connect(
            arguments: ConnectionArguments,
            *,
            handler=native_handler,
            is_launch=suffix == "launch",
        ) -> str | dict[str, object]:
            nonlocal identity
            selected = ControlConnection.parse(
                arguments.get("label", identity.label),
                arguments.get("mode", identity.mode),
            )
            port = (
                arguments.get("port", config.default_port)
                if is_launch
                else resolve_connect_port(config, arguments.get("port"))
            )
            note = handler({**arguments, "port": port})
            if is_launch and not arguments.get("auto_connect", True):
                identity = selected
                return note
            try:
                probe_control_wire(bridge)
                control = selected.initialize(bridge)
            except Exception:  # Failed identity handshakes must discard the socket.
                bridge.disconnect()
                raise
            identity = selected
            project_args = arguments.get("project")
            if project_args is None:
                project = call_gui(bridge, GuiRpcCall("project.info", {}, 6.0)).data
            else:
                project = call_gui(
                    bridge, GuiRpcCall("project.setup", dict(project_args), 6.0)
                ).data["project"]
            spectrums = spectrum_summaries(
                call_gui(bridge, GuiRpcCall("spectrum.list", {}, 6.0))
            )
            fit = fit_summary(call_gui(bridge, GuiRpcCall("fit.result", {}, 6.0)))
            return {
                "connected": True,
                "port": port,
                "launched": is_launch,
                "project": project,
                "spectrums": spectrums,
                "fit": fit,
                "control": control,
            }

        tools[name]["handler"] = connect
        tools[name]["inputSchema"]["properties"].update(
            {
                "label": {"type": "string", "minLength": 1, "default": "agent"},
                "mode": {
                    "type": "string",
                    "enum": ["owner", "readonly"],
                    "default": "owner",
                },
                "project": {
                    "type": "object",
                    "properties": {
                        "chip_name": {"type": "string"},
                        "qub_name": {"type": "string"},
                    },
                    "required": ["chip_name", "qub_name"],
                    "additionalProperties": False,
                },
            }
        )
    return tools, names
