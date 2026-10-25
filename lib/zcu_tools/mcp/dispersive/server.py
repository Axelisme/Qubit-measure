#!/usr/bin/env python
"""MCP server bridge for the dispersive-fit-gui ``RemoteControlAdapter``.

Communicates with an MCP host (Gemini / Claude / VS Code) via stdio JSON-RPC
2.0, and forwards calls to the live dispersive GUI's ``RemoteControlAdapter``
over a single persistent TCP socket. The exposed tools are READ-ONLY: every
analysis method tool is generated 1:1 from the wire-method contract table
(``METHOD_SPECS``, all pure queries — the user drives the GUI); the agent-facing
lifecycle tools (``dispersive_launch`` / ``dispersive_connect`` /
``dispersive_disconnect``) are built by the shared read-only factory and fork
the caller-supplied Python module.

The whole server body (``send_gui_rpc``, the lifecycle tools, cleanup, the stdio
loop) lives in :func:`zcu_tools.mcp.core.readonly_server.build_readonly_server`;
this module keeps only dispersive's config + instructions + the imports the
factory needs. Events are dropped (request/reply, not event subscription).

Threading: see :mod:`zcu_tools.mcp.core.bridge`.
"""

from __future__ import annotations

import runpy
from pathlib import Path
from tempfile import gettempdir

_BOOTSTRAP = runpy.run_path(str(Path(__file__).resolve().parents[1] / "_standalone.py"))
_BOOTSTRAP["bootstrap_standalone_server"](
    __file__,
    required_modules=(
        (
            "qtpy",
            "dispersive-fit-gui MCP server requires the 'gui' extra (qtpy); "
            "'{module}' is missing. Rebuild the environment with:\n"
            "    uv sync --extra gui\n",
        ),
    ),
)

# tolerated: Method contracts need standalone lib setup and qtpy preflight first.
from zcu_tools.gui.app.dispersive.remote.method_specs import (  # noqa: E402
    METHOD_SPECS,
)

# tolerated: Wire version imports must follow lib setup and qtpy preflight.
from zcu_tools.gui.app.dispersive.remote.wire_version import (  # noqa: E402
    WIRE_VERSION as MCP_WIRE_VERSION,
)

# tolerated: Bridge imports must follow standalone lib setup and qtpy preflight.
from zcu_tools.mcp.core.bridge import MCPBridgeConfig  # noqa: E402

# tolerated: Readonly assembly needs standalone lib setup and qtpy preflight first.
from zcu_tools.mcp.core.readonly_server import (  # noqa: E402
    READONLY_MCP_VERSION,
    ReadonlyServer,
    build_readonly_server,
)

# This MCP server's own code revision — reported (not compared) in the version
# note so an agent can confirm a reconnect picked up bridge-side edits.
MCP_VERSION = READONLY_MCP_VERSION

_SERVER_INSTRUCTIONS = """\
Observe a live dispersive-fit-gui (fluxonium dispersive-shift analysis) over a TCP socket.

This bridge is READ-ONLY: the USER drives the analysis in the GUI (load the
fluxonium predictor from the opened entry, load a one-tone spectrum, preprocess it, manually
tune the coupling g and resonator frequency bare_rf until the predicted lines match
the spectrum, and export the entry-scoped dispersive file). The agent's job is to watch and
report current state — there are no load / preprocess / tune / export tools, because
the tuning and the judgement of when it matches need the user's eye on the plot.

The fit inputs come from fluxdep-gui: choose a file under result/<entry>/predictors/
(EJ/EC/EL + affine flux alignment). Export writes dispersive/<component>.json
(g, bare_rf), and only fields declared by the component kind. There is no legacy
params.json reader. project_root contains result/ and Database/.

Getting started:
  1. dispersive_launch opens a GUI subprocess for the user (auto-connects the bridge).
     Or dispersive_connect to attach to a GUI the user already started.
  2. The user does the analysis in the GUI; you observe it with the read tools.
  dispersive_disconnect detaches the bridge without stopping the GUI. There is no
  stop tool — the agent never closes the user's GUI.

Read tools (all pure queries):
  - dispersive_state_check → {has_project, has_fit_inputs, has_onetone,
    has_preprocess, has_result} — how far the user has taken the pipeline.
  - dispersive_project_info → {entry, component, predictor, result_root,
    result_path, predictors_path, dispersive_path} (nullable before opening an entry).
  - dispersive_fit_inputs_info → {has_inputs, params:{EJ,EC,EL} or null, flux_half,
    flux_bias, flux_period, bare_rf_seed} — the selected predictor handoff.
  - dispersive_preprocess_status → {has_preprocess, n_flux, n_freq, edelay}.
  - dispersive_fit_result → {has_result, g, bare_rf, res_dim} — the user's
    accepted tuning result.

A failed call always raises; the read tools are idempotent, so retrying a read is
safe.
"""


def build_server(config: MCPBridgeConfig, repo_root: Path) -> ReadonlyServer:
    """Assemble read-only queries and entry-aware GUI launch tools.

    config supplies transport identity, port and caller-owned log/PID paths.
    repo_root is the GUI module's working directory and default project_root.
    Construction does not launch or connect. Launch accepts project_root, entry
    and optional component, forwards them to the GUI CLI and propagates bridge
    process/transport failures. Invalid argument types raise ValueError.
    """
    server = build_readonly_server(
        config, METHOD_SPECS, repo_root, "dispersive-fit-gui"
    )

    def launch(arguments: dict[str, object]) -> str:
        extra_args: list[str] = []
        for name in ("project_root", "entry", "component"):
            value = arguments.get(name)
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonblank string")
            extra_args.extend(("--" + name.replace("_", "-"), value))
        if arguments.get("component") is not None and arguments.get("entry") is None:
            raise ValueError("component requires entry")
        port = arguments.get("port", config.default_port)
        if type(port) is not int:
            raise ValueError("port must be an integer")
        token = arguments.get("token")
        if token is not None and not isinstance(token, str):
            raise ValueError("token must be a string")
        auto_connect = arguments.get("auto_connect", True)
        if not isinstance(auto_connect, bool):
            raise ValueError("auto_connect must be a boolean")
        return server.bridge.launch(
            repo_root, port, token, auto_connect=auto_connect, extra_args=extra_args
        )

    tool = server.tools[config.tool_prefix + "launch"]
    tool["handler"] = launch
    tool["description"] = (
        "Launch a new dispersive GUI for an entry and optional fluxonium component. "
        "project_root contains result/ and Database/ and defaults to the repository. "
        "Omit entry to choose it in the GUI. Analysis queries remain read-only."
    )
    tool["inputSchema"]["properties"].update(
        {
            "project_root": {
                "type": "string",
                "description": "Project containing result/ and Database/",
            },
            "entry": {"type": "string", "description": "Existing Result entry name"},
            "component": {
                "type": "string",
                "description": "Fluxonium setup component; defaults to qubit alias",
            },
        }
    )
    return server


def main(*, gui_module: str) -> None:
    """Run stdio tools; launch the caller's Python module only on tool request.

    gui_module must be importable from the repository root with python -m.
    EOF disconnects without closing the GUI or controlling hardware.
    """
    config = MCPBridgeConfig(
        app_name="dispersive",
        app_slug="dispersive",
        tool_prefix="dispersive_",
        default_port=8767,
        mcp_version=MCP_VERSION,
        wire_version=MCP_WIRE_VERSION,
        server_display_name="dispersive-fit-gui-control",
        server_instructions=_SERVER_INSTRUCTIONS,
        pid_file=Path(gettempdir()) / "zcu_tools_dispersive_gui.pid",
        log_file=Path(gettempdir()) / "zcu_tools_dispersive_gui_debug.log",
        gui_module=gui_module,
    )
    server = build_server(config, Path(__file__).resolve().parents[4])
    server.main()
