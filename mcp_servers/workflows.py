#!/usr/bin/env python
"""Compose the workflow MCP stdio attachment; GUI launch is an explicit tool."""

from pathlib import Path

from zcu_tools.gui.app.workflows.remote.methods import WIRE_VERSION
from zcu_tools.mcp.core.bridge import MCPBridgeConfig
from zcu_tools.mcp.workflows.server import run_workflow_mcp


def main() -> None:
    """Delegate workflow stdio tools with the repository GUI launch configuration.

    Explicit lifecycle tools launch or connect. The GUI launcher injects lab
    registrations and its source loader; EOF only disconnects the attachment.
    """
    repo_root = Path(__file__).resolve().parents[1]
    config = MCPBridgeConfig(
        tool_prefix="workflows_",
        server_display_name="workflows-gui",
        server_instructions=(
            "Operate GUI-owned coherence_fluxdep, mist_fluxdep or overnight "
            "through the same declared RPC tools. Start's optional name "
            "selects workflow; omission selects coherence_fluxdep. "
            "Read status and use an explicit run_id for commands and data. "
            "Start receipts are pending, not native acceptance. Pause/Stop "
            "receipts are requests, not producer completion. Read figures before "
            "requesting an exact name. No hidden reads, retries or hardware "
            "authorization. A timeout or disconnect does not stop a run."
        ),
        app_name="workflows",
        app_slug="workflows",
        default_port=8768,
        mcp_version=2,
        wire_version=WIRE_VERSION,
        pid_file=repo_root / "logs/workflows/gui.pid",
        log_file=repo_root / "logs/workflows/gui.log",
        run_script_name="run_workflows_gui.py",
    )
    run_workflow_mcp(config, repo_root)


if __name__ == "__main__":
    main()
