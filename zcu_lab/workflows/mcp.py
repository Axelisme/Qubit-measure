"""T1 workflow MCP attachment over the existing GUI RPC owners."""

from pathlib import Path

from zcu_tools.mcp.core.bridge import McpBridge, MCPBridgeConfig
from zcu_tools.mcp.core.stdio_server import ToolTable


def build_workflow_tools(
    config: MCPBridgeConfig, bridge: McpBridge, repo_root: Path
) -> ToolTable:
    """Assemble the nine workflow RPC tools and shared lifecycle callbacks.

    config supplies tool names and connection/launch settings; bridge must use
    that same config, otherwise raise ValueError before assembling any tools.
    repo_root anchors the configured GUI launch script. Every workflow tool sends
    exactly one declared RPC through this bridge, without prereads, retries,
    subscriptions, draft state or operation tracking. GUI invocation errors retain
    code/message/reason; transport errors propagate and timeouts do not cancel.
    workflow.figure validates its one returned PNG with the shared image owner,
    removes base64 from text and returns run_id/name plus inline image content.
    Corrupt base64/PNG raises without retry or file creation. All other native
    replies are unchanged, including failed operation outcomes.
    """
    raise NotImplementedError("Internal workflow MCP assembly pending")


def run_workflow_mcp(
    config: MCPBridgeConfig,
    repo_root: Path,
    *,
    bridge: McpBridge | None = None,
) -> None:
    """Run the workflow MCP stdio loop until EOF, then only disconnect the GUI.

    An omitted bridge creates one instance with config; an injected bridge must
    use config and may have an attached recording Transport. Assemble tools with
    build_workflow_tools, then use the shared stdio error/content projection.
    EOF never stops or kills a launched GUI. No hardware or GUI is started until
    the caller explicitly invokes the shared launch/connect lifecycle tools.
    """
    raise NotImplementedError("Internal workflow MCP stdio composition pending")
