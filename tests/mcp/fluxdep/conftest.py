"""One fresh connection/tool table per public contract test."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from zcu_tools.mcp.core.bridge import McpBridge, MCPBridgeConfig
from zcu_tools.mcp.fluxdep.assembly import build_fluxdep_server

from tests.mcp.fluxdep._support import Client, RecordingTransport


@pytest.fixture
def client(tmp_path: Path) -> Iterator[Client]:
    config = MCPBridgeConfig(
        app_name="fluxdep",
        app_slug="fluxdep",
        tool_prefix="fluxdep_",
        default_port=8766,
        mcp_version=2,
        wire_version=3,
        server_display_name="Fluxdep test",
        server_instructions="",
        pid_file=tmp_path / "gui.pid",
        log_file=tmp_path / "gui.log",
        run_script_name="run_fluxdep_gui.py",
    )
    bridge = McpBridge(config)
    transport = RecordingTransport()
    bridge.set_transport(transport)
    server = build_fluxdep_server(config, tmp_path, bridge=bridge)
    try:
        yield Client(server, transport)
    finally:
        bridge.disconnect()
