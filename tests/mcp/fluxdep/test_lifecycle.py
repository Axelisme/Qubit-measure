"""Connection resumes the analysis from native GUI project and freshness facts."""

from pathlib import Path

import pytest
from zcu_tools.mcp.core.bridge import McpBridge, MCPBridgeConfig
from zcu_tools.mcp.fluxdep.assembly import build_fluxdep_server

from tests.mcp.fluxdep._support import Client, GuiResponse, RecordingTransport


class LifecycleBridge(McpBridge):
    """Replace socket/spawn IO while retaining real RPC correlation and waits."""

    def __init__(self, config: MCPBridgeConfig, transport: RecordingTransport) -> None:
        super().__init__(config, transport=transport)
        self.lifecycle_calls: list[tuple[str, int, str | None]] = []

    def connect(self, port: int, token: str | None = None) -> str:
        """Record an offline connection instead of opening a socket."""
        self.lifecycle_calls.append(("connect", port, token))
        return "connected"

    def launch(self, repo_root: Path, port: int, token: str | None = None, *, auto_connect: bool = True, extra_args: list[str] | None = None) -> str:
        """Record an offline launch instead of spawning a GUI."""
        self.lifecycle_calls.append(("launch", port, token))
        return "launched"


@pytest.mark.parametrize("launch", [False, True])
@pytest.mark.parametrize("setup", [False, True])
def test_connection_returns_project_stages_fit_and_control_in_one_reply(
    client: Client, tmp_path: Path, launch: bool, setup: bool,
) -> None:
    config = client.server.bridge.config
    transport = RecordingTransport()
    bridge = LifecycleBridge(config, transport)
    server = build_fluxdep_server(config, tmp_path, bridge=bridge)
    project = {"chip_name": "chip", "qub_name": "Q1", "result_dir": "/result/chip/Q1", "data_root": "/Database/chip/Q1"}
    control = {"holder": "agent", "locked": False, "online": True, "me": True}
    transport.replies.extend([
        GuiResponse(result={"wire_version": config.wire_version, "gui_version": 7}),
        GuiResponse(result=control),
        GuiResponse(result={"project": project} if setup else project),
        GuiResponse(result={"spectrums": [{"name": "a", "spec_type": "TwoTone", "aligned": True, "points_completed": True, "point_count": 3, "pending": None}]}),
        GuiResponse(result={"has_result": True, "stale": False, "stale_reasons": [], "params": {"EJ": 3}, "quality": {"residuals": [1, 2, 3]}}),
    ])
    arguments: dict[str, object] = {"port": 9001, "token": "secret", "label": "agent", "mode": "owner"}
    if setup:
        arguments["project"] = {"chip_name": "chip", "qub_name": "Q1"}
    try:
        result = server.tools["fluxdep_launch" if launch else "fluxdep_connect"]["handler"](arguments)
        assert result == {
            "connected": True, "port": 9001, "launched": launch,
            "project": project,
            "spectrums": [{"name": "a", "type": "TwoTone", "stage": "sampled", "point_count": 3, "pending": None}],
            "fit": {"has_result": True, "stale": False}, "control": control,
        }
        assert [(r["method"], r["params"]) for r in transport.requests] == [
            ("wire.version", {}), ("control.connect", {"label": "agent", "mode": "owner"}),
            ("project.setup", {"chip_name": "chip", "qub_name": "Q1"}) if setup else ("project.info", {}),
            ("spectrum.list", {}), ("fit.result", {}),
        ]
        assert bridge.lifecycle_calls == [("launch" if launch else "connect", 9001, "secret")]
    finally:
        bridge.disconnect()


def test_incompatible_wire_disconnects_before_identity_or_business_reads(client: Client, tmp_path: Path) -> None:
    config = client.server.bridge.config
    transport = RecordingTransport()
    bridge = LifecycleBridge(config, transport)
    server = build_fluxdep_server(config, tmp_path, bridge=bridge)
    transport.replies.append(GuiResponse(result={"wire_version": config.wire_version + 1, "gui_version": 7}))
    with pytest.raises(RuntimeError, match="incompatible GUI wire version"):
        server.tools["fluxdep_connect"]["handler"]({"port": 9001})
    assert [r["method"] for r in transport.requests] == ["wire.version"]
    assert not transport.is_open
