"""Offline SoC state is the same through GUI socket and public MCP tools."""

from pathlib import Path

import pytest

from ._helpers import Fixture, call, mcp_client, open_client


@pytest.mark.uses_wall_clock
def test_soc_tools_observe_gui_mock_connection_without_hardware(
    qapp, tmp_path: Path
) -> None:
    fx = Fixture(project_root=str(tmp_path), empty_project=True)
    fx.start()
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    sock = open_client(fx.service.port)
    try:
        invoke("connect", {"port": fx.service.port})
        assert invoke("soc_info", {}) == {
            "connected": False,
            "address": None,
            "port": None,
            "description": None,
            "is_mock": False,
        }
        invoke("project", {"chip": "chip", "qubit": "q", "resonator": "res"})
        connected = call(sock, "soc.connect", {"kind": "mock"})
        assert connected["ok"] is True, connected
        summary = invoke("soc_info", {})
        assert summary["connected"] is True
        assert summary["is_mock"] is True
        assert summary["address"] is None and summary["port"] is None
        assert "Generators" in summary["description"]
        assert "Readouts" in summary["description"]
        assert "cfg" not in summary
        full = invoke("soc_info", {"include_cfg": True})
        assert full["cfg"]["gens"] and "fs" in full["cfg"]["gens"][0]
        assert invoke("context_create", {"label": "base"}) == {"label": "base"}
        assert invoke("status", {})["soc"] == {"connected": True, "mock": True}
    finally:
        bridge.disconnect()
        sock.close()
        fx.stop()
