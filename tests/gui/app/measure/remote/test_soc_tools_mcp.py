"""Offline SoC state is the same through GUI socket and public MCP tools."""

from pathlib import Path

import pytest

from ._helpers import Fixture, call, mcp_client, open_client


@pytest.mark.uses_wall_clock
def test_soc_tools_observe_gui_connection_without_hardware(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    import zcu_tools.qick_remote as remote
    from zcu_tools.experiment.simulation import white_noise

    def offline_proxy(ip: str, port: int):
        assert (ip, port) == ("192.0.2.1", 8888)
        env = white_noise()
        return env.soc, env.soccfg

    monkeypatch.setattr(remote, "make_soc_proxy", offline_proxy)
    fx = Fixture(project_root=str(tmp_path), empty_project=True)
    fx.start()
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    sock = open_client(fx.service.port)
    try:
        invoke("connect", {"port": fx.service.port})
        assert invoke("status", {})["soc"] == {"connected": False, "mock": False}
        entry = invoke(
            "rpc_call",
            {
                "method": "entry.new",
                "params": {"name": "q"},
            },
        )
        invoke(
            "rpc_call",
            {
                "method": "soc.connect",
                "params": {"kind": "remote", "ip": "192.0.2.1", "port": 8888},
            },
        )
        summary = invoke("rpc_call", {"method": "soc.info"})
        assert invoke("status", {})["soc"]["connected"] is True
        assert summary["is_mock"] is True
        assert summary["address"] == "192.0.2.1" and summary["port"] == 8888
        assert call(sock, "soc.info", {})["result"]["address"] == "192.0.2.1"
        assert "Generators" in summary["description"]
        assert "Readouts" in summary["description"]
        assert "cfg" not in summary
        full = invoke(
            "rpc_call", {"method": "soc.info", "params": {"include_cfg": True}}
        )
        assert full["cfg"]["gens"] and "fs" in full["cfg"]["gens"][0]
        assert invoke(
            "rpc_call",
            {
                "method": "point.new",
                "params": {"label": "base", "clone_from": None},
            },
        ) == {
            "kind": "saved_point",
            "entry_id": entry["entry_id"],
            "point": "base",
        }
        assert invoke("status", {})["soc"] == {"connected": True, "mock": True}
    finally:
        bridge.disconnect()
        sock.close()
        fx.stop()
