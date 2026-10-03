"""Offline SoC state is the same through GUI socket and public MCP tools."""

import threading
from pathlib import Path

import pytest
from zcu_tools.device.fake import FakeDevice
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.program.v2.mocksoc import MockQickSoc

from ._helpers import Fixture, call, mcp_client, open_client


@pytest.mark.uses_wall_clock
def test_soc_tools_observe_gui_connection_without_hardware(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    import zcu_tools.qick_remote as remote
    from zcu_tools.program.v2.mocksoc import make_mock_soc

    def offline_proxy(ip: str, port: int):
        assert (ip, port) == ("192.0.2.1", 8888)
        return make_mock_soc()

    monkeypatch.setattr(remote, "make_soc_proxy", offline_proxy)
    fx = Fixture(project_root=str(tmp_path), empty_project=True)
    fx.start()
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    sock = open_client(fx.service.port)
    try:
        invoke("connect", {"port": fx.service.port})
        assert invoke("status", {})["soc"] == {"connected": False, "mock": False}
        invoke(
            "rpc_call",
            {
                "method": "project.apply",
                "params": {"chip_name": "chip", "qub_name": "q", "res_name": "res"},
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
        assert summary["is_mock"] is False
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
                "method": "context.new",
                "params": {
                    "label": "base",
                    "bind_device": None,
                    "clone_from": "current",
                },
            },
        ) == {"label": "base", "has_active_context": True}
        assert invoke("status", {})["soc"] == {"connected": True, "mock": False}
    finally:
        bridge.disconnect()
        sock.close()
        fx.stop()


@pytest.fixture()
def simulation_client(qapp, tmp_path: Path, request: pytest.FixtureRequest):
    fx = Fixture(project_root=str(tmp_path), empty_project=True)
    fx.start()
    bridge, invoke = mcp_client(fx.service.port, tmp_path, request=request)
    try:
        invoke("connect", {"port": fx.service.port})
        yield fx, invoke
    finally:
        bridge.disconnect()
        assert vars(fx.ctrl)["_background_svc"].quiesce()
        fx.stop()


@pytest.mark.uses_wall_clock
def test_simulation_initialize_binds_live_flux_and_reuses_gui_environment(
    simulation_client,
):
    fx, invoke = simulation_client
    for params in ({"kind": "mock"}, {"kind": "mock", "ip": "192.0.2.1", "port": 8888}):
        with pytest.raises(GuiRpcError, match="invalid_params"):
            invoke("rpc_call", {"method": "soc.connect", "params": params})
    assert fx.state.session_env.soc is None
    assert invoke("rpc_call", {"method": "device.list"}) == {"devices": []}

    started = invoke("rpc_call", {"method": "simulation.initialize"})
    assert (
        invoke("wait", {"op": started["handle"], "timeout": 3})["status"] == "finished"
    )
    soc = fx.state.session_env.soc
    assert isinstance(soc, MockQickSoc)
    assert soc.flux_source is not None
    predictor = fx.state.session_env.predictor
    assert predictor is not None
    assert invoke("status", {})["soc"] == {"connected": True, "mock": True}
    for value in (0.001, 0.002):
        snapshot = invoke(
            "rpc_call", {"method": "device.snapshot", "params": {"name": "fake_flux"}}
        )["snapshot"]
        assert snapshot["status"] == "connected"
        applied = invoke(
            "rpc_call",
            {
                "method": "device.setup",
                "params": {"name": "fake_flux", "updates": {"value": value}},
            },
        )
        assert (
            invoke("wait", {"op": applied["handle"], "timeout": 3})["status"]
            == "finished"
        )
        assert soc.flux_source() == pytest.approx(value)

    # The GUI facet and RPC reuse the same already-bound environment.
    gui_op = fx.ctrl.setup_control.start_simulated_environment()
    sock = open_client(fx.service.port)
    try:
        outcome = call(sock, "operation.await", {"operation_id": gui_op, "timeout": 3})
        assert outcome["result"]["status"] == "finished"
    finally:
        sock.close()
    again = invoke("rpc_call", {"method": "simulation.initialize"})
    assert invoke("wait", {"op": again["handle"], "timeout": 3})["status"] == "finished"
    assert fx.state.session_env.soc is soc
    assert fx.state.session_env.predictor is predictor
    assert soc.flux_source() == pytest.approx(0.002)


@pytest.mark.uses_wall_clock
def test_simulation_initialize_reports_setup_failure(simulation_client, monkeypatch):
    fx, invoke = simulation_client

    def fail_setup(self, *args, **kwargs):
        raise OSError("fake source setup failed")

    monkeypatch.setattr(FakeDevice, "setup", fail_setup)
    started = invoke("rpc_call", {"method": "simulation.initialize"})
    outcome = invoke("wait", {"op": started["handle"], "timeout": 3})
    assert outcome["status"] == "failed"
    assert "fake source setup failed" in str(outcome)
    assert fx.state.session_env.soc is None


@pytest.mark.uses_wall_clock
def test_simulation_initialize_rejects_concurrent_entry(simulation_client, monkeypatch):
    _, invoke = simulation_client
    release = threading.Event()
    setup = FakeDevice.setup

    def blocked_setup(self, *args, **kwargs):
        assert release.wait(5), "test must release simulated device setup"
        return setup(self, *args, **kwargs)

    monkeypatch.setattr(FakeDevice, "setup", blocked_setup)
    started = invoke("rpc_call", {"method": "simulation.initialize"})
    try:
        with pytest.raises(GuiRpcError, match="precondition_failed"):
            invoke("rpc_call", {"method": "simulation.initialize"})
    finally:
        release.set()
    assert (
        invoke("wait", {"op": started["handle"], "timeout": 3})["status"] == "finished"
    )
