"""Device RPCs through MCP, the GUI socket and GUI-owned device state."""

from pathlib import Path

import pytest
from zcu_tools.device import DeviceManager, FakeDevice

from ._helpers import Fixture, mcp_client

pytestmark = pytest.mark.uses_wall_clock


@pytest.fixture()
def device_client(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manager = DeviceManager()
    drivers = {
        name: FakeDevice(address=name, fast_mode=False)
        for name in ("bias", "ramp", "slow")
    }
    for name, driver in drivers.items():
        manager.register_device(name, driver)
    fx = Fixture(
        device_registry=manager,
        driver_factory=lambda type_name, address: drivers[address],
        project_root=str(tmp_path / "project"),
    )
    fx.start()
    # Status is outside this device seam; Fixture has no real SoC.
    monkeypatch.setattr("zcu_tools.mcp.measure.tools_lifecycle.status", lambda *_: {})
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    try:
        invoke("connect", {"port": fx.service.port})
        yield invoke
    finally:
        bridge.disconnect()
        # Fixture teardown must drain the Controller-owned runner before GC.
        fx.ctrl.close()
        fx.stop()
        manager.close_all_devices()


def test_long_fake_ramp_returns_cancellable_opaque_operation(device_client) -> None:
    invoke = device_client
    op = None
    try:
        before = invoke(
            "rpc_call",
            {
                "method": "device.snapshot",
                "params": {"name": "ramp"},
            },
        )["snapshot"]
        running = invoke(
            "rpc_call",
            {
                "method": "device.setup",
                "params": {
                    "name": "ramp",
                    "cfg": {**before["info"], "value": 0.25, "rampstep": 0.00001},
                },
            },
        )
        op = running["handle"]
        assert isinstance(op, int) and op > 0
        assert invoke("wait", {"op": op, "timeout": 0})["status"] == "running"
        cached = invoke(
            "rpc_call",
            {
                "method": "device.snapshot",
                "params": {"name": "ramp"},
            },
        )["snapshot"]
        assert cached["status"] == "setting_up"
        assert cached["info"] is not None
        assert cached["fields"] == before["fields"]
        cancelled = invoke("cancel", {"op": op})
        assert cancelled["status"] in ("cancelled", "cancelling")
        # The admitted synthetic operation must publish terminal; 30s only guards a hang.
        assert invoke("wait", {"op": op, "timeout": 30})["status"] == "cancelled"
        assert (
            invoke(
                "rpc_call",
                {
                    "method": "device.snapshot",
                    "params": {"name": "ramp"},
                },
            )["snapshot"]["status"]
            == "connected"
        )
    finally:
        if op is not None:
            invoke("cancel", {"op": op})


def test_fake_device_snapshot_reports_stored_native_unit(device_client) -> None:
    invoke = device_client
    snapshot = invoke(
        "rpc_call", {"method": "device.snapshot", "params": {"name": "bias"}}
    )["snapshot"]
    assert snapshot["unit"] == "native"
    assert snapshot["info"]["value_unit"] == "native"
