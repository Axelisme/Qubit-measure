"""Device RPCs through MCP, the GUI socket and GUI-owned device state."""

import threading
from pathlib import Path

import pytest
from zcu_tools.device import DeviceManager, FakeDevice
from zcu_tools.mcp.measure.session import GuiRpcError

from tests.gui._qt_waits import qt_until

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


def test_manual_fake_device_is_rejected_through_mcp(device_client) -> None:
    invoke = device_client
    before = invoke("rpc_call", {"method": "device.list"})
    with pytest.raises(GuiRpcError, match="simulation.*publish"):
        invoke(
            "rpc_call",
            {
                "method": "device.connect",
                "params": {
                    "name": "manual",
                    "type_name": "FakeDevice",
                    "address": "none",
                },
            },
        )
    assert invoke("rpc_call", {"method": "device.list"}) == before
    assert invoke("rpc_call", {"method": "device.active_operations"}) == {
        "operations": []
    }


def test_fake_device_connect_set_disconnect_reconnect_and_forget(device_client) -> None:
    invoke = device_client
    disconnected = invoke(
        "rpc_call", {"method": "device.disconnect", "params": {"name": "bias"}}
    )
    assert (
        invoke("wait", {"op": disconnected["handle"], "timeout": 30})["status"]
        == "finished"
    )
    started = invoke(
        "rpc_call",
        {
            "method": "device.reconnect",
            "params": {"name": "bias"},
        },
    )
    # The admitted synthetic operation must publish terminal; 30s only guards a hang.
    assert (
        invoke("wait", {"op": started["handle"], "timeout": 30})["status"] == "finished"
    )
    try:
        connected = invoke(
            "rpc_call",
            {
                "method": "device.snapshot",
                "params": {"name": "bias"},
            },
        )["snapshot"]
        assert connected["name"] == "bias"
        assert connected["type_name"] == "FakeDevice"
        assert connected["status"] == "connected"
        assert connected["address"] == "bias"
        fields = {field["name"]: field for field in connected["fields"]}
        assert fields["value"]["current"] == 0.0
        assert fields["address"]["settable"] is False
        assert fields["output"]["choices"] == ["on", "off"]
        rejected_address = invoke(
            "rpc_call",
            {
                "method": "device.setup",
                "params": {
                    "name": "bias",
                    "cfg": {**connected["info"], "address": "evil"},
                },
            },
        )
        # The admitted synthetic operation must publish terminal; 30s only guards a hang.
        failed = invoke("wait", {"op": rejected_address["handle"], "timeout": 30})
        assert failed["status"] == "failed"
        assert "address" in str(failed)
        unchanged = invoke(
            "rpc_call", {"method": "device.snapshot", "params": {"name": "bias"}}
        )["snapshot"]
        assert unchanged["info"] == connected["info"]
        with pytest.raises(GuiRpcError, match="output"):
            invoke(
                "rpc_call",
                {
                    "method": "device.setup",
                    "params": {
                        "name": "bias",
                        "cfg": {**connected["info"], "output": "unknown"},
                    },
                },
            )
        applied = invoke(
            "rpc_call",
            {
                "method": "device.setup",
                "params": {
                    "name": "bias",
                    "cfg": {**connected["info"], "value": 0.02, "output": "on"},
                },
            },
        )
        # The admitted synthetic operation must publish terminal; 30s only guards a hang.
        assert (
            invoke("wait", {"op": applied["handle"], "timeout": 30})["status"]
            == "finished"
        )
        snapshot = invoke(
            "rpc_call",
            {
                "method": "device.snapshot",
                "params": {"name": "bias"},
            },
        )["snapshot"]
        assert snapshot["info"]["value"] == pytest.approx(0.02)
        assert snapshot["info"]["output"] == "on"
        disconnected = invoke(
            "rpc_call",
            {
                "method": "device.disconnect",
                "params": {"name": "bias"},
            },
        )
        # The admitted synthetic operation must publish terminal; 30s only guards a hang.
        assert (
            invoke("wait", {"op": disconnected["handle"], "timeout": 30})["status"]
            == "finished"
        )
        remembered = invoke(
            "rpc_call",
            {
                "method": "device.snapshot",
                "params": {"name": "bias"},
            },
        )["snapshot"]
        assert remembered["status"] == "memory_only"
        assert remembered["fields"] == []
        reconnected = invoke(
            "rpc_call",
            {
                "method": "device.reconnect",
                "params": {"name": "bias"},
            },
        )
        # The admitted synthetic operation must publish terminal; 30s only guards a hang.
        assert (
            invoke("wait", {"op": reconnected["handle"], "timeout": 30})["status"]
            == "finished"
        )
    finally:
        disconnected = invoke(
            "rpc_call",
            {
                "method": "device.disconnect",
                "params": {"name": "bias", "remember": False},
            },
        )
        # The admitted synthetic operation must publish terminal; 30s only guards a hang.
        assert (
            invoke("wait", {"op": disconnected["handle"], "timeout": 30})["status"]
            == "finished"
        )
    assert "bias" not in {
        row["name"] for row in invoke("rpc_call", {"method": "device.list"})["devices"]
    }


def test_failed_connect_never_reports_success_or_registers_device(
    device_client,
) -> None:
    invoke = device_client
    before = invoke("rpc_call", {"method": "device.list"})["devices"]
    started = invoke(
        "rpc_call",
        {
            "method": "device.connect",
            "params": {
                "name": "broken",
                "type_name": "UnknownDriver",
                "address": "none",
            },
        },
    )
    # The admitted synthetic operation must publish terminal; 30s only guards a hang.
    outcome = invoke("wait", {"op": started["handle"], "timeout": 30})
    assert outcome["status"] == "failed"
    assert outcome["error"]
    assert invoke("rpc_call", {"method": "device.list"}) == {
        "devices": sorted(
            before
            + [
                {
                    "name": "broken",
                    "type_name": "UnknownDriver",
                    "status": "memory_only",
                }
            ],
            key=lambda row: row["name"],
        )
    }
    snapshot = invoke(
        "rpc_call", {"method": "device.snapshot", "params": {"name": "broken"}}
    )["snapshot"]
    assert snapshot["status"] == "memory_only"
    assert snapshot["address"] == "none"
    assert snapshot["error"] == outcome["error"]["message"]
    assert "UnknownDriver" in snapshot["error"]
    assert snapshot["info"] is None
    assert snapshot["fields"] == []
    with pytest.raises(GuiRpcError, match="no Info model"):
        invoke(
            "rpc_call",
            {"method": "device.setup", "params": {"name": "broken", "cfg": {}}},
        )


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
        disconnected = invoke(
            "rpc_call",
            {
                "method": "device.disconnect",
                "params": {"name": "ramp", "remember": False},
            },
        )
        # The admitted synthetic operation must publish terminal; 30s only guards a hang.
        assert (
            invoke("wait", {"op": disconnected["handle"], "timeout": 30})["status"]
            == "finished"
        )


def test_connect_wait_timeout_does_not_cancel_operation(
    device_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invoke = device_client
    entered = threading.Event()
    release = threading.Event()

    original_info = FakeDevice.get_info

    def held_fake_info(device: FakeDevice):
        entered.set()
        # The test finally releases the held collaborator.
        if not release.wait(30):
            raise TimeoutError("fake driver was not released")
        return original_info(device)

    monkeypatch.setattr("zcu_tools.device.fake.FakeDevice.get_info", held_fake_info)
    disconnected = invoke(
        "rpc_call", {"method": "device.disconnect", "params": {"name": "slow"}}
    )
    assert (
        invoke("wait", {"op": disconnected["handle"], "timeout": 30})["status"]
        == "finished"
    )
    try:
        started = invoke(
            "rpc_call",
            {"method": "device.reconnect", "params": {"name": "slow"}},
        )
        op = started["handle"]
        assert isinstance(op, int) and op > 0
        # device.connect submitted the held driver; observe entry independently of timeout.
        qt_until(entered.is_set, label="device open entered")
        assert invoke("wait", {"op": op, "timeout": 0})["status"] == "running"
        assert (
            invoke(
                "rpc_call",
                {
                    "method": "device.snapshot",
                    "params": {"name": "slow"},
                },
            )["snapshot"]["status"]
            == "connecting"
        )
    finally:
        release.set()
    # The admitted synthetic operation must publish terminal; 30s only guards a hang.
    assert invoke("wait", {"op": op, "timeout": 30})["status"] == "finished"
    assert (
        invoke(
            "rpc_call",
            {
                "method": "device.snapshot",
                "params": {"name": "slow"},
            },
        )["snapshot"]["status"]
        == "connected"
    )
    disconnected = invoke(
        "rpc_call",
        {
            "method": "device.disconnect",
            "params": {"name": "slow", "remember": False},
        },
    )
    # The admitted synthetic operation must publish terminal; 30s only guards a hang.
    assert (
        invoke("wait", {"op": disconnected["handle"], "timeout": 30})["status"]
        == "finished"
    )


def test_fake_device_snapshot_reports_stored_native_unit(device_client) -> None:
    invoke = device_client
    try:
        snapshot = invoke(
            "rpc_call", {"method": "device.snapshot", "params": {"name": "bias"}}
        )["snapshot"]
        assert snapshot["unit"] == "native"
        assert snapshot["info"]["value_unit"] == "native"
    finally:
        disconnected = invoke(
            "rpc_call", {"method": "device.disconnect", "params": {"name": "bias"}}
        )
        # The admitted synthetic operation must publish terminal; 30s only guards a hang.
        assert (
            invoke("wait", {"op": disconnected["handle"], "timeout": 30})["status"]
            == "finished"
        )
