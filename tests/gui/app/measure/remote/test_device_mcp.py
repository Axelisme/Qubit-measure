"""Device RPCs through MCP, the GUI socket and GUI-owned device state."""

import threading
from pathlib import Path

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError

from ._helpers import Fixture, mcp_client

pytestmark = pytest.mark.uses_wall_clock


@pytest.fixture()
def device_client(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fx = Fixture()
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
        assert vars(fx.ctrl)["_background_svc"].quiesce()
        fx.stop()


def test_fake_device_connect_set_disconnect_reconnect_and_forget(device_client) -> None:
    invoke = device_client
    assert invoke("rpc_call", {"method": "device.list"}) == {"devices": []}
    started = invoke(
        "rpc_call",
        {
            "method": "device.connect",
            "params": {
                "name": "bias",
                "type_name": "FakeDevice",
                "address": "none",
            },
        },
    )
    assert (
        invoke("wait", {"op": started["handle"], "timeout": 3})["status"] == "finished"
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
        assert connected["address"] == "none"
        fields = {field["name"]: field for field in connected["fields"]}
        assert fields["value"]["current"] == 0.0
        assert fields["address"]["settable"] is False
        assert fields["output"]["choices"] == ["on", "off"]
        for updates, field in [
            ({"address": "evil"}, "address"),
            ({"output": "unknown"}, "output"),
        ]:
            with pytest.raises(GuiRpcError, match=field):
                invoke(
                    "rpc_call",
                    {
                        "method": "device.setup",
                        "params": {
                            "name": "bias",
                            "updates": updates,
                        },
                    },
                )
        applied = invoke(
            "rpc_call",
            {
                "method": "device.setup",
                "params": {
                    "name": "bias",
                    "updates": {"value": 0.02, "output": "on"},
                },
            },
        )
        assert (
            invoke("wait", {"op": applied["handle"], "timeout": 3})["status"]
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
        assert (
            invoke("wait", {"op": disconnected["handle"], "timeout": 3})["status"]
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
        assert (
            invoke("wait", {"op": reconnected["handle"], "timeout": 3})["status"]
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
        assert (
            invoke("wait", {"op": disconnected["handle"], "timeout": 3})["status"]
            == "finished"
        )
    assert invoke("rpc_call", {"method": "device.list"}) == {"devices": []}


def test_failed_connect_never_reports_success_or_registers_device(
    device_client,
) -> None:
    invoke = device_client
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
    outcome = invoke("wait", {"op": started["handle"], "timeout": 3})
    assert outcome["status"] == "failed"
    assert outcome["error"]
    assert invoke("rpc_call", {"method": "device.list"}) == {"devices": []}


def test_long_fake_ramp_returns_cancellable_opaque_operation(device_client) -> None:
    invoke = device_client
    connected = invoke(
        "rpc_call",
        {
            "method": "device.connect",
            "params": {
                "name": "ramp",
                "type_name": "FakeDevice",
                "address": "none",
            },
        },
    )
    assert (
        invoke("wait", {"op": connected["handle"], "timeout": 3})["status"]
        == "finished"
    )
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
                    "updates": {"value": 0.25, "rampstep": 0.00001},
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
        assert invoke("wait", {"op": op, "timeout": 3})["status"] == "cancelled"
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
        assert (
            invoke("wait", {"op": disconnected["handle"], "timeout": 3})["status"]
            == "finished"
        )


def test_connect_wait_timeout_does_not_cancel_operation(
    device_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invoke = device_client
    entered = threading.Event()
    release = threading.Event()

    def held_fake_open(_device, _resource_manager) -> None:
        entered.set()
        if not release.wait(4):
            raise TimeoutError("fake driver was not released")

    monkeypatch.setattr(
        "zcu_tools.device.fake.FakeDevice._open_session", held_fake_open
    )
    try:
        started = invoke(
            "rpc_call",
            {
                "method": "device.connect",
                "params": {
                    "name": "slow",
                    "type_name": "FakeDevice",
                    "address": "none",
                },
            },
        )
        op = started["handle"]
        assert isinstance(op, int) and op > 0
        assert invoke("wait", {"op": op, "timeout": 0.05})["status"] == "running"
        assert entered.is_set()
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
    assert invoke("wait", {"op": op, "timeout": 3})["status"] == "finished"
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
    assert (
        invoke("wait", {"op": disconnected["handle"], "timeout": 3})["status"]
        == "finished"
    )
