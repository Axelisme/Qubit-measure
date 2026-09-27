"""Device tools through MCP, the GUI socket and GUI-owned device state."""

import re
import threading
from pathlib import Path

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError

from ._helpers import Fixture, call, mcp_client, open_client

pytestmark = pytest.mark.uses_wall_clock


@pytest.fixture()
def device_client(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fx = Fixture()
    fx.start()
    # Status is outside this device seam; Fixture has no real SoC.
    monkeypatch.setattr("zcu_tools.mcp.measure.tools_lifecycle.status", lambda *_: {})
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    sock = open_client(fx.service.port)
    try:
        invoke("connect", {"port": fx.service.port})
        yield fx, invoke, sock
    finally:
        bridge.disconnect()
        sock.close()
        # Fixture teardown must drain the Controller-owned runner before GC.
        assert vars(fx.ctrl)["_background_svc"].quiesce()
        fx.stop()


def test_fake_device_connect_set_disconnect_reconnect_and_forget(device_client) -> None:
    _, invoke, sock = device_client
    assert invoke("devices", {}) == []
    try:
        connected = invoke(
            "device_connect",
            {"name": "bias", "type": "FakeDevice", "address": "none"},
        )
        assert connected["name"] == "bias"
        assert connected["type"] == "FakeDevice"
        assert connected["connected"] is True
        assert connected["address"] == "none"
        assert (
            call(sock, "device.snapshot", {"name": "bias"})["result"]["snapshot"][
                "status"
            ]
            == "connected"
        )
        fields = {field["name"]: field for field in connected["fields"]}
        assert fields["value"]["current"] == 0.0
        assert fields["address"]["settable"] is False
        assert fields["output"]["choices"] == ["on", "off"]
        assert invoke("devices", {}) == [
            {"name": "bias", "type": "FakeDevice", "connected": True}
        ]
        with pytest.raises((ValueError, GuiRpcError), match="address"):
            invoke("device_set", {"name": "bias", "values": {"address": "evil"}})
        with pytest.raises((ValueError, GuiRpcError), match="output"):
            invoke("device_set", {"name": "bias", "values": {"output": "unknown"}})
        assert invoke("devices", {"name": "bias"})["fields"] == connected["fields"]
        applied = invoke(
            "device_set", {"name": "bias", "values": {"value": 0.02, "output": "on"}}
        )
        assert {field["name"]: field["current"] for field in applied["fields"]}[
            "value"
        ] == pytest.approx(0.02)
        assert (
            call(sock, "device.snapshot", {"name": "bias"})["result"]["snapshot"][
                "info"
            ]["output"]
            == "on"
        )
        assert invoke("device_disconnect", {"name": "bias"}) == {
            "name": "bias",
            "connected": False,
            "forgotten": False,
        }
        assert invoke("devices", {}) == [
            {"name": "bias", "type": "FakeDevice", "connected": False}
        ]
        assert invoke("devices", {"name": "bias"})["fields"] == []
        reconnected = invoke("device_connect", {"name": "bias"})
        assert reconnected["connected"] is True
        assert invoke("device_disconnect", {"name": "bias", "forget": True}) == {
            "name": "bias",
            "connected": False,
            "forgotten": True,
        }
        assert invoke("devices", {}) == []
    finally:
        for device in invoke("devices", {}):
            if device["connected"]:
                invoke("device_disconnect", {"name": device["name"], "forget": True})
            else:
                call(sock, "device.forget", {"name": device["name"]})


def test_failed_connect_never_reports_success_or_registers_device(
    device_client,
) -> None:
    _, invoke, _ = device_client
    with pytest.raises(GuiRpcError, match="failed") as raised:
        invoke(
            "device_connect",
            {"name": "broken", "type": "UnknownDriver", "address": "none"},
        )
    assert raised.value.reason == "operation_failed"
    assert invoke("devices", {}) == []


def test_long_fake_ramp_returns_cancellable_opaque_operation(device_client) -> None:
    _, invoke, sock = device_client
    invoke("device_connect", {"name": "ramp", "type": "FakeDevice", "address": "none"})
    op = None
    try:
        running = invoke(
            "device_set",
            {"name": "ramp", "values": {"value": 0.25, "rampstep": 0.00001}},
        )
        assert running["status"] == "running"
        op = running["op"]
        assert isinstance(op, int) and op > 0
        assert invoke("wait", {"op": op, "timeout": 0})["status"] == "running"
        cancelled = invoke("cancel", {"op": op})
        assert cancelled["status"] in ("cancelled", "cancelling")
        assert invoke("wait", {"op": op, "timeout": 3})["status"] == "cancelled"
        assert (
            call(sock, "device.snapshot", {"name": "ramp"})["result"]["snapshot"][
                "status"
            ]
            == "connected"
        )
    finally:
        if op is not None:
            invoke("cancel", {"op": op})
        invoke("device_disconnect", {"name": "ramp", "forget": True})


def test_connect_timeout_returns_recoverable_operation(
    device_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, invoke, sock = device_client
    entered = threading.Event()
    release = threading.Event()

    def held_fake_open(_device, _resource_manager) -> None:
        entered.set()
        if not release.wait(4):
            raise TimeoutError("fake driver was not released")

    monkeypatch.setattr(
        "zcu_tools.device.fake.FakeDevice._open_session", held_fake_open
    )
    # Shorten only the test's deadline; the public tool still uses the same wait path.
    monkeypatch.setattr(
        "zcu_tools.mcp.measure.tools_device._TERMINAL_WAIT_SECONDS", 0.05
    )
    try:
        with pytest.raises(GuiRpcError) as raised:
            invoke(
                "device_connect",
                {"name": "slow", "type": "FakeDevice", "address": "none"},
            )
        assert entered.is_set()
        assert raised.value.reason == "device_pending"
        assert raised.value.code == "timeout"
        match = re.search(r"wait\(op=(\d+)\)", str(raised.value))
        assert match is not None
        op = int(match.group(1))
        assert op > 0
        assert (
            call(sock, "device.snapshot", {"name": "slow"})["result"]["snapshot"][
                "status"
            ]
            == "connecting"
        )
    finally:
        release.set()
    assert invoke("wait", {"op": op, "timeout": 3})["status"] == "finished"
    assert invoke("devices", {"name": "slow"})["connected"] is True
    assert (
        invoke("device_disconnect", {"name": "slow", "forget": True})["forgotten"]
        is True
    )
