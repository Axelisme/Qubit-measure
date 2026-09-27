"""Device tools through MCP, the GUI socket and GUI-owned device state."""

from pathlib import Path

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError

from ._helpers import Fixture, call, mcp_client, open_client

pytestmark = pytest.mark.uses_wall_clock


def test_fake_device_connect_set_disconnect_reconnect_and_forget(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx = Fixture()
    fx.start()
    # Status is outside this device seam; Fixture has no real SoC.
    monkeypatch.setattr("zcu_tools.mcp.measure.tools_lifecycle.status", lambda *_: {})
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    sock = open_client(fx.service.port)
    try:
        invoke("connect", {"port": fx.service.port})
        assert invoke("devices", {}) == []
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
        reconnected = invoke("device_connect", {"name": "bias"})
        assert reconnected["connected"] is True
        assert invoke("device_disconnect", {"name": "bias", "forget": True}) == {
            "name": "bias",
            "connected": False,
            "forgotten": True,
        }
        assert invoke("devices", {}) == []
    finally:
        bridge.disconnect()
        sock.close()
        # Fixture teardown must drain the Controller-owned runner before GC.
        assert vars(fx.ctrl)["_background_svc"].quiesce()
        fx.stop()
