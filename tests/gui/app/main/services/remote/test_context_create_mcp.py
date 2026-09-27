"""Named contexts and clone errors through the live GUI socket and MCP."""

from pathlib import Path

import pytest

from ._helpers import Fixture, call, mcp_client, open_client


@pytest.mark.uses_wall_clock
def test_context_create_named_clone_and_invalid_source_do_not_change_active(
    qapp, tmp_path: Path
) -> None:
    fx = Fixture(project_root=str(tmp_path), empty_project=True)
    fx.start()
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    sock = open_client(fx.service.port)
    try:
        invoke("connect", {"port": fx.service.port})
        invoke("project", {"chip": "chip", "qubit": "q", "resonator": "res"})
        assert invoke("contexts", {}) == {"active": None, "labels": []}

        assert invoke("context_create", {"label": "base", "clone_from": None}) == {
            "label": "base"
        }
        assert call(sock, "context.md_set_attr", {"key": "freq", "value": 5.0})["ok"]
        assert invoke("context_create", {"label": "copy"}) == {"label": "copy"}
        assert (
            call(sock, "context.md_get_attr", {"key": "freq"})["result"]["value"] == 5.0
        )
        assert invoke("contexts", {}) == {
            "active": "copy",
            "labels": ["base", "copy"],
        }

        failed = call(
            sock,
            "context.new",
            {"label": "bad", "bind_device": None, "clone_from": "missing"},
        )
        assert failed["ok"] is False
        assert failed["error"]["code"] == "invalid_params"
        assert "base" in failed["error"]["message"]
        assert invoke("contexts", {}) == {
            "active": "copy",
            "labels": ["base", "copy"],
        }

        unsafe = call(
            sock,
            "context.new",
            {"label": "../escape", "bind_device": None, "clone_from": None},
        )
        assert unsafe["ok"] is False
        assert unsafe["error"]["code"] == "invalid_params"
        assert invoke("contexts", {}) == {
            "active": "copy",
            "labels": ["base", "copy"],
        }
        assert not (tmp_path / "result" / "chip" / "q" / "escape").exists()

        assert invoke("context_use", {"label": "base"}) == {"label": "base"}
        unknown = call(sock, "context.use", {"label": "ghost"})
        assert unknown["ok"] is False
        assert "base" in unknown["error"]["message"]
        assert invoke("contexts", {})["active"] == "base"
    finally:
        bridge.disconnect()
        sock.close()
        fx.stop()


@pytest.mark.uses_wall_clock
def test_context_create_bound_device_reads_value_and_rejects_missing_device(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    from zcu_tools.device.fake import FakeDeviceInfo
    from zcu_tools.gui.session.services.device import DeviceService
    from zcu_tools.gui.session.state import DeviceState, DeviceStatus

    fx = Fixture(project_root=str(tmp_path), empty_project=True)
    fx.start()
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    sock = open_client(fx.service.port)
    try:
        invoke("connect", {"port": fx.service.port})
        invoke("project", {"chip": "chip", "qubit": "q", "resonator": "res"})
        fx.state.put_device(
            DeviceState(
                name="flux",
                type_name="FakeDevice",
                address="none",
                status=DeviceStatus.CONNECTED,
                remember=False,
                info=FakeDeviceInfo(address="none", value=0.25),
            )
        )
        monkeypatch.setattr(
            DeviceService,
            "get_device_info",
            lambda self, name: FakeDeviceInfo(address="none", value=0.25),
        )
        bound = invoke("context_create", {"bind_device": "flux", "clone_from": None})
        assert bound["label"].endswith("_0.250")
        assert invoke("contexts", {}) == {
            "active": bound["label"],
            "labels": [bound["label"]],
        }

        missing = call(
            sock,
            "context.new",
            {"label": "bad", "bind_device": "ghost", "clone_from": None},
        )
        assert missing["ok"] is False
        assert missing["error"]["code"] == "invalid_params"
        assert missing["error"]["reason"] == "invalid_bind_device"
        monkeypatch.setattr(DeviceService, "get_device_info", lambda self, name: None)
        no_value = call(
            sock,
            "context.new",
            {"label": "no_value", "bind_device": "flux", "clone_from": None},
        )
        assert no_value["ok"] is False
        assert no_value["error"]["code"] == "precondition_failed"
        assert no_value["error"]["reason"] == "missing_device_value"
        assert invoke("contexts", {}) == {
            "active": bound["label"],
            "labels": [bound["label"]],
        }
    finally:
        bridge.disconnect()
        sock.close()
        fx.stop()


@pytest.mark.uses_wall_clock
def test_context_create_default_without_active_is_empty_and_explicit_null_skips_clone(
    qapp, tmp_path: Path
) -> None:
    fx = Fixture(project_root=str(tmp_path), empty_project=True)
    fx.start()
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    sock = open_client(fx.service.port)
    try:
        invoke("connect", {"port": fx.service.port})
        invoke("project", {"chip": "chip", "qubit": "q", "resonator": "res"})
        assert invoke("context_create", {"label": "base"}) == {"label": "base"}
        assert call(sock, "context.md_get", {})["result"] == {"keys": []}
        assert call(sock, "context.md_set_attr", {"key": "freq", "value": 5.0})["ok"]
        assert invoke("context_create", {"label": "empty", "clone_from": None}) == {
            "label": "empty"
        }
        assert call(sock, "context.md_get", {})["result"] == {"keys": []}
        assert invoke("contexts", {}) == {
            "active": "empty",
            "labels": ["base", "empty"],
        }
    finally:
        bridge.disconnect()
        sock.close()
        fx.stop()
