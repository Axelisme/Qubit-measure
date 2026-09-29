"""Named contexts and clone errors through the live GUI socket and MCP."""

from pathlib import Path

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError

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
        assert unknown["error"]["reason"] == "unknown_context"
        duplicate = call(
            sock,
            "context.new",
            {"label": "copy", "bind_device": None, "clone_from": None},
        )
        assert duplicate["ok"] is False
        assert duplicate["error"]["code"] == "invalid_params"
        assert duplicate["error"]["reason"] == "context_exists"
        assert "different label" in duplicate["error"]["message"]
        assert invoke("contexts", {})["active"] == "base"
    finally:
        bridge.disconnect()
        sock.close()
        fx.stop()


@pytest.mark.uses_wall_clock
@pytest.mark.parametrize(
    ("source_file", "invalid_content"),
    [
        ("module_cfg.yaml", "modules: [\n"),
        ("module_cfg.yaml", None),
        ("meta_info.json", "{broken"),
        ("meta_info.json", "null"),
    ],
)
def test_invalid_context_source_and_occupied_directory_preserve_selection(
    qapp, tmp_path: Path, source_file: str, invalid_content: str | None
) -> None:
    fx = Fixture(project_root=str(tmp_path), empty_project=True)
    fx.start()
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    sock = open_client(fx.service.port)
    try:
        invoke("connect", {"port": fx.service.port})
        invoke("project", {"chip": "chip", "qubit": "q", "resonator": "res"})
        invoke("context_create", {"label": "source", "clone_from": None})
        invoke("context_create", {"label": "active", "clone_from": None})
        exp_dir = Path(invoke("project", {})["result_dir"]) / "exps"
        occupied = exp_dir / "occupied"
        occupied.mkdir()
        user_file = occupied / "user.txt"
        user_file.write_text("keep me", encoding="utf-8")
        conflict = call(
            sock,
            "context.new",
            {"label": "occupied", "bind_device": None, "clone_from": None},
        )
        assert conflict["ok"] is False
        assert conflict["error"]["reason"] == "context_exists"
        assert user_file.read_text(encoding="utf-8") == "keep me"
        assert call(sock, "context.active", {})["result"]["label"] == "active"

        source = exp_dir / "source" / source_file
        assert source.is_file()
        other_file = (
            exp_dir
            / "source"
            / (
                "meta_info.json"
                if source_file == "module_cfg.yaml"
                else "module_cfg.yaml"
            )
        )
        other_content = other_file.read_bytes()
        if invalid_content is None:
            source.unlink()
        else:
            source.write_text(invalid_content, encoding="utf-8")

        with pytest.raises(GuiRpcError):
            invoke("context_create", {"label": "failed", "clone_from": "source"})
        assert call(sock, "context.active", {})["result"]["label"] == "active"
        assert invoke("contexts", {}) == {
            "active": "active",
            "labels": ["active", "source"],
        }
        assert not (exp_dir / "failed").exists()

        with pytest.raises(GuiRpcError):
            invoke("context_use", {"label": "source"})
        assert call(sock, "context.active", {})["result"]["label"] == "active"
        assert invoke("contexts", {}) == {
            "active": "active",
            "labels": ["active", "source"],
        }
        if invalid_content is None:
            assert not source.exists()
        else:
            assert source.read_text(encoding="utf-8") == invalid_content
        assert other_file.read_bytes() == other_content
        assert user_file.read_text(encoding="utf-8") == "keep me"
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
