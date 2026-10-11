"""Named native points and clone errors through the live GUI socket and MCP."""

from pathlib import Path

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.resources.entry import ResultEntry, source_path

from ._helpers import Fixture, call, mcp_client, open_client


def selection(invoke):
    """Read selection metadata without claiming a full-context observation."""
    target = invoke(
        "rpc_call", {"method": "context.snapshot", "params": {"path": "target"}}
    )["value"]
    return {
        "active": target["point"] if target["kind"] == "saved_point" else None,
        "labels": invoke("rpc_call", {"method": "context.labels"})["labels"],
    }


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
        entry = invoke("rpc_call", {"method": "entry.new", "params": {"name": "q"}})
        assert selection(invoke) == {"active": None, "labels": []}
        assert invoke(
            "rpc_call",
            {"method": "point.new", "params": {"label": "base", "clone_from": None}},
        ) == {"kind": "saved_point", "entry_id": entry["entry_id"], "point": "base"}
        context = fx.ctrl.context_control
        context.set_parameter(context.read_context().target, "general.ext.freq", 5.0)
        assert invoke(
            "rpc_call",
            {"method": "point.new", "params": {"label": "copy", "clone_from": "base"}},
        ) == {"kind": "saved_point", "entry_id": entry["entry_id"], "point": "copy"}
        assert (
            call(sock, "context.snapshot", {"path": "point.general.ext.freq"})[
                "result"
            ]["value"]
            == 5.0
        )
        assert selection(invoke) == {"active": "copy", "labels": ["base", "copy"]}

        failed = call(sock, "point.new", {"label": "bad", "clone_from": "missing"})
        assert failed["ok"] is False
        assert "missing" in failed["error"]["message"]
        assert selection(invoke) == {"active": "copy", "labels": ["base", "copy"]}
        unsafe = call(sock, "point.new", {"label": "../escape", "clone_from": None})
        assert unsafe["ok"] is False
        assert unsafe["error"]["code"] == "invalid_params"
        assert selection(invoke) == {"active": "copy", "labels": ["base", "copy"]}
        assert not (Path(entry["result_path"]) / "escape").exists()

        assert invoke(
            "rpc_call", {"method": "point.use", "params": {"label": "base"}}
        ) == {"kind": "saved_point", "entry_id": entry["entry_id"], "point": "base"}
        unknown = call(sock, "point.use", {"label": "ghost"})
        assert unknown["ok"] is False
        assert "ghost" in unknown["error"]["message"]
        duplicate = call(sock, "point.new", {"label": "copy", "clone_from": None})
        assert duplicate["ok"] is False
        assert duplicate["error"]["code"] == "invalid_params"
        assert "exist" in duplicate["error"]["message"]
        assert selection(invoke)["active"] == "base"
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
        info = invoke("rpc_call", {"method": "entry.new", "params": {"name": "q"}})
        for label in ("source", "active"):
            invoke("rpc_call", {"method": "point.new", "params": {"label": label}})
        entry = ResultEntry.open(
            info["name"],
            result_root=Path(info["result_path"]).parent,
            database_root=Path(info["database_path"]).parent,
        )
        point_source = source_path(entry.use_point("source"))
        library_source = entry.module_library("source").source_path
        exp_dir = point_source.parent.parent
        occupied = exp_dir / "occupied"
        occupied.mkdir()
        user_file = occupied / "user.txt"
        user_file.write_text("keep me", encoding="utf-8")
        conflict = call(sock, "point.new", {"label": "occupied", "clone_from": None})
        assert conflict["ok"] is False
        assert "exist" in conflict["error"]["message"]
        assert user_file.read_text(encoding="utf-8") == "keep me"
        assert call(sock, "context.active", {})["result"]["label"] == "active"

        source, other_file = (
            (library_source, point_source)
            if source_file == "module_cfg.yaml"
            else (point_source, library_source)
        )
        assert source.is_file()
        other_content = other_file.read_bytes()
        if invalid_content is None:
            source.unlink()
        else:
            source.write_text(invalid_content, encoding="utf-8")
        with pytest.raises(GuiRpcError):
            invoke(
                "rpc_call",
                {
                    "method": "point.new",
                    "params": {"label": "failed", "clone_from": "source"},
                },
            )
        assert call(sock, "context.active", {})["result"]["label"] == "active"
        assert selection(invoke) == {"active": "active", "labels": ["active", "source"]}
        assert not (exp_dir / "failed").exists()
        with pytest.raises(GuiRpcError):
            invoke("rpc_call", {"method": "point.use", "params": {"label": "source"}})
        assert call(sock, "context.active", {})["result"]["label"] == "active"
        assert selection(invoke) == {"active": "active", "labels": ["active", "source"]}
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
def test_context_create_default_without_active_is_empty_and_explicit_null_skips_clone(
    qapp, tmp_path: Path
) -> None:
    fx = Fixture(project_root=str(tmp_path), empty_project=True)
    fx.start()
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    sock = open_client(fx.service.port)
    try:
        invoke("connect", {"port": fx.service.port})
        entry = invoke("rpc_call", {"method": "entry.new", "params": {"name": "q"}})
        assert invoke(
            "rpc_call",
            {"method": "point.new", "params": {"label": "base", "clone_from": None}},
        ) == {"kind": "saved_point", "entry_id": entry["entry_id"], "point": "base"}
        assert (
            call(sock, "context.snapshot", {})["result"]["point"]["general"]["ext"]
            == {}
        )
        context = fx.ctrl.context_control
        context.set_parameter(context.read_context().target, "general.ext.freq", 5.0)
        assert invoke(
            "rpc_call",
            {"method": "point.new", "params": {"label": "empty", "clone_from": None}},
        ) == {"kind": "saved_point", "entry_id": entry["entry_id"], "point": "empty"}
        assert (
            call(sock, "context.snapshot", {})["result"]["point"]["general"]["ext"]
            == {}
        )
        assert selection(invoke) == {"active": "empty", "labels": ["base", "empty"]}
    finally:
        bridge.disconnect()
        sock.close()
        fx.stop()
