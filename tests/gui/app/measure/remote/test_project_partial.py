"""Atomic project updates and context deactivation through the GUI socket."""

from pathlib import Path

import pytest

from ._helpers import Fixture, call, mcp_client, open_client


@pytest.mark.uses_wall_clock
def test_project_tool_reads_gui_state_and_deactivates_on_identity_change(
    qapp, tmp_path: Path
) -> None:
    fx = Fixture(project_root=str(tmp_path), empty_project=True)
    fx.start()
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    sock = open_client(fx.service.port)
    try:
        invoke("connect", {"port": fx.service.port})
        original = invoke(
            "project", {"chip": "chip-a", "qubit": "q1", "resonator": "res"}
        )
        assert original["chip"] == "chip-a"
        created = call(sock, "context.new", {"bind_device": None, "clone_from": None})
        assert created["ok"] is True
        assert (
            call(sock, "context.active")["result"]["label"]
            == created["result"]["label"]
        )

        changed = invoke("project", {"chip": "chip-b"})
        assert changed["chip"] == "chip-b"
        assert changed["qubit"] == "q1"
        assert changed["resonator"] == "res"
        assert changed["result_dir"] != original["result_dir"]
        assert invoke("project", {}) == changed
        assert call(sock, "context.active")["result"]["label"] is None
    finally:
        bridge.disconnect()
        sock.close()
        fx.stop()


@pytest.mark.uses_wall_clock
def test_project_partial_update_and_invalid_scope_preserve_shared_state(
    qapp, tmp_path: Path
) -> None:
    fx = Fixture(project_root=str(tmp_path), empty_project=True)
    fx.start()
    sock = open_client(fx.service.port)
    try:
        assert call(sock, "project.info")["error"]["reason"] == "no_project"
        missing = call(sock, "project.apply", {"chip_name": "chip-a"})
        assert missing["ok"] is False
        assert missing["error"]["reason"] == "missing_project_fields"
        assert call(sock, "project.info")["error"]["reason"] == "no_project"

        first = call(
            sock,
            "project.apply",
            {"chip_name": "chip-a", "qub_name": "q1", "res_name": "res"},
        )
        assert first["ok"] is True
        original_scope = first["result"]["scope_id"]
        created = call(sock, "context.new", {"bind_device": None, "clone_from": None})
        assert created["ok"] is True
        label = created["result"]["label"]
        assert call(sock, "context.active")["result"]["label"] == label

        same = call(sock, "project.apply", {"chip_name": "chip-a"})
        assert same["ok"] is True
        assert same["result"]["scope_id"] == original_scope
        assert call(sock, "context.active")["result"]["label"] == label

        invalid = call(
            sock,
            "project.apply",
            {"chip_name": "chip-b", "scope_id": original_scope},
        )
        assert invalid["ok"] is False
        assert invalid["error"]["reason"] == "scope_identity_mismatch"
        assert call(sock, "project.info")["result"]["chip_name"] == "chip-a"
        assert call(sock, "context.active")["result"]["label"] == label

        changed = call(sock, "project.apply", {"chip_name": "chip-b"})
        assert changed["ok"] is True
        assert changed["result"]["chip_name"] == "chip-b"
        assert changed["result"]["qub_name"] == "q1"
        assert changed["result"]["res_name"] == "res"
        assert changed["result"]["scope_id"] != original_scope
        assert call(sock, "context.active")["result"]["label"] is None
        assert call(sock, "project.info")["result"]["chip_name"] == "chip-b"

        restored = call(
            sock, "project.apply", {"chip_name": "chip-a", "scope_id": original_scope}
        )
        assert restored["ok"] is True
        assert label in call(sock, "context.labels")["result"]["labels"]
        assert call(sock, "context.active")["result"]["label"] is None
    finally:
        sock.close()
        fx.stop()
