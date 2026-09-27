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

        assert invoke("context_use", {"label": "base"}) == {"label": "base"}
        unknown = call(sock, "context.use", {"label": "ghost"})
        assert unknown["ok"] is False
        assert "base" in unknown["error"]["message"]
        assert invoke("contexts", {})["active"] == "base"
    finally:
        bridge.disconnect()
        sock.close()
        fx.stop()
