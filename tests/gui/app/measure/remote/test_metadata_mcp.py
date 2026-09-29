"""MetaDict tools preserve GUI state, bounded summaries and confirmed prefix."""

from pathlib import Path
from typing import Any

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError

from ._helpers import Fixture, call, mcp_client, open_client


@pytest.mark.uses_wall_clock
def test_md_tools_share_gui_values_and_stop_after_a_failed_key(
    qapp, tmp_path: Path
) -> None:
    fx = Fixture(project_root=str(tmp_path), empty_project=True)
    fx.start()
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    sock = open_client(fx.service.port)
    try:
        invoke("connect", {"port": fx.service.port})
        invoke("project", {"chip": "chip", "qubit": "q", "resonator": "res"})
        invoke("context_create", {"label": "base"})
        assert invoke("md_get", {}) == {"values": {}}

        matrix = [[1, 2], [3, 4]]
        assert invoke("md_set", {"values": {"freq": 5.0, "matrix": matrix}}) == {
            "freq": {"before": None, "after": 5.0},
            "matrix": {"before": None, "after": matrix},
        }
        assert invoke("md_get", {}) == {
            "values": {"freq": 5.0, "matrix": "2 × 2 matrix"}
        }
        assert invoke("md_get", {"keys": ["matrix"]}) == {"values": {"matrix": matrix}}
        assert invoke("context_create", {"label": "clone"}) == {"label": "clone"}
        assert invoke("context_use", {"label": "base"}) == {"label": "base"}
        assert invoke("md_get", {"keys": ["matrix"]}) == {"values": {"matrix": matrix}}
        assert (
            call(sock, "context.md_get_attr", {"key": "freq"})["result"]["value"] == 5.0
        )
        with pytest.raises(GuiRpcError, match="missing"):
            invoke("md_get", {"keys": ["missing"]})
        missing = call(sock, "context.md_get_attr", {"key": "missing"})
        assert missing["ok"] is False
        assert missing["error"]["reason"] == "unknown_md_key"
        assert "freq" in missing["error"]["message"]

        with pytest.raises(GuiRpcError, match="items.*freq.*before.*after"):
            invoke("md_set", {"values": {"freq": 6.0, "items": 7, "later": 8}})
        assert (
            call(sock, "context.md_get_attr", {"key": "freq"})["result"]["value"] == 6.0
        )
        assert call(sock, "context.md_get", {})["result"]["keys"] == ["freq", "matrix"]
    finally:
        bridge.disconnect()
        sock.close()
        fx.stop()


@pytest.mark.uses_wall_clock
def test_md_set_reports_confirmed_prefix_when_second_socket_write_fails(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    fx = Fixture(project_root=str(tmp_path), empty_project=True)
    fx.start()
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    sock = open_client(fx.service.port)
    try:
        invoke("connect", {"port": fx.service.port})
        invoke("project", {"chip": "chip", "qubit": "q", "resonator": "res"})
        invoke("context_create", {"label": "base"})
        original_send = bridge.send_rpc_raw

        def drop_second(
            method: str, params: dict[str, Any], timeout_seconds: float
        ) -> dict[str, Any]:
            if method == "context.md_set_attr" and params.get("key") == "second":
                bridge.disconnect()
                raise ConnectionError("GUI socket closed during second write")
            return original_send(method, params, timeout_seconds)

        monkeypatch.setattr(bridge, "send_rpc_raw", drop_second)
        with pytest.raises(GuiRpcError) as failure:
            invoke("md_set", {"values": {"first": 1, "second": 2, "later": 3}})
        message = str(failure.value)
        assert "second" in message
        assert "first" in message and "before" in message and "after" in message
        assert "may also have applied" in message
        assert call(sock, "context.md_get", {})["result"]["keys"] == ["first"]
        assert (
            call(sock, "context.md_get_attr", {"key": "first"})["result"]["value"] == 1
        )
    finally:
        bridge.disconnect()
        sock.close()
        fx.stop()


@pytest.mark.uses_wall_clock
@pytest.mark.parametrize(
    "payload",
    [
        {"nested": [{"__complex__": "not-a-tag"}]},
        {"__complex__": [1, 2]},
        {"__metadict_string__": "literal"},
    ],
)
def test_md_set_rejects_reserved_literal_tags_before_changing_context(
    qapp, tmp_path: Path, payload: dict[str, Any]
) -> None:
    fx = Fixture(project_root=str(tmp_path), empty_project=True)
    fx.start()
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    sock = open_client(fx.service.port)
    try:
        invoke("connect", {"port": fx.service.port})
        invoke("project", {"chip": "chip", "qubit": "q", "resonator": "res"})
        invoke("context_create", {"label": "base"})
        invoke("md_set", {"values": {"stable": {"note": "safe"}}})
        meta_path = (
            Path(invoke("project", {})["result_dir"]) / "exps/base/meta_info.json"
        )
        assert meta_path.is_file()

        with pytest.raises(GuiRpcError) as failure:
            invoke("md_set", {"values": {"first": 1, "payload": payload, "later": 3}})
        message = str(failure.value)
        assert "payload" in message
        assert "first" in message and "before" in message and "after" in message
        assert "reserved" in message
        assert set(call(sock, "context.md_get", {})["result"]["keys"]) == {
            "stable",
            "first",
        }
        assert invoke("md_get", {"keys": ["stable", "first"]}) == {
            "values": {"stable": {"note": "safe"}, "first": 1}
        }
        assert "payload" not in meta_path.read_text(encoding="utf-8")
        assert invoke("context_create", {"label": "copy", "clone_from": "base"}) == {
            "label": "copy"
        }
        assert invoke("context_use", {"label": "base"}) == {"label": "base"}
        assert invoke("md_get", {"keys": ["stable", "first"]}) == {
            "values": {"stable": {"note": "safe"}, "first": 1}
        }
    finally:
        bridge.disconnect()
        sock.close()
        fx.stop()
