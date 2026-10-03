"""MetaDict RPC values, receipts, summaries and persisted literal validation."""

from pathlib import Path
from typing import Any

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError

from ._helpers import Fixture, call, mcp_client, open_client


@pytest.mark.uses_wall_clock
def test_md_rpc_shares_gui_values_and_rejects_reserved_keys(
    qapp, tmp_path: Path
) -> None:
    fx = Fixture(project_root=str(tmp_path), empty_project=True)
    fx.start()
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    sock = open_client(fx.service.port)
    try:
        invoke("connect", {"port": fx.service.port})
        invoke(
            "rpc_call",
            {
                "method": "project.apply",
                "params": {"chip_name": "chip", "qub_name": "q", "res_name": "res"},
            },
        )
        invoke("rpc_call", {"method": "context.new", "params": {"label": "base"}})
        assert (
            invoke(
                "rpc_call", {"method": "context.md_get", "params": {"summaries": True}}
            )["values"]
            == {}
        )

        matrix = [[1, 2], [3, 4]]
        for key, value in (("freq", 5.0), ("matrix", matrix)):
            receipt = invoke(
                "rpc_call",
                {
                    "method": "context.md_set_attr",
                    "params": {"key": key, "value": value, "receipt": True},
                },
            )
            assert receipt["before"] is None
            assert receipt["after"] == value
        assert invoke(
            "rpc_call", {"method": "context.md_get", "params": {"summaries": True}}
        )["values"] == {"freq": 5.0, "matrix": "2 × 2 matrix"}
        assert (
            invoke(
                "rpc_call",
                {"method": "context.md_get_attr", "params": {"key": "matrix"}},
            )["value"]
            == matrix
        )
        invoke(
            "rpc_call",
            {
                "method": "context.new",
                "params": {"label": "clone", "clone_from": "current"},
            },
        )
        invoke("rpc_call", {"method": "context.use", "params": {"label": "base"}})
        assert (
            invoke(
                "rpc_call",
                {"method": "context.md_get_attr", "params": {"key": "matrix"}},
            )["value"]
            == matrix
        )
        assert (
            call(sock, "context.md_get_attr", {"key": "freq"})["result"]["value"] == 5.0
        )
        with pytest.raises(GuiRpcError, match="missing") as missing:
            invoke(
                "rpc_call",
                {"method": "context.md_get_attr", "params": {"key": "missing"}},
            )
        assert missing.value.reason == "unknown_md_key"
        assert "freq" in str(missing.value)
        with pytest.raises(GuiRpcError):
            invoke(
                "rpc_call",
                {
                    "method": "context.md_set_attr",
                    "params": {"key": "items", "value": 7},
                },
            )
        assert call(sock, "context.md_get", {})["result"]["keys"] == ["freq", "matrix"]
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
    try:
        invoke("connect", {"port": fx.service.port})
        project = invoke(
            "rpc_call",
            {
                "method": "project.apply",
                "params": {"chip_name": "chip", "qub_name": "q", "res_name": "res"},
            },
        )
        invoke("rpc_call", {"method": "context.new", "params": {"label": "base"}})
        invoke(
            "rpc_call",
            {
                "method": "context.md_set_attr",
                "params": {"key": "stable", "value": {"note": "safe"}},
            },
        )
        meta_path = Path(project["result_dir"]) / "exps/base/meta_info.json"
        before = meta_path.read_bytes()
        with pytest.raises(GuiRpcError, match="reserved"):
            invoke(
                "rpc_call",
                {
                    "method": "context.md_set_attr",
                    "params": {"key": "payload", "value": payload},
                },
            )
        assert meta_path.read_bytes() == before
        assert invoke("rpc_call", {"method": "context.snapshot"})["md"] == {
            "stable": {"note": "safe"}
        }
        invoke(
            "rpc_call",
            {
                "method": "context.new",
                "params": {"label": "copy", "clone_from": "base"},
            },
        )
        assert invoke("rpc_call", {"method": "context.snapshot"})["md"] == {
            "stable": {"note": "safe"}
        }
        invoke("rpc_call", {"method": "context.use", "params": {"label": "base"}})
        assert invoke("rpc_call", {"method": "context.snapshot"})["md"] == {
            "stable": {"note": "safe"}
        }
    finally:
        bridge.disconnect()
        fx.stop()
