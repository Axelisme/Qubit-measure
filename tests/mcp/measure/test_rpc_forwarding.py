"""Public MCP forwarding with GUI-owned concurrency enforcement."""

from pathlib import Path
from typing import Any

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError

from ._support import make_client


def test_rpc_preserves_reply_and_sends_one_request(tmp_path: Path) -> None:
    expected = {"loaded": True, "cfg_backfill": "not_applied"}
    client = make_client(tmp_path, lambda method, params: expected)
    client.context.session.ensure_connected()
    client.transport.sent.clear()
    params = {"tab_id": "tab-1", "data_path": "saved.h5"}

    assert (
        client.call("rpc_call", {"method": "tab.load_data", "params": params})
        == expected
    )
    assert client.transport.sent == [("tab.load_data", params)]


def test_rpc_stale_requires_explicit_retry_by_caller(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    client.context.session.ensure_connected()
    client.transport.sent.clear()
    client.transport.replies["tab.load_data"] = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "reason": "stale_version",
            "message": "read the current result first",
            "data": {"stale": ["tab:tab-1:result"]},
        },
    }
    params = {"tab_id": "tab-1", "data_path": "saved.h5"}
    with pytest.raises(GuiRpcError) as failure:
        client.call("rpc_call", {"method": "tab.load_data", "params": params})
    assert failure.value.reason == "stale_version"
    assert failure.value.code == "precondition_failed"
    assert "run result" in str(failure.value)
    assert client.transport.sent == [("tab.load_data", params)]


def test_internal_read_returns_full_operational_snapshot(tmp_path: Path) -> None:
    snapshot: dict[str, Any] = {
        "tabs": [
            {"tab_id": "tab-1", "result_state": {"revision": 8, "available": True}}
        ]
    }
    client = make_client(tmp_path, lambda method, params: snapshot)
    client.context.session.ensure_connected()
    client.transport.sent.clear()

    assert (
        client.context.session.read_internal("tab.snapshot", {"tab_id": "tab-1"})
        == snapshot
    )
    assert client.transport.sent == [("tab.snapshot", {"tab_id": "tab-1"})]


def test_waveform_save_and_preview_are_explicit_independent_calls(
    tmp_path: Path,
) -> None:
    client = make_client(tmp_path)
    client.context.session.ensure_connected()
    client.transport.sent.clear()
    client.transport.replies["arb_waveform.set"] = {
        "ok": True,
        "result": {"success": True, "status": "created"},
    }
    client.transport.replies["arb_waveform.preview"] = {
        "ok": False,
        "error": {"code": "internal_error", "message": "PNG export unavailable"},
    }
    params = {
        "name": "pulse",
        "recipe": {
            "segments": [{"duration": 1.0, "formula": "0"}],
            "normalize": "none",
        },
    }

    assert client.call(
        "rpc_call", {"method": "arb_waveform.set", "params": params}
    ) == {"success": True, "status": "created"}
    assert client.transport.sent == [("arb_waveform.set", params)]
    with pytest.raises(GuiRpcError, match="PNG export unavailable"):
        client.call(
            "rpc_call", {"method": "arb_waveform.preview", "params": {"name": "pulse"}}
        )
    assert client.transport.sent == [
        ("arb_waveform.set", params),
        ("arb_waveform.preview", {"name": "pulse"}),
    ]
