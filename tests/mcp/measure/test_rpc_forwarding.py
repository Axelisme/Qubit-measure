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


@pytest.mark.parametrize(
    ("method", "params", "reply"),
    [
        (
            "context.ml_edit",
            {"name": "readout", "changes": []},
            {"applied": [], "failed": None, "skipped": []},
        ),
        (
            "project.info",
            {},
            {"chip_name": "chip", "qub_name": "q1", "result_dir": "/results/q1"},
        ),
        ("context.labels", {}, {"labels": ["zero", "half"]}),
    ],
)
def test_rpc_forwards_public_queries_and_tool_backed_commands(
    tmp_path: Path,
    method: str,
    params: dict[str, Any],
    reply: dict[str, Any],
) -> None:
    client = make_client(tmp_path, lambda _method, _params: reply)
    client.context.session.ensure_connected()
    client.transport.sent.clear()

    assert client.call("rpc_call", {"method": method, "params": params}) == reply
    assert client.transport.sent == [(method, params)]


def test_rpc_tool_backed_operation_keeps_handle_usable_for_cancel(
    tmp_path: Path,
) -> None:
    client = make_client(tmp_path)
    client.context.session.ensure_connected()
    client.transport.sent.clear()
    client.transport.replies["tab.analyze"] = {
        "ok": True,
        "result": {"operation_id": 73, "status": "running"},
    }
    client.transport.replies["operation.cancel"] = {
        "ok": True,
        "result": {"status": "cancelled"},
    }
    params = {"tab_id": "tab-1"}

    result = client.call("rpc_call", {"method": "tab.analyze", "params": params})
    assert result == {"handle": 1, "status": "running"}
    client.call("cancel", {"op": result["handle"]})
    assert client.transport.sent == [
        ("tab.analyze", params),
        ("operation.cancel", {"operation_id": 73}),
    ]


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
