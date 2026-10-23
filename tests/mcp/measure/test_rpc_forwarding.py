"""Public MCP forwarding with GUI-owned concurrency enforcement."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError

from ._support import make_client


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"summaries": True},
        {"summaries": False},
        {"path": "aliases.0.resolved", "summaries": True},
        {"path": ""},
    ],
)
def test_context_browse_forwarding_preserves_one_request_and_reply(
    tmp_path: Path,
    params: dict[str, object],
) -> None:
    expected = {
        "target": {"kind": "saved_point", "entry_id": "entry", "point": "working"},
        "aliases": [{"alias": "drive", "value": "Q1", "resolved": "Q1"}],
        "path": "aliases.0.resolved",
        "value": "Q1",
    }
    client = make_client(tmp_path, lambda method, params: expected)
    client.context.session.ensure_connected()
    client.transport.sent.clear()
    assert (
        client.call("rpc_call", {"method": "context.snapshot", "params": params})
        == expected
    )
    assert client.transport.sent == [("context.snapshot", params)]


def test_rpc_preserves_reply_and_sends_one_request(tmp_path: Path) -> None:
    expected = {"loaded": True, "analysis_error": None}
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
            {"kind": "module", "name": "Q1.readout.seed", "edits": []},
            {"applied": [], "failed": None, "skipped": []},
        ),
        (
            "entry.info",
            {},
            {
                "entry": {"entry_id": "entry-uuid", "name": "synthetic"},
                "files": [
                    {
                        "document": "parameters",
                        "path": "/entry/points/base/point.yaml",
                        "revision": "a" * 64,
                    }
                ],
            },
        ),
        (
            "context.reload",
            {
                "path": "/entry/points/base/point.yaml",
                "expected_revision": "a" * 64,
            },
            {
                "document": "parameters",
                "path": "/entry/points/base/point.yaml",
                "revision": "b" * 64,
            },
        ),
        (
            "context.reload",
            {
                "path": "/entry/points/base/aliases.yaml",
                "expected_revision": None,
            },
            {
                "document": "aliases",
                "path": "/entry/points/base/aliases.yaml",
                "revision": "c" * 64,
            },
        ),
        ("value.list", {}, {"keys": ["Q1.rate"]}),
        ("value.read", {"key": "Q1.rate"}, {"value": 10.0}),
        (
            "tab.parameter_read",
            {"tab_id": "tab-1", "path": "Q1.rate"},
            {"value": 10.0},
        ),
        (
            "context.set_parameter",
            {"path": "Q1.rate", "value": 20.0, "expected_revision": "a" * 64},
            {"before": 10.0, "after": 20.0},
        ),
        (
            "context.aliases_set",
            {"rows": [{"alias": "driver", "value": "Q1"}]},
            {"rows": [{"alias": "driver", "value": "Q1"}]},
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


def test_rpc_admission_error_requires_explicit_retry_by_caller(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    client.context.session.ensure_connected()
    client.transport.sent.clear()
    client.transport.replies["tab.load_data"] = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "reason": "not_holder",
            "message": "caller does not hold control",
        },
    }
    params = {"tab_id": "tab-1", "data_path": "saved.h5"}
    with pytest.raises(GuiRpcError) as failure:
        client.call("rpc_call", {"method": "tab.load_data", "params": params})
    assert failure.value.reason == "not_holder"
    assert failure.value.code == "precondition_failed"
    assert "caller does not hold control" in str(failure.value)
    assert client.transport.sent == [("tab.load_data", params)]


@pytest.mark.parametrize(
    ("method", "params"),
    [("point.list", {}), ("point.use", {"label": "base"})],
)
def test_point_reads_and_use_forward_one_request(
    tmp_path: Path,
    method: str,
    params: Mapping[str, object],
) -> None:
    client = make_client(tmp_path, lambda method, params: {"selected": "base"})
    client.context.session.ensure_connected()
    client.transport.sent.clear()
    assert client.call("rpc_call", {"method": method, "params": dict(params)}) == {
        "selected": "base"
    }
    assert client.transport.sent == [(method, dict(params))]


def test_point_new_gui_failure_is_preserved_without_retry(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    client.context.session.ensure_connected()
    client.transport.sent.clear()
    data = {"entry_id": "entry-uuid", "point": "retained", "created": True}
    client.transport.replies["point.new"] = {
        "ok": False,
        "error": {
            "code": "controller_error",
            "reason": "point_binding_failed",
            "message": "created but not installed",
            "data": data,
        },
    }
    params = {"label": "retained"}
    with pytest.raises(GuiRpcError) as failure:
        client.call("rpc_call", {"method": "point.new", "params": params})
    assert failure.value.reason == "point_binding_failed"
    assert failure.value.code == "controller_error"
    assert "created but not installed" in str(failure.value)
    assert client.transport.sent == [("point.new", params)]


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
