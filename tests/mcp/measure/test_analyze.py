"""Public analyze tool contracts over the recording GUI transport."""

import base64
from pathlib import Path

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError

from ._support import make_client


@pytest.mark.parametrize(
    "payload", [None, {"command": "set_value", "args": {"x": 2}}, {"command": "done"}]
)
def test_interact_forwards_once_and_materializes_session_image(tmp_path, payload):
    png = b"\x89PNG\r\n\x1a\nfixture"
    state = {"value": 2}

    def respond(method, params):
        assert method == "tab.interact"
        expected = {"tab_id": "t"}
        if payload is not None:
            expected["payload"] = payload
        assert params == expected
        return {
            "plugin": "generic-test",
            "state": state,
            "info": {"label": "picker"},
            "commands": [{"name": "set_value"}, {"name": "done"}],
            "preview_active": True,
            "figure": {"png_b64": base64.b64encode(png).decode(), "bytes": len(png)},
        }

    client = make_client(tmp_path, respond)
    arguments = {"tab": "t"}
    if payload is not None:
        arguments["payload"] = payload
    result = client.call("tab_interact", arguments)
    assert result["state"] == state
    assert result["info"] == {"label": "picker"}
    assert result["commands"] == [{"name": "set_value"}, {"name": "done"}]
    assert result["plugin"] == "generic-test"
    assert result["preview_active"] is True
    path = Path(result["figure"])
    assert path.is_absolute() and path.read_bytes() == png
    assert [
        name
        for name, _ in client.transport.sent
        if name not in ("wire.version", "rpc.catalog")
    ] == ["tab.interact"]
    client.context.session.cleanup_pngs()
    assert not path.exists()


def test_interact_headless_and_wire_failure_do_not_retry(tmp_path):
    client = make_client(tmp_path, lambda method, params: {"state": {}, "figure": None})
    assert client.call("tab_interact", {"tab": "t"}) == {"state": {}, "figure": None}
    client.transport.sent.clear()
    client.transport.replies["tab.interact"] = {
        "ok": False,
        "error": {"code": "precondition_failed", "message": "no active session"},
    }
    with pytest.raises(GuiRpcError, match="no active session"):
        client.call("tab_interact", {"tab": "t", "payload": {"command": "done"}})
    assert client.transport.sent == [
        ("tab.interact", {"tab_id": "t", "payload": {"command": "done"}})
    ]


@pytest.mark.parametrize(
    "stage, method, result_method",
    [
        ("primary", "tab.analyze", "tab.get_analyze_result"),
        ("post", "tab.post_analyze", "tab.get_post_analyze_result"),
    ],
)
def test_finished_analysis_uses_start_facts_without_hidden_pre_reads(
    tmp_path, stage, method, result_method
):
    def respond(name, params):
        if name == method:
            assert params == {"tab_id": "t", "updates": {"gain": 2}}
            return {
                "operation_id": 71,
                "interactive": False,
                "params": {"gain": 2, "model": "fit"},
                "invalidated_on_success": ["post.writeback"],
            }
        if name == "operation.await":
            assert params["timeout"] == 2.0
            return {"reason": "completed", "status": "finished"}
        if name == result_method:
            return {"summary": {"frequency": 5.0}}
        raise AssertionError(name)

    client = make_client(tmp_path, respond)
    client.transport.replies["tab.get_figure"] = {
        "ok": False,
        "error": {"code": "precondition_failed", "message": "no figure"},
    }
    assert client.call(
        "tab_analyze", {"tab": "t", "stage": stage, "params": {"gain": 2}}
    ) == {
        "status": "finished",
        "summary": {"frequency": 5.0},
        "figure": None,
        "params": {"gain": 2, "model": "fit"},
        "invalidated": ["post.writeback"],
    }
    assert [
        name
        for name, _ in client.transport.sent
        if name not in ("wire.version", "rpc.catalog")
    ] == [
        method,
        "operation.await",
        result_method,
        "tab.get_figure",
    ]


@pytest.mark.parametrize("status", ["interactive", "running", "failed"])
def test_unfinished_analysis_never_reads_success_payload(tmp_path, status):
    def respond(method, params):
        if method == "tab.analyze":
            assert params == {"tab_id": "t", "updates": {}}
            return {
                "operation_id": 71,
                "interactive": status == "interactive",
                "params": {},
                "invalidated_on_success": [],
            }
        if method == "operation.await":
            if status == "running":
                return {"reason": "timeout"}
            return {"reason": "completed", "status": "failed", "error": "fit failed"}
        if method == "operation.progress":
            return {"active": False}
        raise AssertionError(method)

    client = make_client(tmp_path, respond)
    result = client.call("tab_analyze", {"tab": "t"})
    assert result["status"] == status
    assert isinstance(result["op"], int)
    if status == "failed":
        assert result["error"] == "fit failed"
    methods = [
        name
        for name, _ in client.transport.sent
        if name not in ("wire.version", "rpc.catalog")
    ]
    assert (
        methods
        == {
            "interactive": ["tab.analyze"],
            "running": ["tab.analyze", "operation.await", "operation.progress"],
            "failed": ["tab.analyze", "operation.await"],
        }[status]
    )
