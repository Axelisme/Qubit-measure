"""Public analyze tool contracts through stdio and the recording GUI transport."""

import base64
import io
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from zcu_tools.mcp.core.stdio_server import run_stdio_loop

from ._support import MeasureClient, RpcResponder, make_client

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


@pytest.fixture
def clients(tmp_path: Path) -> Iterator[list[MeasureClient]]:
    created: list[MeasureClient] = []
    yield created
    for client in created:
        client.context.session.cleanup_pngs()


def _client(
    tmp_path: Path, clients: list[MeasureClient], responder: RpcResponder
) -> MeasureClient:
    client = make_client(tmp_path, responder)
    clients.append(client)
    return client


def _call_stdio(
    monkeypatch: pytest.MonkeyPatch,
    client: MeasureClient,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    stdin = io.TextIOWrapper(
        io.BytesIO((json.dumps(request) + "\n").encode()), encoding="utf-8"
    )
    out = io.BytesIO()
    stdout = io.TextIOWrapper(out, encoding="utf-8", write_through=True)
    with monkeypatch.context() as patch:
        patch.setattr(sys, "stdin", stdin)
        patch.setattr(sys, "stdout", stdout)
        run_stdio_loop(client.context.config, client.tools)
    return json.loads(out.getvalue())["result"]


def _data(reply: dict[str, Any]) -> dict[str, Any]:
    assert not reply.get("isError"), reply
    return json.loads(reply["content"][0]["text"])


def _assert_figure(reply: dict[str, Any], *, present: bool) -> Path | None:
    data = _data(reply)
    if not present:
        assert data["figure"] is None
        assert len(reply["content"]) == 1
        return None
    path = Path(data["figure"])
    assert path.is_absolute()
    assert path.read_bytes() == _PNG
    assert reply["content"][1:] == [
        {
            "type": "image",
            "mimeType": "image/png",
            "data": base64.b64encode(_PNG).decode("ascii"),
        }
    ]
    return path


def _methods(client: MeasureClient) -> list[str]:
    return [
        name
        for name, _ in client.transport.sent
        if name not in ("wire.version", "rpc.catalog")
    ]


@pytest.mark.parametrize(
    "payload", [None, {"command": "set_value", "args": {"x": 2}}, {"command": "done"}]
)
def test_interact_forwards_once_and_materializes_session_image(
    tmp_path, clients, monkeypatch, payload
):
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
            "figure": {"png_b64": base64.b64encode(_PNG).decode(), "bytes": len(_PNG)},
        }

    client = _client(tmp_path, clients, respond)
    arguments = {"tab": "t"}
    if payload is not None:
        arguments["payload"] = payload
    reply = _call_stdio(monkeypatch, client, "tab_interact", arguments)
    result = _data(reply)
    assert result["state"] == state
    assert result["info"] == {"label": "picker"}
    assert result["commands"] == [{"name": "set_value"}, {"name": "done"}]
    assert result["plugin"] == "generic-test"
    assert result["preview_active"] is True
    path = _assert_figure(reply, present=True)
    assert _methods(client) == ["tab.interact"]
    client.context.session.cleanup_pngs()
    assert path is not None and not path.exists()


def test_interact_headless_and_wire_failure_do_not_retry(
    tmp_path, clients, monkeypatch
):
    client = _client(
        tmp_path, clients, lambda method, params: {"state": {}, "figure": None}
    )
    reply = _call_stdio(monkeypatch, client, "tab_interact", {"tab": "t"})
    assert _data(reply) == {"state": {}, "figure": None}
    _assert_figure(reply, present=False)
    client.transport.sent.clear()
    client.transport.replies["tab.interact"] = {
        "ok": False,
        "error": {"code": "precondition_failed", "message": "no active session"},
    }
    reply = _call_stdio(
        monkeypatch,
        client,
        "tab_interact",
        {"tab": "t", "payload": {"command": "done"}},
    )
    assert reply["isError"] is True
    assert len(reply["content"]) == 1
    assert "no active session" in reply["content"][0]["text"]
    assert client.transport.sent == [
        ("tab.interact", {"tab_id": "t", "payload": {"command": "done"}})
    ]


@pytest.mark.parametrize(
    "stage, method, result_method, pane",
    [
        ("primary", "tab.analyze", "tab.get_analyze_result", "analysis"),
        ("post", "tab.post_analyze", "tab.get_post_analyze_result", "post_analysis"),
    ],
)
@pytest.mark.parametrize("has_figure", [True, False])
def test_finished_analysis_uses_start_facts_without_hidden_pre_reads(
    tmp_path, clients, monkeypatch, stage, method, result_method, pane, has_figure
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
        if name == "tab.get_figure":
            assert params["subtab_id"] == pane
            Path(params["out_path"]).write_bytes(_PNG)
            return {"saved_to": params["out_path"]}
        raise AssertionError(name)

    client = _client(tmp_path, clients, respond)
    if not has_figure:
        client.transport.replies["tab.get_figure"] = {
            "ok": False,
            "error": {"code": "precondition_failed", "message": "no figure"},
        }
    reply = _call_stdio(
        monkeypatch,
        client,
        "tab_analyze",
        {"tab": "t", "stage": stage, "params": {"gain": 2}},
    )
    result = _data(reply)
    assert result == {
        "status": "finished",
        "summary": {"frequency": 5.0},
        "figure": result["figure"],
        "params": {"gain": 2, "model": "fit"},
        "invalidated": ["post.writeback"],
    }
    _assert_figure(reply, present=has_figure)
    assert _methods(client) == [
        method,
        "operation.await",
        result_method,
        "tab.get_figure",
    ]


@pytest.mark.parametrize("status", ["running", "failed", "cancelled"])
def test_unfinished_analysis_never_reads_success_payload(
    tmp_path, clients, monkeypatch, status
):
    def respond(method, params):
        if method == "tab.analyze":
            assert params == {"tab_id": "t", "updates": {}}
            return {
                "operation_id": 71,
                "interactive": False,
                "params": {},
                "invalidated_on_success": [],
            }
        if method == "operation.await":
            if status == "running":
                return {"reason": "timeout"}
            return {
                "reason": "completed",
                "status": status,
                **({"error": "fit failed"} if status == "failed" else {}),
            }
        if method == "operation.progress":
            return {"active": False}
        raise AssertionError(method)

    client = _client(tmp_path, clients, respond)
    reply = _call_stdio(monkeypatch, client, "tab_analyze", {"tab": "t"})
    result = _data(reply)
    assert result["status"] == status
    assert isinstance(result["op"], int)
    if status == "failed":
        assert result["error"] == "fit failed"
    assert len(reply["content"]) == 1
    expected_methods = ["tab.analyze", "operation.await"]
    if status == "running":
        expected_methods.append("operation.progress")
    assert _methods(client) == expected_methods


@pytest.mark.parametrize("has_figure", [True, False])
def test_interactive_start_hands_off_current_state_without_waiting(
    tmp_path, clients, monkeypatch, has_figure
):
    interaction = {
        "plugin": "picker",
        "state": {"value": 3},
        "commands": [{"name": "done"}],
        "info": {"label": "fit"},
        "preview_active": False,
        "figure": (
            {"png_b64": base64.b64encode(_PNG).decode()} if has_figure else None
        ),
    }

    def respond(method, params):
        if method == "tab.analyze":
            return {
                "operation_id": 71,
                "interactive": True,
                "params": {"gain": 2},
                "invalidated_on_success": [],
            }
        assert method == "tab.interact"
        assert params == {"tab_id": "t"}
        return interaction

    client = _client(tmp_path, clients, respond)
    reply = _call_stdio(monkeypatch, client, "tab_analyze", {"tab": "t"})
    data = _data(reply)
    assert data["status"] == "interactive"
    assert data["tab"] == "t"
    assert isinstance(data["op"], int)
    assert data["params"] == {"gain": 2}
    for key in ("state", "commands", "plugin", "info", "preview_active"):
        assert data[key] == interaction[key]
    _assert_figure(reply, present=has_figure)
    assert _methods(client) == ["tab.analyze", "tab.interact"]


@pytest.mark.parametrize("tool", ["tab_analyze", "tab_interact"])
def test_malformed_interactive_image_is_a_tool_error_not_a_success(
    tmp_path, clients, monkeypatch, tool
):
    def respond(method, params):
        if method == "tab.analyze":
            return {"operation_id": 71, "interactive": True, "params": {}}
        assert method == "tab.interact"
        return {"figure": {"png_b64": "not valid base64!"}, "state": {}}

    client = _client(tmp_path, clients, respond)
    reply = _call_stdio(monkeypatch, client, tool, {"tab": "t"})
    assert reply["isError"] is True
    assert len(reply["content"]) == 1
    assert "base64" in reply["content"][0]["text"]
    expected = (
        ["tab.analyze", "tab.interact"] if tool == "tab_analyze" else ["tab.interact"]
    )
    assert _methods(client) == expected
