"""Public analyze tool contracts through stdio and the recording GUI transport."""

import base64
import io
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from threading import Event
from typing import Any

import pytest
from zcu_tools.mcp.core.stdio_server import run_stdio_loop

from ._support import MeasureClient, RpcResponder, make_client

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
_INVALID_PNGS = [
    pytest.param(b"hello", id="non-png"),
    pytest.param(b"hello" + _PNG[-12:], id="non-png-with-iend"),
    pytest.param(_PNG[:8], id="signature-only"),
    pytest.param(_PNG[:46], id="truncated-pixels"),
    pytest.param(_PNG[:-12], id="missing-iend"),
    pytest.param(_PNG[:-1], id="truncated-iend-crc"),
    pytest.param(_PNG[:55] + bytes([_PNG[55] ^ 1]) + _PNG[56:], id="bad-idat-crc"),
]


@pytest.fixture
def clients(tmp_path: Path) -> Iterator[list[MeasureClient]]:
    created: list[MeasureClient] = []
    yield created
    for client in created:
        client.context.session.close()


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


def _result_reply(
    pane: str, names: list[str], params: dict[str, Any]
) -> dict[str, Any]:
    return {
        "summary": {"frequency": 5.0},
        "params": params,
        "operation_state": {
            f"{pane}_state": {
                "figure_names": names,
                "has_figure": bool(names),
                "available": True,
            },
        },
    }


@pytest.mark.parametrize("payload", [None, {"command": "set_value", "args": {"x": 2}}])
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
            "operation_id": 71,
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
        tmp_path,
        clients,
        lambda method, params: {"operation_id": 71, "state": {}, "figure": None},
    )
    reply = _call_stdio(monkeypatch, client, "tab_interact", {"tab": "t"})
    assert _data(reply) == {"handle": 1, "state": {}, "figure": None}
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
        (
            "tab.interact",
            {"tab_id": "t", "payload": {"command": "done"}, "include_figure": False},
        )
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
            assert params["operation_id"] == 71
            assert 0 < params["timeout"] <= 0.25
            return {"reason": "completed", "status": "finished"}
        if name == result_method:
            assert params == {"tab_id": "t", "operation_id": 71}
            return _result_reply(
                pane,
                ["fit", "residual"] if has_figure else [],
                {"gain": 2, "model": "fit"},
            )
        if name == "tab.save_image":
            assert params == {
                "tab_id": "t",
                "subtab_id": pane,
                "figure_name": params["figure_name"],
                "image_path": None,
                "operation_id": 71,
            }
            return {"image_path": f"/actual/{params['figure_name']}.png"}
        if name == "tab.get_figure":
            assert params == {
                "tab_id": "t",
                "subtab_id": pane,
                "operation_id": 71,
            }
            return {"png_b64": base64.b64encode(_PNG).decode()}
        raise AssertionError(name)

    client = _client(tmp_path, clients, respond)
    reply = _call_stdio(
        monkeypatch,
        client,
        "tab_analyze",
        {"tab": "t", "stage": stage, "params": {"gain": 2}},
    )
    result = _data(reply)
    assert result["status"] == "finished"
    assert result["execution"]
    assert result["stage"] == stage
    assert result["tab"] == "t"
    assert isinstance(result["op"], int)
    assert result["result"]["summary"] == {"frequency": 5.0}
    assert result["result"]["params"] == {"gain": 2, "model": "fit"}
    assert result["params"] == result["result"]["params"]
    assert result["invalidated"] == ["post.writeback"]
    assert result["operation_outcome"]["status"] == "finished"
    assert result["save_status"] == ("saved" if has_figure else "not_available")
    assert result["saved_images"] == (
        [
            {"figure_name": "fit", "image_path": "/actual/fit.png"},
            {"figure_name": "residual", "image_path": "/actual/residual.png"},
        ]
        if has_figure
        else []
    )
    assert result["remaining_images"] == []
    assert result["unconfirmed_image"] is None
    assert result["error"] is None
    _assert_figure(reply, present=has_figure)
    assert _methods(client) == [
        method,
        "operation.await",
        result_method,
        *(["tab.save_image", "tab.save_image", "tab.get_figure"] if has_figure else []),
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
    result = json.loads(reply["content"][0]["text"])
    assert bool(reply.get("isError")) == (status == "failed")
    assert result["status"] == status
    assert isinstance(result["op"], int)
    assert result["execution"]
    assert result["result"] is None
    assert result["save_status"] == "not_started"
    if status == "failed":
        assert result["error"]["message"] == "fit failed"
        assert result["error"]["phase"] == "operation"
    assert len(reply["content"]) == 1
    methods = _methods(client)
    assert methods[0] == "tab.analyze"
    assert methods[1:] and set(methods[1:]) == {"operation.await"}


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
        if method == "operation.await":
            return {"reason": "user_feedback", "status": "running"}
        assert method == "tab.interact"
        assert params == {"tab_id": "t"}
        return {**interaction, "operation_id": 71}

    client = _client(tmp_path, clients, respond)
    reply = _call_stdio(monkeypatch, client, "tab_analyze", {"tab": "t"})
    data = _data(reply)
    assert data["status"] == "interactive"
    assert data["tab"] == "t"
    assert isinstance(data["op"], int)
    assert data["params"] == {"gain": 2}
    assert data["execution"]
    for key in ("state", "commands", "plugin", "info", "preview_active"):
        assert data["interaction"][key] == interaction[key]
    _assert_figure(reply, present=has_figure)
    assert [m for m in _methods(client) if m != "operation.await"] == [
        "tab.analyze",
        "tab.interact",
    ]


def test_interactive_analysis_completes_after_gui_done(tmp_path, clients, monkeypatch):
    done = Event()
    saving = Event()

    def respond(method, params):
        if method == "tab.analyze":
            return {
                "operation_id": 71,
                "interactive": True,
                "params": {"gain": 2},
                "invalidated_on_success": ["post_analysis"],
            }
        if method == "tab.interact":
            return {
                "operation_id": 71,
                "state": {"value": 3},
                "commands": [{"name": "done"}],
                "figure": None,
            }
        if method == "operation.await":
            return (
                {"reason": "completed", "status": "finished"}
                if done.is_set()
                else {"reason": "user_feedback", "status": "running"}
            )
        if method == "tab.get_analyze_result":
            return _result_reply("analysis", ["fit"], {"gain": 2})
        if method == "tab.save_image":
            saving.set()
            return {"image_path": "/actual/fit.png"}
        if method == "tab.get_figure":
            return {"png_b64": base64.b64encode(_PNG).decode()}
        raise AssertionError(method)

    client = _client(tmp_path, clients, respond)
    started = _data(_call_stdio(monkeypatch, client, "tab_analyze", {"tab": "t"}))
    assert started["status"] == "interactive"
    execution = started["execution"]
    assert not saving.is_set()
    done.set()  # The GUI user, not an MCP command, completes the original operation.
    assert saving.wait(2), "completion stopped when the first tool call returned"
    reply = _call_stdio(
        monkeypatch, client, "wait", {"execution": execution, "timeout": 2}
    )
    completed = _data(reply)
    assert completed["execution"] == execution
    assert completed["op"] == started["op"]
    assert completed["status"] == "finished"
    assert completed["saved_images"] == [
        {"figure_name": "fit", "image_path": "/actual/fit.png"}
    ]
    assert completed["invalidated"] == ["post_analysis"]
    _assert_figure(reply, present=True)
    effects = [m for m in _methods(client) if m != "operation.await"]
    assert effects == [
        "tab.analyze",
        "tab.interact",
        "tab.get_analyze_result",
        "tab.save_image",
        "tab.get_figure",
    ]


@pytest.mark.parametrize("registered", [True, False], ids=["mcp-origin", "gui-origin"])
@pytest.mark.parametrize("outcome", ["finished", "failed"])
def test_done_joins_original_completion_without_duplicate_saves(
    tmp_path, clients, monkeypatch, registered, outcome
):
    done = Event()

    def respond(method, params):
        if method == "tab.analyze":
            return {
                "operation_id": 71,
                "interactive": True,
                "params": {"gain": 2},
                "invalidated_on_success": [],
            }
        if method == "tab.interact":
            if "payload" in params:
                assert params["payload"] == {"command": "done"}
                assert params["include_figure"] is False
                done.set()
            return {
                "operation_id": 71,
                "state": {"value": 3},
                "commands": [{"name": "done"}],
                "figure": None,
            }
        if method == "operation.await":
            return (
                {
                    "reason": "completed",
                    "status": outcome,
                    "error": "fit failed" if outcome == "failed" else None,
                }
                if done.is_set()
                else {"reason": "user_feedback", "status": "running"}
            )
        if method == "tab.get_analyze_result":
            return _result_reply("analysis", ["fit"], {"gain": 2})
        if method == "tab.save_image":
            return {"image_path": "/actual/fit.png"}
        if method == "tab.get_figure":
            return {"png_b64": base64.b64encode(_PNG).decode()}
        raise AssertionError(method)

    client = _client(tmp_path, clients, respond)
    started = (
        _data(_call_stdio(monkeypatch, client, "tab_analyze", {"tab": "t"}))
        if registered
        else None
    )
    reply = _call_stdio(
        monkeypatch,
        client,
        "tab_interact",
        {"tab": "t", "payload": {"command": "done"}},
    )
    assert bool(reply.get("isError")) is (outcome == "failed")
    completed = json.loads(reply["content"][0]["text"])
    assert completed["status"] == outcome
    assert completed["op"] == 1
    execution = completed["execution"]
    if started is not None:
        assert execution == started["execution"]
    terminal = _data(
        _call_stdio(monkeypatch, client, "wait", {"execution": execution, "timeout": 0})
    )
    assert terminal["status"] == outcome
    assert completed["save_status"] == (
        "saved" if outcome == "finished" else "not_started"
    )
    assert completed["saved_images"] == (
        [{"figure_name": "fit", "image_path": "/actual/fit.png"}]
        if outcome == "finished"
        else []
    )
    if outcome == "finished":
        assert completed["params"] == {"gain": 2}
        _assert_figure(reply, present=True)
    else:
        assert completed["error"]["reason"] == "analysis_failed"
    methods = _methods(client)
    assert methods.count("tab.analyze") == int(registered)
    assert methods.count("tab.interact") == 1 + int(registered)
    assert methods.count("tab.get_analyze_result") == int(outcome == "finished")
    assert methods.count("tab.save_image") == int(outcome == "finished")
    assert set(methods) <= {
        "tab.analyze",
        "tab.interact",
        "operation.await",
        "tab.get_analyze_result",
        "tab.save_image",
        "tab.get_figure",
    }


@pytest.mark.parametrize("tool", ["tab_analyze", "tab_interact"])
def test_malformed_interactive_image_is_a_tool_error_not_a_success(
    tmp_path, clients, monkeypatch, tool
):
    def respond(method, params):
        if method == "tab.analyze":
            return {
                "operation_id": 71,
                "interactive": True,
                "params": {},
                "invalidated_on_success": [],
            }
        if method == "operation.await":
            return {"reason": "user_feedback", "status": "running"}
        assert method == "tab.interact"
        return {
            "operation_id": 71,
            "figure": {"png_b64": "not valid base64!"},
            "state": {},
        }

    client = _client(tmp_path, clients, respond)
    reply = _call_stdio(monkeypatch, client, tool, {"tab": "t"})
    assert reply["isError"] is True
    assert len(reply["content"]) == 1
    assert "base64" in reply["content"][0]["text"]
    if tool == "tab_analyze":
        partial = json.loads(reply["content"][0]["text"])
        assert partial["status"] == "interactive"
        assert partial["execution"]
    expected = (
        ["tab.analyze", "tab.interact"] if tool == "tab_analyze" else ["tab.interact"]
    )
    assert [m for m in _methods(client) if m != "operation.await"] == expected


@pytest.mark.parametrize("tool", ["tab_analyze", "tab_interact"])
@pytest.mark.parametrize("png", _INVALID_PNGS)
def test_invalid_interactive_png_is_a_tool_error_without_retry(
    tmp_path, clients, monkeypatch, tool, png
):
    def respond(method, params):
        if method == "tab.analyze":
            return {
                "operation_id": 71,
                "interactive": True,
                "params": {},
                "invalidated_on_success": [],
            }
        if method == "operation.await":
            return {"reason": "user_feedback", "status": "running"}
        assert method == "tab.interact"
        return {
            "operation_id": 71,
            "figure": {"png_b64": base64.b64encode(png).decode()},
            "state": {},
        }

    client = _client(tmp_path, clients, respond)
    reply = _call_stdio(monkeypatch, client, tool, {"tab": "t"})
    assert reply["isError"] is True
    assert len(reply["content"]) == 1
    assert reply["content"][0]["type"] == "text"
    assert "Invalid PNG image" in reply["content"][0]["text"]
    if tool == "tab_analyze":
        partial = json.loads(reply["content"][0]["text"])
        assert partial["status"] == "interactive"
        assert partial["execution"]
    expected = (
        ["tab.analyze", "tab.interact"] if tool == "tab_analyze" else ["tab.interact"]
    )
    assert [m for m in _methods(client) if m != "operation.await"] == expected


@pytest.mark.parametrize(
    "stage, method, result_method",
    [
        ("primary", "tab.analyze", "tab.get_analyze_result"),
        ("post", "tab.post_analyze", "tab.get_post_analyze_result"),
    ],
)
@pytest.mark.parametrize("png", _INVALID_PNGS)
def test_invalid_finished_png_is_a_tool_error_without_retry(
    tmp_path, clients, monkeypatch, stage, method, result_method, png
):
    def respond(name, params):
        if name == method:
            return {
                "operation_id": 71,
                "interactive": False,
                "params": {},
                "invalidated_on_success": [],
            }
        if name == "operation.await":
            return {"reason": "completed", "status": "finished"}
        if name == result_method:
            return _result_reply(
                "analysis" if stage == "primary" else "post_analysis",
                ["fit"],
                {},
            )
        if name == "tab.save_image":
            return {"image_path": "/actual/fit.png"}
        assert name == "tab.get_figure"
        return {"png_b64": base64.b64encode(png).decode()}

    client = _client(tmp_path, clients, respond)
    reply = _call_stdio(
        monkeypatch, client, "tab_analyze", {"tab": "t", "stage": stage}
    )
    assert reply["isError"] is True
    assert len(reply["content"]) == 1
    assert reply["content"][0]["type"] == "text"
    data = json.loads(reply["content"][0]["text"])
    assert data["status"] == "failed"
    assert "Invalid PNG image" in data["error"]["message"]
    assert data["error"]["phase"] == "figure_read"
    assert data["result"]["summary"] == {"frequency": 5.0}
    assert data["saved_images"] == [
        {"figure_name": "fit", "image_path": "/actual/fit.png"}
    ]
    assert data["save_status"] == "saved"
    assert _methods(client) == [
        method,
        "operation.await",
        result_method,
        "tab.save_image",
        "tab.get_figure",
    ]


@pytest.mark.parametrize("outcome", ["finished", "failed"])
def test_execution_query_observes_background_completion_without_reconnect(
    tmp_path, clients, monkeypatch, outcome
):
    release = Event()
    awaiting = Event()

    def respond(method, params):
        if method == "tab.analyze":
            return {
                "operation_id": 71,
                "interactive": False,
                "params": {"model": "fit"},
                "invalidated_on_success": [],
            }
        if method == "operation.await":
            awaiting.set()
            assert release.wait(10), "test did not release the GUI operation"
            return {"reason": "completed", "status": outcome, "error": None}
        if method == "tab.get_analyze_result":
            return _result_reply("analysis", ["fit"], {"model": "fit"})
        if method == "tab.save_image":
            return {"image_path": "/actual/fit.png"}
        if method == "tab.get_figure":
            return {"png_b64": base64.b64encode(_PNG).decode()}
        raise AssertionError(method)

    client = _client(tmp_path, clients, respond)
    try:
        started = _data(_call_stdio(monkeypatch, client, "tab_analyze", {"tab": "t"}))
        assert awaiting.wait(2)
        assert started["status"] == "running"
        execution = started["execution"]
        running = _data(
            _call_stdio(
                monkeypatch, client, "wait", {"execution": execution, "timeout": 0}
            )
        )
        assert running["status"] == "running"
        assert running["cancel_requested"] is False
        assert running["elapsed_s"] >= 0
        snapshot = client.call("status", {"execution": execution})
        snapshot["params"]["model"] = "caller mutation"
        assert client.call("status", {"execution": execution})["params"] == {
            "model": "fit"
        }
        assert _methods(client) == ["tab.analyze", "operation.await"]
        release.set()
        completed_reply = _call_stdio(
            monkeypatch, client, "wait", {"execution": execution, "timeout": 2}
        )
        completed = _data(completed_reply)
        assert completed["status"] == outcome
        assert completed["operation_outcome"]["status"] == outcome
        assert completed["cancel_requested"] is False
        assert completed["save_status"] == (
            "saved" if outcome == "finished" else "not_started"
        )
        assert completed["saved_images"] == (
            [{"figure_name": "fit", "image_path": "/actual/fit.png"}]
            if outcome == "finished"
            else []
        )
        _assert_figure(completed_reply, present=outcome == "finished")
        sent = list(client.transport.sent)
        client.transport.close()
        terminal = _data(
            _call_stdio(monkeypatch, client, "status", {"execution": execution})
        )
        repeated = _data(
            _call_stdio(
                monkeypatch, client, "wait", {"execution": execution, "timeout": 0}
            )
        )
        assert terminal == {
            key: value for key, value in repeated.items() if key != "elapsed_s"
        }
        assert terminal["status"] == outcome
        assert client.transport.sent == sent
    finally:
        release.set()


def _inject_analysis_rejection(
    client: MeasureClient,
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rejection = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "reason": "superseded_result",
            "message": "original analysis was replaced",
        },
    }
    if failure == "result_rejected":
        client.transport.replies["tab.get_analyze_result"] = rejection
    elif failure == "save_rejected":
        client.transport.replies["tab.save_image"] = lambda params: (
            rejection
            if params["figure_name"] == "residual"
            else {"ok": True, "result": {"image_path": "/actual/fit.png"}}
        )
    elif failure == "preview_rejected":
        client.transport.replies["tab.get_figure"] = rejection
    elif failure == "local_write":

        def reject_write(path, data):
            raise OSError("preview filesystem is full")

        monkeypatch.setattr(Path, "write_bytes", reject_write)


@pytest.mark.parametrize(
    "failure,phase,save_status,confirmed,unconfirmed",
    [
        ("result_rejected", "result_read", "not_started", [], None),
        ("save_rejected", "image_save", "incomplete", ["fit"], None),
        ("save_lost", "image_save", "unknown", ["fit"], "residual"),
        ("save_bad_path", "image_save", "unknown", ["fit"], "residual"),
        ("after_result_eof", "image_save", "incomplete", [], None),
        ("after_save_eof", "figure_read", "saved", ["fit", "residual"], None),
        ("preview_rejected", "figure_read", "saved", ["fit", "residual"], None),
        ("local_write", "figure_read", "saved", ["fit", "residual"], None),
    ],
)
def test_analysis_failure_retains_confirmed_prefix_without_replay(
    tmp_path,
    clients,
    monkeypatch,
    failure,
    phase,
    save_status,
    confirmed,
    unconfirmed,
):
    def respond(method, params):
        if method == "tab.analyze":
            return {
                "operation_id": 71,
                "interactive": False,
                "params": {"gain": 2},
                "invalidated_on_success": [],
            }
        if method == "operation.await":
            return {"reason": "completed", "status": "finished"}
        if method == "tab.get_analyze_result":
            if failure == "after_result_eof":
                client.transport.is_open = False
            return _result_reply("analysis", ["fit", "residual"], {"gain": 2})
        if method == "tab.save_image":
            if params["figure_name"] == "residual":
                if failure == "save_lost":
                    client.transport.is_open = False
                    raise OSError("socket closed during image save")
                if failure == "save_bad_path":
                    return {"image_path": None}
                if failure == "after_save_eof":
                    client.transport.is_open = False
            return {"image_path": f"/actual/{params['figure_name']}.png"}
        assert method == "tab.get_figure"
        return {"png_b64": base64.b64encode(_PNG).decode()}

    client = _client(tmp_path, clients, respond)
    _inject_analysis_rejection(client, failure, monkeypatch)

    reply = _call_stdio(monkeypatch, client, "tab_analyze", {"tab": "t"})
    assert reply["isError"] is True
    assert len(reply["content"]) == 1
    data = json.loads(reply["content"][0]["text"])
    assert data["status"] == "failed"
    assert data["error"]["phase"] == phase
    assert data["save_status"] == save_status
    assert data["saved_images"] == [
        {"figure_name": name, "image_path": f"/actual/{name}.png"} for name in confirmed
    ]
    assert data["unconfirmed_image"] == unconfirmed
    if failure == "result_rejected":
        assert data["result"] is None
        assert data["remaining_images"] is None
    else:
        assert data["result"]["summary"] == {"frequency": 5.0}
        assert data["remaining_images"] == [
            name for name in ["fit", "residual"] if name not in confirmed
        ]
    assert data["figure"] is None
    methods = _methods(client)
    assert methods.count("tab.analyze") == 1
    assert methods.count("tab.get_analyze_result") == 1
    assert methods.count("tab.get_figure") == (
        1 if failure in ("preview_rejected", "local_write") else 0
    )
    assert [
        params["figure_name"]
        for method, params in client.transport.sent
        if method == "tab.save_image"
    ] == (
        []
        if failure in ("result_rejected", "after_result_eof")
        else ["fit", "residual"]
    )
