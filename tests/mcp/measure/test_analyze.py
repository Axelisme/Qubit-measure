"""Public analyze tool contracts through stdio and the recording GUI transport."""

import base64
import io
import json
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Thread
from typing import Any

import pytest
from zcu_tools.mcp.core.bridge import GuiTransportTimeoutError
from zcu_tools.mcp.core.stdio_server import run_stdio_loop
from zcu_tools.mcp.measure.session import GuiConnection

from ._support import MeasureClient, RpcResponder, make_client

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
_AMBIGUOUS_SAVE_ERRORS = {
    "handler_timeout": {"code": "timeout", "message": "GUI handler timed out"},
    "encoding_failed": {
        "code": "internal",
        "reason": "response_encoding_failed",
        "message": "request may have executed",
    },
}
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
    tmp_path: Path, clients: list[MeasureClient], responder: RpcResponder | None = None
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
    return json.loads(out.getvalue(), parse_constant=_reject_json_constant)["result"]


def _call_full_execution_stdio(
    monkeypatch: pytest.MonkeyPatch,
    client: MeasureClient,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Keep wire delivery checks while explicitly reading captured native detail."""
    reply = _call_stdio(monkeypatch, client, name, arguments)
    data = json.loads(reply["content"][0]["text"])
    full = _call_stdio(
        monkeypatch,
        client,
        "status",
        {"execution": data["execution"], "detail": "full"},
    )
    if "elapsed_s" in data:
        full_data = json.loads(full["content"][0]["text"])
        full_data["elapsed_s"] = data["elapsed_s"]
        full["content"][0]["text"] = json.dumps(
            full_data, separators=(",", ":"), allow_nan=False
        )
    reply["content"][0] = full["content"][0]
    return reply


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def _data(reply: dict[str, Any]) -> dict[str, Any]:
    assert not reply.get("isError"), reply
    return json.loads(reply["content"][0]["text"], parse_constant=_reject_json_constant)


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


@pytest.mark.parametrize("stage", ["primary", "post"])
@pytest.mark.parametrize("outcome", ["finished", "failed", "partial"])
def test_analysis_initial_wait_and_status_share_the_same_summary(
    tmp_path, clients, stage, outcome
):
    pane = "analysis" if stage == "primary" else "post_analysis"

    def respond(method, params):
        if method in {"tab.analyze", "tab.post_analyze"}:
            return _start_reply({"gain": 2.0}, [])
        if method == "operation.await":
            return {
                "reason": "completed",
                "status": "failed" if outcome == "failed" else "finished",
                "error": "fit failed" if outcome == "failed" else None,
            }
        if method in {"tab.get_analyze_result", "tab.get_post_analyze_result"}:
            return _result_reply(pane, ["fit", "residual"], {"gain": 2.0})
        if method == "tab.save_image":
            return {"image_path": "/actual/" + params["figure_name"] + ".png"}
        if method == "tab.get_figure":
            return {"png_b64": base64.b64encode(_PNG).decode()}
        raise AssertionError(method)

    client = _client(tmp_path, clients, respond)
    if outcome == "partial":

        def save(params):
            if params["figure_name"] == "residual":
                return {
                    "ok": False,
                    "error": {
                        "code": "precondition_failed",
                        "reason": "save_failed",
                        "message": "Destination unavailable",
                    },
                }
            return {"ok": True, "result": respond("tab.save_image", params)}

        client.transport.replies["tab.save_image"] = save
    initial = client.call("tab_analyze", {"tab": "t", "stage": stage})
    execution = initial.data["execution"]
    before = len(client.transport.sent)
    summary = client.call("status", {"execution": execution})
    full = client.call("status", {"execution": execution, "detail": "full"})
    waited = client.call("wait", {"execution": execution, "timeout": 0})
    assert initial.data == summary
    assert {k: v for k, v in waited.data.items() if k != "elapsed_s"} == summary
    assert len(client.transport.sent) == before
    assert summary["recipe"] is None
    assert summary["run_id"] is None
    assert summary["run_op"] is None
    assert summary["analysis"]["stage"] == stage
    assert summary["steps"]["analysis"][stage]["status"] == (
        "failed" if outcome == "failed" else "finished"
    )
    assert summary["status"] == ("finished" if outcome == "finished" else "failed")
    if outcome != "failed":
        assert summary["analysis"][stage]["params"] == {"gain": 2.0}
        assert summary["analysis"][stage]["details"] == {"frequency": 5.0}
        assert summary["artifacts"][pane]["fit"]["members"]["image"] == [
            {"path": "/actual/fit.png", "status": "saved"}
        ]
    else:
        assert summary["error"] == full["error"]
    if outcome == "partial":
        assert summary["steps"]["analysis_save"][stage]["status"] == "incomplete"
        assert summary["error"]["phase"] == "image_save"


@pytest.mark.parametrize("stage", ["primary", "post"])
@pytest.mark.parametrize(
    "receipt, reason, step_status",
    [
        ("lost", "connection_lost", "unknown"),
        ("timeout", "gui_transport_timeout", "unknown"),
        ("handler_timeout", "gui_handler_timeout", "unknown"),
        ("internal", "injected", "unknown"),
        ("stale", "stale_version", "not_started"),
        ("busy", "operation_busy", "not_started"),
    ],
)
def test_analysis_start_failure_retains_a_queryable_execution(
    tmp_path, clients, monkeypatch, stage, receipt, reason, step_status
):
    method = "tab.analyze" if stage == "primary" else "tab.post_analyze"
    client = _client(tmp_path, clients)
    pending = Event()
    if receipt in {"handler_timeout", "internal", "stale", "busy"}:
        client.transport.replies[method] = {
            "ok": False,
            "error": {
                "code": {
                    "handler_timeout": "timeout",
                    "internal": "internal",
                    "stale": "precondition_failed",
                    "busy": "busy",
                }[receipt],
                "reason": None if receipt == "handler_timeout" else reason,
                "message": "Analysis start did not provide a handle",
            },
        }
    elif receipt == "timeout":
        send_line = client.transport.send_line

        def send(payload):
            if payload["method"] != method:
                return send_line(payload)
            client.transport.sent.append((payload["method"], payload["params"]))
            raise GuiTransportTimeoutError(method, 0.01)

        monkeypatch.setattr(client.transport, "send_line", send)
    else:
        pending = _hold_wire_reply(client, monkeypatch, method)

    with ThreadPoolExecutor(max_workers=1) as pool:
        called = pool.submit(client.call, "tab_analyze", {"tab": "t", "stage": stage})
        if receipt == "lost":
            assert pending.wait(1)
            client.transport.close()
            assert client.transport.on_closed is not None
            client.transport.on_closed(None)
        initial = called.result(timeout=2)
    assert initial.is_error is True
    key = initial.data["execution"]
    before = len(client.transport.sent)
    summary = client.call("status", {"execution": key})
    full = client.call("status", {"execution": key, "detail": "full"})
    waited = client.call("wait", {"execution": key, "timeout": 0})
    assert summary == initial.data
    assert {k: v for k, v in waited.data.items() if k != "elapsed_s"} == summary
    assert summary["status"] == "failed"
    assert summary["steps"]["analysis"][stage]["status"] == step_status
    assert summary["error"]["reason"] == reason
    assert full["start"]["status"] == step_status
    assert full["op"] is None
    assert full["params"] is None
    assert summary["steps"]["analysis_save"][stage]["status"] == "not_started"
    assert summary["artifacts"]["analysis"] == {}
    assert summary["artifacts"]["post_analysis"] == {}
    assert len(client.transport.sent) == before
    assert _methods(client) == [method]


@pytest.mark.parametrize("stage", ["primary", "post"])
@pytest.mark.parametrize("cancel", [False, True])
def test_pending_analysis_receipt_preserves_identity_and_cancel_intent(
    tmp_path, clients, stage, cancel
):
    method = "tab.analyze" if stage == "primary" else "tab.post_analyze"
    pane = "analysis" if stage == "primary" else "post_analysis"
    pending, release = Event(), Event()

    def respond(name, params):
        if name == method:
            pending.set()
            assert release.wait(2)
            return _start_reply({"gain": 2.0}, [])
        if name == "operation.cancel":
            return {"status": "cancelling"}
        if name == "operation.await":
            return {"reason": "completed", "status": "finished"}
        if name in {"tab.get_analyze_result", "tab.get_post_analyze_result"}:
            return _result_reply(pane, [], {"gain": 2.0})
        raise AssertionError(name)

    client = _client(tmp_path, clients, respond)
    with ThreadPoolExecutor(max_workers=2) as pool:
        called = pool.submit(client.call, "tab_analyze", {"tab": "t", "stage": stage})
        try:
            assert pending.wait(1)
            snapshots = client.context.session.executions.snapshots()
            assert len(snapshots) == 1
            key = snapshots[0].execution
            before = len(client.transport.sent)
            summary = client.call("status", {"execution": key})
            full = client.call("status", {"execution": key, "detail": "full"})
            assert summary["op"] is None
            assert summary["steps"]["analysis"][stage]["status"] == "unknown"
            assert full["start"]["status"] == "unknown"
            assert len(client.transport.sent) == before
            if cancel:
                requested = pool.submit(client.call, "cancel", {"execution": key})
                stopped = requested.result(timeout=0.5)
                assert stopped.data["cancel_requested"] is True
                assert stopped.data["gui_cancel"]["status"] == "not_needed"
                assert len(client.transport.sent) == before
            release.set()
            initial = called.result(timeout=2)
            completed = client.call("wait", {"execution": key, "timeout": 2})
        finally:
            release.set()
    assert initial.data["execution"] == key
    assert completed.data["execution"] == key
    assert completed.data["status"] == ("cancelled" if cancel else "finished")
    assert completed.data["steps"]["analysis"][stage] == {
        "status": "finished",
        "reason": "completed",
    }
    full = client.call("status", {"execution": key, "detail": "full"})
    assert full["op"] == 1
    assert full["start"]["status"] == "running"
    assert full["cancel_requested"] == cancel
    assert full["operation_outcome"]["status"] == "finished"
    assert len(client.context.session.executions.snapshots()) == 1
    assert _methods(client) == (
        [method, "operation.cancel", "operation.await"]
        if cancel
        else [method, "operation.await", "tab.get_analyze_result"
              if stage == "primary" else "tab.get_post_analyze_result"]
    )


def _start_reply(
    params: dict[str, Any], invalidated: list[str], *, interactive: bool = False
) -> dict[str, Any]:
    return {
        "operation_id": 71,
        "interactive": interactive,
        "params": params,
        "invalidated_on_success": invalidated,
    }


def _result_reply(
    pane: str, names: list[str], params: dict[str, Any]
) -> dict[str, Any]:
    return {
        "summary": {"frequency": 5.0},
        "invalid": [],
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
@pytest.mark.parametrize("has_invalid", [True, False])
def test_finished_analysis_uses_start_facts_without_hidden_pre_reads(
    tmp_path,
    clients,
    monkeypatch,
    stage,
    method,
    result_method,
    pane,
    has_figure,
    has_invalid,
):
    def respond(name, params):
        if name == method:
            assert params == {"tab_id": "t", "updates": {"gain": 2}}
            return _start_reply({"gain": 2, "model": "fit"}, ["post.writeback"])
        if name == "operation.await":
            assert params["operation_id"] == 71
            assert 0 < params["timeout"] <= 0.25
            return {"reason": "completed", "status": "finished"}
        if name == result_method:
            assert params == {"tab_id": "t", "operation_id": 71}
            observed = _result_reply(
                pane,
                ["fit", "residual"] if has_figure else [],
                {"gain": 2, "model": "fit"},
            )
            if has_invalid:
                observed["summary"].update(
                    frequency_error=None, warnings=["singular error"]
                )
                observed["invalid"] = [
                    {"path": "summary.frequency_error", "reason": "non_finite"}
                ]
            return observed
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
    reply = _call_full_execution_stdio(
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
    assert result["result"]["summary"] == (
        {"frequency": 5.0, "frequency_error": None, "warnings": ["singular error"]}
        if has_invalid
        else {"frequency": 5.0}
    )
    assert result["result"]["invalid"] == (
        [{"path": "summary.frequency_error", "reason": "non_finite"}]
        if has_invalid
        else []
    )
    for tool, arguments in [
        ("status", {"execution": result["execution"]}),
        ("wait", {"execution": result["execution"], "timeout": 0}),
    ]:
        observed = _data(
            _call_full_execution_stdio(monkeypatch, client, tool, arguments)
        )
        assert observed["result"] == result["result"]
        assert observed["saved_images"] == result["saved_images"]
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
            return _start_reply({}, [])
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
    reply = _call_full_execution_stdio(monkeypatch, client, "tab_analyze", {"tab": "t"})
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
            return _start_reply({"gain": 2}, [], interactive=True)
        if method == "operation.await":
            return {"reason": "user_feedback", "status": "running"}
        assert method == "tab.interact"
        assert params == {"tab_id": "t"}
        return {**interaction, "operation_id": 71}

    client = _client(tmp_path, clients, respond)
    reply = _call_full_execution_stdio(monkeypatch, client, "tab_analyze", {"tab": "t"})
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
            return _start_reply({"gain": 2}, ["post_analysis"], interactive=True)
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
    started = _data(
        _call_full_execution_stdio(monkeypatch, client, "tab_analyze", {"tab": "t"})
    )
    assert started["status"] == "interactive"
    execution = started["execution"]
    assert not saving.is_set()
    done.set()  # The GUI user, not an MCP command, completes the original operation.
    assert saving.wait(2), "completion stopped when the first tool call returned"
    reply = _call_full_execution_stdio(
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
            return _start_reply({"gain": 2}, [], interactive=True)
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
        _data(
            _call_full_execution_stdio(monkeypatch, client, "tab_analyze", {"tab": "t"})
        )
        if registered
        else None
    )
    reply = _call_full_execution_stdio(
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
        _call_full_execution_stdio(
            monkeypatch, client, "wait", {"execution": execution, "timeout": 0}
        )
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


@pytest.mark.parametrize("bad_image", [False, True])
def test_rejected_done_keeps_registered_interaction_editable(
    tmp_path, clients, monkeypatch, bad_image
):
    value = 1

    def respond(method, params):
        nonlocal value
        if method == "tab.analyze":
            return _start_reply({}, [], interactive=True)
        if method == "operation.await":
            return {"reason": "user_feedback", "status": "running"}
        if method == "tab.interact":
            if "payload" in params:
                assert params["payload"] == {"command": "set", "value": 2}
                value = 2
            return {
                "operation_id": 71,
                "state": {"value": value},
                "commands": [{"name": "set"}, {"name": "done"}],
                "figure": {"png_b64": "invalid"} if value == 2 and bad_image else None,
            }
        raise AssertionError(method)

    client = _client(tmp_path, clients, respond)
    started = _data(
        _call_full_execution_stdio(monkeypatch, client, "tab_analyze", {"tab": "t"})
    )
    execution = started["execution"]
    client.transport.replies["tab.interact"] = {
        "ok": False,
        "error": {"code": "invalid_params", "message": "selection required"},
    }
    rejected = _call_stdio(
        monkeypatch,
        client,
        "tab_interact",
        {"tab": "t", "payload": {"command": "done"}},
    )
    assert rejected["isError"] is True
    assert "selection required" in rejected["content"][0]["text"]
    snapshot = client.call("status", {"execution": execution, "detail": "full"})
    assert snapshot["status"] == "interactive"
    assert snapshot["error"] is None
    assert snapshot["interaction"]["state"] == {"value": 1}
    del client.transport.replies["tab.interact"]
    updated = _call_stdio(
        monkeypatch,
        client,
        "tab_interact",
        {"tab": "t", "payload": {"command": "set", "value": 2}},
    )
    assert bool(updated.get("isError")) is bad_image
    data = json.loads(updated["content"][0]["text"])
    assert data["execution"] == execution
    snapshot = client.call("status", {"execution": execution, "detail": "full"})
    assert snapshot["status"] == "interactive"
    assert snapshot["interaction"]["state"] == {"value": 2}
    assert bool(snapshot["interaction"].get("delivery_error")) is bad_image
    assert snapshot["save_status"] == "not_started"
    assert set(_methods(client)) == {"tab.analyze", "tab.interact", "operation.await"}
    assert _methods(client).count("tab.interact") == 3


@pytest.mark.parametrize("selector", ["op", "execution"])
@pytest.mark.parametrize("outcome", ["finished", "cancelled", "failed"])
@pytest.mark.parametrize("gui_cancel", ["cancelling", "not_cancellable", "failed"])
def test_cancel_latches_intent_and_retains_original_terminal(
    tmp_path, clients, monkeypatch, selector, outcome, gui_cancel
):
    terminal = Event()

    def respond(method, params):
        if method == "tab.analyze":
            return _start_reply({}, [], interactive=True)
        if method == "tab.interact":
            return {"operation_id": 71, "state": {}, "figure": None}
        if method == "operation.await":
            assert params["operation_id"] == 71
            return (
                {"reason": "completed", "status": outcome, "error": "real failure"}
                if terminal.is_set()
                else {"reason": "user_feedback", "status": "running"}
            )
        if method == "operation.cancel":
            assert params["operation_id"] == 71
            return {"status": gui_cancel}
        raise AssertionError(method)

    client = _client(tmp_path, clients, respond)
    started = _data(
        _call_full_execution_stdio(monkeypatch, client, "tab_analyze", {"tab": "t"})
    )
    if gui_cancel in ("failed", "not_cancellable"):
        client.transport.replies["operation.cancel"] = {
            "ok": False,
            "error": {
                "code": "precondition_failed",
                "reason": gui_cancel,
                "message": "cannot send cancel",
            },
        }
    args = {selector: started[selector]}
    cancelled = _call_stdio(monkeypatch, client, "cancel", args)
    assert bool(cancelled.get("isError")) is (gui_cancel == "failed")
    data = json.loads(cancelled["content"][0]["text"])
    assert data["execution"] == started["execution"]
    assert data["cancel_requested"] is True
    assert data["gui_cancel"]["status"] == (
        "requested" if gui_cancel == "cancelling" else gui_cancel
    )
    if gui_cancel == "failed":
        assert "cannot send cancel" in data["gui_cancel"]["error"]["message"]
    repeated = _call_stdio(monkeypatch, client, "cancel", args)
    assert (
        json.loads(repeated["content"][0]["text"])["gui_cancel"] == data["gui_cancel"]
    )
    assert _methods(client).count("operation.cancel") == 1
    terminal.set()
    completed = _data(
        _call_full_execution_stdio(
            monkeypatch,
            client,
            "wait",
            {"execution": started["execution"], "timeout": 2},
        )
    )
    assert completed["status"] == ("failed" if outcome == "failed" else "cancelled")
    assert completed["operation_outcome"]["status"] == outcome
    assert completed["cancel_requested"] is True
    assert completed["result"] is None
    assert completed["save_status"] == "not_started"
    assert set(_methods(client)) <= {
        "tab.analyze",
        "tab.interact",
        "operation.await",
        "operation.cancel",
    }
    frozen = client.call(
        "status", {"execution": started["execution"], "detail": "full"}
    )
    final = _data(_call_stdio(monkeypatch, client, "cancel", args))
    assert final["gui_cancel"]["status"] == "not_needed"
    assert (
        client.call("status", {"execution": started["execution"], "detail": "full"})
        == frozen
    )
    assert _methods(client).count("operation.cancel") == 1


def _admitted_save_terminal(client: MeasureClient, outcome: str) -> dict[str, Any]:
    if outcome == "lost":
        client.transport.close()
        assert client.transport.on_closed is not None
        client.transport.on_closed(EOFError("save reply lost"))
        return {"ok": True, "result": {"image_path": "/unconfirmed/fit.png"}}
    if outcome in _AMBIGUOUS_SAVE_ERRORS:
        return {"ok": False, "error": _AMBIGUOUS_SAVE_ERRORS[outcome]}
    if outcome == "rejected":
        return {
            "ok": False,
            "error": {"code": "precondition_failed", "message": "disk rejected export"},
        }
    return {"ok": True, "result": {"image_path": "/actual/fit.png"}}


@pytest.mark.parametrize("remaining", [False, True])
@pytest.mark.parametrize(
    "save_outcome", ["saved", "rejected", "lost", *_AMBIGUOUS_SAVE_ERRORS]
)
def test_cancel_during_admitted_save_retains_the_real_reply(
    tmp_path, clients, monkeypatch, remaining, save_outcome
):
    saving, release, intent = Event(), Event(), Event()
    read_internal = GuiConnection.read_internal

    def read(connection, method, params, **kwargs):
        if method == "operation.cancel":
            intent.set()
        return read_internal(connection, method, params, **kwargs)

    monkeypatch.setattr(GuiConnection, "read_internal", read)

    def respond(method, params):
        if method == "tab.analyze":
            return _start_reply({}, [])
        if method == "operation.await":
            return {"reason": "completed", "status": "finished"}
        if method == "tab.get_analyze_result":
            return _result_reply(
                "analysis",
                ["prefix", "fit", "residual"] if remaining else ["prefix", "fit"],
                {},
            )
        if method == "operation.cancel":
            return {"status": "finished"}
        raise AssertionError(method)

    client = _client(tmp_path, clients, respond)

    def save(params):
        if params["figure_name"] == "prefix":
            return {"ok": True, "result": {"image_path": "/actual/prefix.png"}}
        assert params["figure_name"] == "fit"
        saving.set()
        assert release.wait(10), "test did not release admitted save"
        return _admitted_save_terminal(client, save_outcome)

    client.transport.replies["tab.save_image"] = save
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            started = _data(
                _call_full_execution_stdio(
                    monkeypatch, client, "tab_analyze", {"tab": "t"}
                )
            )
            assert saving.wait(2)
            request = pool.submit(
                client.call, "cancel", {"execution": started["execution"]}
            )
            assert intent.wait(2)
            pending = client.call(
                "status", {"execution": started["execution"], "detail": "full"}
            )
            assert pending["cancel_requested"] is True
            assert pending["status"] == "running"
            assert pending["unconfirmed_image"] == "fit"
        finally:
            release.set()
        cancelled = request.result(timeout=3)
    assert cancelled.data["cancel_requested"] is True
    completed = _data(
        _call_full_execution_stdio(
            monkeypatch,
            client,
            "wait",
            {"execution": started["execution"], "timeout": 2},
        )
    )
    assert completed["status"] == ("cancelled" if save_outcome == "saved" else "failed")
    assert completed["operation_outcome"]["status"] == "finished"
    assert completed["result"]["summary"] == {"frequency": 5.0}
    confirmed = ["prefix", "fit"] if save_outcome == "saved" else ["prefix"]
    assert completed["saved_images"] == [
        {"figure_name": name, "image_path": f"/actual/{name}.png"} for name in confirmed
    ]
    unknown = save_outcome == "lost" or save_outcome in _AMBIGUOUS_SAVE_ERRORS
    assert completed["save_status"] == (
        ("incomplete" if remaining else "saved")
        if save_outcome == "saved"
        else "unknown"
        if unknown
        else "incomplete"
    )
    assert completed["unconfirmed_image"] == ("fit" if unknown else None)
    assert completed["remaining_images"] == (
        ([] if save_outcome == "saved" else ["fit"])
        + (["residual"] if remaining else [])
    )
    if save_outcome != "saved":
        assert completed["error"]["phase"] == "image_save"
    _assert_ambiguous_save_error(completed, save_outcome)
    assert _methods(client).count("tab.save_image") == 2
    assert "tab.get_figure" not in _methods(client)


def test_cancel_rejects_save_queued_behind_another_rpc(tmp_path, clients, monkeypatch):
    save_ready, blocker_entered, release_blocker = Event(), Event(), Event()
    intent, dispatch_save = Event(), Event()
    send_rpc = GuiConnection.send_gui_rpc
    read_internal = GuiConnection.read_internal

    def send(connection, method, params, *args, **kwargs):
        if method == "tab.save_image":
            save_ready.set()
            assert blocker_entered.wait(10), "test did not occupy the RPC lock"
            dispatch_save.set()
        return send_rpc(connection, method, params, *args, **kwargs)

    def read(connection, method, params, **kwargs):
        if method == "operation.cancel":
            intent.set()
        return read_internal(connection, method, params, **kwargs)

    monkeypatch.setattr(GuiConnection, "send_gui_rpc", send)
    monkeypatch.setattr(GuiConnection, "read_internal", read)

    def respond(method, params):
        if method == "tab.analyze":
            return _start_reply({}, [])
        if method == "operation.await":
            return {"reason": "completed", "status": "finished"}
        if method == "tab.get_analyze_result":
            return _result_reply("analysis", ["fit"], {})
        if method == "project.info":
            blocker_entered.set()
            assert release_blocker.wait(10), "test did not release the RPC lock"
            return {"project": "test"}
        if method == "operation.cancel":
            return {"status": "finished"}
        raise AssertionError(method)

    client = _client(tmp_path, clients, respond)
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            started = _data(
                _call_full_execution_stdio(
                    monkeypatch, client, "tab_analyze", {"tab": "t"}
                )
            )
            assert save_ready.wait(2)
            blocker = pool.submit(
                client.call, "rpc_call", {"method": "project.info", "params": {}}
            )
            assert dispatch_save.wait(2)
            request = pool.submit(client.call, "cancel", {"op": started["op"]})
            assert intent.wait(2)
            pending = client.call(
                "status", {"execution": started["execution"], "detail": "full"}
            )
            assert pending["cancel_requested"] is True
            assert pending["unconfirmed_image"] is None
        finally:
            blocker_entered.set()
            release_blocker.set()
        blocker.result(timeout=3)
        assert request.result(timeout=3).data["cancel_requested"] is True
    completed = _data(
        _call_full_execution_stdio(
            monkeypatch,
            client,
            "wait",
            {"execution": started["execution"], "timeout": 2},
        )
    )
    assert completed["status"] == "cancelled"
    assert completed["saved_images"] == []
    assert completed["remaining_images"] == ["fit"]
    assert completed["save_status"] == "not_started"
    assert "tab.save_image" not in _methods(client)
    assert "tab.get_figure" not in _methods(client)


def _hold_wire_reply(
    client: MeasureClient, monkeypatch: pytest.MonkeyPatch, method: str
) -> Event:
    """Model a GUI request that was dispatched but has not replied."""
    sent = Event()
    send_line = client.transport.send_line

    def send(payload):
        if payload["method"] == method:
            client.transport.sent.append((payload["method"], payload["params"]))
            sent.set()
        else:
            send_line(payload)

    monkeypatch.setattr(client.transport, "send_line", send)
    return sent


@pytest.mark.parametrize("cancel_first", [False, True])
def test_close_drains_pending_save_and_cancel_before_png_cleanup(
    tmp_path, clients, monkeypatch, cancel_first
):
    intent = Event()
    read_internal = GuiConnection.read_internal

    def read(connection, method, params, **kwargs):
        if method == "operation.cancel":
            intent.set()
        return read_internal(connection, method, params, **kwargs)

    monkeypatch.setattr(GuiConnection, "read_internal", read)

    client = _client(tmp_path, clients)
    client.transport.replies.update(
        {
            method: {"ok": True, "result": result}
            for method, result in {
                "tab.analyze": {
                    "operation_id": 71,
                    "interactive": False,
                    "params": {},
                    "invalidated_on_success": [],
                },
                "operation.await": {"reason": "completed", "status": "finished"},
                "tab.get_analyze_result": _result_reply("analysis", ["fit"], {}),
            }.items()
        }
    )
    session = client.context.session
    image = session.write_png(_PNG)
    saving = _hold_wire_reply(client, monkeypatch, "tab.save_image")
    cleanup_pngs = session.cleanup_pngs
    cleanup_states = []

    def cleanup():
        cleanup_states.extend(session.executions.snapshots())
        assert image.exists()
        cleanup_pngs()

    request = None
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            started = _data(
                _call_full_execution_stdio(
                    monkeypatch, client, "tab_analyze", {"tab": "t"}
                )
            )
            assert saving.wait(2)
            if cancel_first:
                request = pool.submit(
                    client.call, "cancel", {"execution": started["execution"]}
                )
                assert intent.wait(2)
            with monkeypatch.context() as patch:
                patch.setattr(session, "cleanup_pngs", cleanup)
                pool.submit(session.close).result(timeout=2)
            if cancel_first:
                assert request is not None
                cancelled = request.result(timeout=2)
                assert cancelled.is_error is True
                assert cancelled.data["gui_cancel"]["status"] == "failed"
                assert cancelled.data["cancel_requested"] is True
        finally:
            client.context.bridge.disconnect()
    assert len(cleanup_states) == 1
    assert cleanup_states[0].status == "failed"
    assert cleanup_states[0].phase == "terminal"
    assert not image.parent.exists()
    completed = client.call(
        "status", {"execution": started["execution"], "detail": "full"}
    )
    assert completed["status"] == "failed"
    assert completed["save_status"] == "unknown"
    assert completed["unconfirmed_image"] == "fit"
    assert completed["error"]["phase"] == "image_save"
    assert completed["cancel_requested"] is cancel_first
    assert "operation.cancel" not in _methods(client)


@pytest.mark.parametrize("interactive", [False, True])
def test_close_after_start_receipt_retains_operation_without_new_work(
    tmp_path, clients, monkeypatch, interactive
):
    send_rpc = GuiConnection.send_gui_rpc

    def respond(method, params):
        assert method == "tab.analyze"
        return _start_reply({}, [], interactive=interactive)

    client = _client(tmp_path, clients, respond)

    def send(connection, method, params, *args, **kwargs):
        reply = send_rpc(connection, method, params, *args, **kwargs)
        if method == "tab.analyze":
            client.context.session.close()
        return reply

    monkeypatch.setattr(GuiConnection, "send_gui_rpc", send)
    reply = _call_full_execution_stdio(monkeypatch, client, "tab_analyze", {"tab": "t"})
    assert reply["isError"] is True
    stopped = json.loads(reply["content"][0]["text"])
    assert stopped["op"] == 1
    assert stopped["tab"] == "t"
    assert stopped["status"] == "failed"
    assert stopped["error"]["reason"] == "session_closed"
    assert stopped["operation_outcome"] is None
    assert stopped["cancel_requested"] is False
    assert stopped["save_status"] == "not_started"
    assert _methods(client) == ["tab.analyze"]
    assert (
        client.call("status", {"execution": stopped["execution"], "detail": "full"})
        == stopped
    )


@pytest.mark.parametrize("interactive", [False, True])
def test_worker_start_failure_keeps_the_admitted_operation_receipt(
    tmp_path, clients, monkeypatch, interactive
):
    start_thread = Thread.start

    def start(thread):
        if thread.name.startswith("analysis-"):
            raise RuntimeError("no worker available")
        start_thread(thread)

    monkeypatch.setattr(Thread, "start", start)

    def respond(method, params):
        assert method == "tab.analyze"
        return _start_reply({}, [], interactive=interactive)

    client = _client(tmp_path, clients, respond)
    reply = _call_full_execution_stdio(monkeypatch, client, "tab_analyze", {"tab": "t"})
    assert reply["isError"] is True
    stopped = json.loads(reply["content"][0]["text"])
    assert stopped["op"] == 1
    assert stopped["status"] == "failed"
    assert stopped["error"]["reason"] == "worker_start_failed"
    assert stopped["operation_outcome"] is None
    assert _methods(client) == ["tab.analyze"]


def test_gui_completion_during_initial_handoff_keeps_the_execution(
    tmp_path, clients, monkeypatch
):
    saved = Event()
    send_rpc = GuiConnection.send_gui_rpc

    def send(connection, method, params, *args, **kwargs):
        if method == "tab.interact":
            assert saved.wait(3), "analysis did not continue before initial read"
        return send_rpc(connection, method, params, *args, **kwargs)

    monkeypatch.setattr(GuiConnection, "send_gui_rpc", send)

    def respond(method, params):
        if method == "tab.analyze":
            return _start_reply({}, [], interactive=True)
        if method == "operation.await":
            return {"reason": "completed", "status": "finished"}
        if method == "tab.get_analyze_result":
            return _result_reply("analysis", ["fit"], {})
        if method == "tab.save_image":
            return {"image_path": "/actual/fit.png"}
        if method == "tab.get_figure":
            saved.set()
            return {"png_b64": base64.b64encode(_PNG).decode()}
        raise AssertionError(method)

    client = _client(tmp_path, clients, respond)
    client.transport.replies["tab.interact"] = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "message": "interactive session already completed",
        },
    }
    started = _data(
        _call_full_execution_stdio(monkeypatch, client, "tab_analyze", {"tab": "t"})
    )
    assert started["op"] == 1
    completed = _data(
        _call_full_execution_stdio(
            monkeypatch,
            client,
            "wait",
            {"execution": started["execution"], "timeout": 2},
        )
    )
    assert completed["status"] == "finished"
    assert completed["saved_images"] == [
        {"figure_name": "fit", "image_path": "/actual/fit.png"}
    ]
    assert _methods(client).count("tab.analyze") == 1
    assert _methods(client).count("tab.save_image") == 1


@pytest.mark.parametrize("tool", ["tab_analyze", "tab_interact"])
def test_malformed_interactive_image_is_a_tool_error_not_a_success(
    tmp_path, clients, monkeypatch, tool
):
    def respond(method, params):
        if method == "tab.analyze":
            return _start_reply({}, [], interactive=True)
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
            return _start_reply({}, [], interactive=True)
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
            return _start_reply({}, [])
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
    reply = _call_full_execution_stdio(
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
            return _start_reply({"model": "fit"}, [])
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
        started = _data(
            _call_full_execution_stdio(monkeypatch, client, "tab_analyze", {"tab": "t"})
        )
        assert awaiting.wait(2)
        assert started["status"] == "running"
        execution = started["execution"]
        running = _data(
            _call_full_execution_stdio(
                monkeypatch, client, "wait", {"execution": execution, "timeout": 0}
            )
        )
        assert running["status"] == "running"
        assert running["cancel_requested"] is False
        assert running["elapsed_s"] >= 0
        snapshot = client.call("status", {"execution": execution, "detail": "full"})
        snapshot["params"]["model"] = "caller mutation"
        assert client.call("status", {"execution": execution, "detail": "full"})[
            "params"
        ] == {"model": "fit"}
        assert _methods(client) == ["tab.analyze", "operation.await"]
        release.set()
        completed_reply = _call_full_execution_stdio(
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
            _call_full_execution_stdio(
                monkeypatch, client, "status", {"execution": execution}
            )
        )
        repeated = _data(
            _call_full_execution_stdio(
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


def _assert_ambiguous_save_error(data: dict[str, Any], failure: str) -> None:
    if failure not in _AMBIGUOUS_SAVE_ERRORS:
        return
    envelope = _AMBIGUOUS_SAVE_ERRORS[failure]
    assert data["error"]["code"] == envelope["code"]
    assert data["error"]["reason"] == envelope.get("reason", "gui_handler_timeout")
    assert envelope["message"] in data["error"]["message"]


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
    elif failure == "save_rejected" or failure in _AMBIGUOUS_SAVE_ERRORS:
        save_error = (
            {"ok": False, "error": _AMBIGUOUS_SAVE_ERRORS[failure]}
            if failure in _AMBIGUOUS_SAVE_ERRORS
            else rejection
        )
        client.transport.replies["tab.save_image"] = lambda params: (
            save_error
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
        ("handler_timeout", "image_save", "unknown", ["fit"], "residual"),
        ("encoding_failed", "image_save", "unknown", ["fit"], "residual"),
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
            return _start_reply({"gain": 2}, [])
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

    reply = _call_full_execution_stdio(monkeypatch, client, "tab_analyze", {"tab": "t"})
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
    _assert_ambiguous_save_error(data, failure)
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
