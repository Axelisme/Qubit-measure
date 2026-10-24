"""Analysis results, interaction, and image delivery through the recording GUI seam."""

import base64
import json
from dataclasses import asdict
from threading import Event, get_ident

import pytest

from ._analyze_support import (
    PNG as _PNG,
)
from ._analyze_support import (
    analysis_client as _client,
)
from ._analyze_support import (
    analysis_result_reply as _result_reply,
)
from ._analyze_support import (
    analysis_start_reply as _start_reply,
)
from ._analyze_support import (
    assert_figure as _assert_figure,
)
from ._analyze_support import (
    call_full_execution_stdio as _call_full_execution_stdio,
)
from ._analyze_support import (
    call_stdio as _call_stdio,
)
from ._analyze_support import (
    sent_methods as _methods,
)
from ._analyze_support import (
    stdio_data as _data,
)
from ._save_report_support import analysis_report


@pytest.mark.parametrize(
    "stage,method,result_method,pane",
    [
        ("primary", "tab.analyze", "tab.get_analyze_result", "analysis"),
        ("post", "tab.post_analyze", "tab.get_post_analyze_result", "post_analysis"),
    ],
)
def test_figures_only_analysis_preserves_empty_summary(
    tmp_path, clients, monkeypatch, stage, method, result_method, pane
):
    def respond(name, params):
        if name == method:
            return {**_start_reply({}, []), "params": None}
        if name == "operation.await":
            return {"reason": "completed", "status": "finished"}
        if name == result_method:
            return {**_result_reply(pane, ["fit"], {}), "summary": {}, "params": None}
        if name == "tab.save_artifacts":
            return asdict(analysis_report(pane, ["fit"]))
        assert name == "tab.get_figure"
        return {"png_b64": base64.b64encode(_PNG).decode()}

    client = _client(tmp_path, clients, respond)
    reply = _call_full_execution_stdio(
        monkeypatch, client, "tab_analyze", {"tab": "t", "stage": stage}
    )
    result = _data(reply)
    assert result["status"] == "finished"
    assert result["result"]["summary"] == {}
    assert result["result"]["params"] is None
    _assert_figure(reply, present=True)


_INVALID_PNGS = [
    pytest.param(b"hello", id="non-png"),
    pytest.param(b"hello" + _PNG[-12:], id="non-png-with-iend"),
    pytest.param(_PNG[:8], id="signature-only"),
    pytest.param(_PNG[:46], id="truncated-pixels"),
    pytest.param(_PNG[:-12], id="missing-iend"),
    pytest.param(_PNG[:-1], id="truncated-iend-crc"),
    pytest.param(_PNG[:55] + bytes([_PNG[55] ^ 1]) + _PNG[56:], id="bad-idat-crc"),
]


@pytest.mark.parametrize("payload", [None, {"command": "set_value", "args": {"x": 2}}])
def test_interact_forwards_once_and_delivers_inline_image(
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
    _assert_figure(reply, present=True)
    assert _methods(client) == ["tab.interact"]


def test_interact_headless_and_wire_failure_do_not_retry(
    tmp_path, clients, monkeypatch
):
    client = _client(
        tmp_path,
        clients,
        lambda method, params: {"operation_id": 71, "state": {}, "figure": None},
    )
    reply = _call_stdio(monkeypatch, client, "tab_interact", {"tab": "t"})
    assert _data(reply) == {"handle": 1, "state": {}}
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
        if name == "tab.save_artifacts":
            assert params == {
                "tab_id": "t",
                "artifacts": [{"section": pane, "name": "results"}],
                "operation_id": 71,
            }
            return asdict(
                analysis_report(pane, ["fit", "residual"] if has_figure else [])
            )
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
        assert observed["save_report"] == result["save_report"]
    summary = client.call("status", {"execution": result["execution"]})
    assert summary["analysis"][stage]["summary"] == result["result"]["summary"]
    assert summary["analysis"][stage]["invalid"] == result["result"]["invalid"]
    assert summary["invalid"] == result["result"]["invalid"]
    assert result["result"]["params"] == {"gain": 2, "model": "fit"}
    assert result["params"] == result["result"]["params"]
    assert result["invalidated"] == ["post.writeback"]
    assert result["operation_outcome"]["status"] == "finished"
    assert result["save_status"] == "saved"
    assert result["save_report"] == asdict(
        analysis_report(pane, ["fit", "residual"] if has_figure else [])
    )
    assert summary["artifacts"] == result["save_report"]["artifacts"]
    assert result["error"] is None
    _assert_figure(reply, present=has_figure)
    assert _methods(client) == [
        method,
        "operation.await",
        result_method,
        "tab.save_artifacts",
        *(["tab.get_figure"] if has_figure else []),
        "tab.writeback_preview",
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
            return {"reason": "timeout", "status": "running"}
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
    summary = client.call("status", {"execution": data["execution"]})
    assert summary["interaction"] == {
        "plugin": "picker",
        "state": {"value": 3},
        "commands": [{"name": "done"}],
        "info": {"label": "fit"},
        "preview_active": False,
        "handle": data["op"],
        "execution": data["execution"],
    }
    assert (
        len(
            _call_stdio(
                monkeypatch, client, "status", {"execution": data["execution"]}
            )["content"]
        )
        == 1
    )
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
                else {"reason": "timeout", "status": "running"}
            )
        if method == "tab.get_analyze_result":
            return _result_reply("analysis", ["fit"], {"gain": 2})
        if method == "tab.save_artifacts":
            saving.set()
            return asdict(analysis_report("analysis", ["fit"]))
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
    # The completion worker reaches Save after the GUI completion is released. The 30s limit only guards deadlock.
    assert saving.wait(30), "completion stopped when the first tool call returned"
    reply = _call_full_execution_stdio(
        # The released worker completes this execution; 30s only guards deadlock.
        monkeypatch,
        client,
        "wait",
        {"execution": execution, "timeout": 30},
    )
    completed = _data(reply)
    assert completed["execution"] == execution
    assert completed["op"] == started["op"]
    assert completed["status"] == "finished"
    assert completed["save_report"] == asdict(analysis_report("analysis", ["fit"]))
    assert completed["invalidated"] == ["post_analysis"]
    _assert_figure(reply, present=True)
    effects = [m for m in _methods(client) if m != "operation.await"]
    assert effects == [
        "tab.analyze",
        "tab.interact",
        "tab.get_analyze_result",
        "tab.save_artifacts",
        "tab.get_figure",
        "tab.writeback_preview",
    ]


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
            return {"reason": "timeout", "status": "running"}
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


@pytest.mark.parametrize("tool", ["tab_analyze", "tab_interact"])
def test_malformed_interactive_image_is_a_tool_error_not_a_success(
    tmp_path, clients, monkeypatch, tool
):
    def respond(method, params):
        if method == "tab.analyze":
            return _start_reply({}, [], interactive=True)
        if method == "operation.await":
            return {"reason": "timeout", "status": "running"}
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
            return {"reason": "timeout", "status": "running"}
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
        if name == "tab.save_artifacts":
            pane = "analysis" if stage == "primary" else "post_analysis"
            return asdict(analysis_report(pane, ["fit"]))
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
    pane = "analysis" if stage == "primary" else "post_analysis"
    assert data["save_report"] == asdict(analysis_report(pane, ["fit"]))
    assert data["save_status"] == "saved"
    assert _methods(client) == [
        method,
        "operation.await",
        result_method,
        "tab.save_artifacts",
        "tab.get_figure",
    ]


@pytest.mark.parametrize(
    "stage,method,result_method,pane",
    [
        ("primary", "tab.analyze", "tab.get_analyze_result", "analysis"),
        ("post", "tab.post_analyze", "tab.get_post_analyze_result", "post_analysis"),
    ],
)
@pytest.mark.parametrize("outcome", ["finished", "cancelled", "failed"])
def test_analysis_can_complete_on_recipe_worker(
    tmp_path, clients, stage, method, result_method, pane, outcome
):
    caller = get_ident()
    await_callers: list[int] = []

    def respond(name, params):
        if name == method:
            return _start_reply({"gain": 2}, [])
        if name == "operation.await":
            await_callers.append(get_ident())
            assert params["operation_id"] == 71
            return {"reason": "completed", "status": outcome, "error": "fit failed"}
        if name == "tab.save_artifacts":
            assert params["artifacts"] == [{"section": pane, "name": "results"}]
            return asdict(analysis_report(pane))
        assert name == result_method
        return _result_reply(pane, [], {"gain": 2})

    client = _client(tmp_path, clients, respond)
    connection = client.context.bound().gui
    started = connection.send_gui_rpc(method, {"tab_id": "t", "updates": {"gain": 2}})
    execution = client.context.session.executions.start(
        connection, "t", stage, started, start_worker=False
    )
    assert execution.snapshot().status == "running"
    assert _methods(client) == [method]

    snapshot = execution.complete_in_current_worker()
    assert snapshot.status == outcome
    assert snapshot.phase == "terminal"
    assert await_callers == [caller]
    assert (
        client.context.session.executions.get(snapshot.execution).snapshot() == snapshot
    )
    if outcome == "finished":
        assert snapshot.result is not None
        assert snapshot.result.summary == {"frequency": 5.0}
        assert snapshot.writeback == {
            "has_draft": False,
            "items": [],
            "destination_context": {},
        }
        assert _methods(client) == [
            method,
            "operation.await",
            result_method,
            "tab.save_artifacts",
            "tab.writeback_preview",
        ]
    else:
        assert snapshot.result is None
        assert _methods(client) == [method, "operation.await"]
    with pytest.raises(ValueError, match="completion already started"):
        execution.complete_in_current_worker()
