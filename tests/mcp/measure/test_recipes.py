"""Recipe promises through the shipped tool table and GUI transport seam."""

import base64
from copy import deepcopy
from pathlib import Path
from threading import Event, Thread
from time import sleep
from typing import Any

import pytest
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.measure import tools_recipes
from zcu_tools.mcp.measure.session import GuiRpcError, MeasureMcpSession

from ._support import make_client

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _scalar(value: object) -> dict[str, Any]:
    return {
        "kind": "scalar",
        "valid": True,
        "input": {
            "mode": "direct",
            "raw": value,
            "resolved": value,
            "error": None,
            "validation_error": None,
        },
    }


def _section(**children: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "section", "valid": True, "children": children}


class LookbackGui:
    """GUI collaborator with distinct Run, save and analysis wire identities."""

    def __init__(self):
        self.publication: dict[str, Any] = {
            "cfg_ref": {"cfg_id": "cfg", "revision": "2"},
            "status": "Valid",
            "source_basis": [],
            "diagnostics": [],
            "tree": _section(
                rounds=_scalar(3),
                modules=_section(
                    reset={"kind": "reference", "valid": True, "ref": None},
                    init_pulse={"kind": "reference", "valid": True, "ref": None},
                    readout={
                        "kind": "reference",
                        "valid": True,
                        "ref": None,
                        "error": None,
                        "children": {
                            "pulse_cfg": _section(freq=_scalar(5000.0)),
                            "ro_cfg": _section(
                                ro_freq=_scalar(5000.0),
                                ro_length=_scalar(2.0),
                                trig_offset=_scalar(0.1),
                            ),
                        },
                    },
                ),
            ),
        }
        self.ran = False
        self.raw_saved = False
        self.md: dict[str, Any] = {}

    def _observations(self) -> dict[str, dict[str, Any]]:
        return {
            "context.snapshot": {
                "label": "sample",
                "md": self.md,
                "ml": {"modules": {}, "waveforms": {}},
            },
            "tab.new": {"tab_id": "t"},
            "soc.info": {"connected": True, "cfg": {}},
            "device.list": {"devices": [{"name": "bias"}]},
            "device.snapshot": {"snapshot": {"name": "bias", "info": {"value": 0.0}}},
            "tab.snapshot": {
                "tabs": [
                    {
                        "tab_id": "t",
                        "adapter_name": "lookback",
                        "interaction": {
                            "is_running": False,
                            "is_analyzing": False,
                            "is_saving_data": False,
                        },
                        "result_state": {
                            "available": self.ran,
                            "revision": 1,
                            "source_operation_id": 71 if self.ran else None,
                        },
                    }
                ]
            },
        }

    def _edit(self, params: dict[str, Any]) -> None:
        assert params["expected"] == self.publication["cfg_ref"]
        for edit in params["edits"]:
            node = self.publication["tree"]
            for part in edit["path"]:
                node = node["children"][part]
            if node["kind"] == "reference":
                node["ref"] = edit["value"].get("__ref") if edit["value"] else None
            else:
                value = edit["value"]
                if isinstance(value, dict) and "__expr" in value:
                    node.update(_scalar(self.md.get(value["__expr"])))
                    node["input"].update(mode="expression", raw=value["__expr"])
                else:
                    node.update(_scalar(value))
        revision = int(self.publication["cfg_ref"]["revision"]) + 1
        self.publication["cfg_ref"]["revision"] = str(revision)

    def __call__(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        observations = self._observations()
        if method in observations:
            expected = {
                "tab.new": {"adapter_name": "lookback"},
                "soc.info": {"include_cfg": True},
                "device.snapshot": {"name": "bias"},
            }
            if method in expected:
                assert params == expected[method]
            return deepcopy(observations[method])
        if method in ("tab.get_cfg", "tab.edit_cfg", "tab.reset_cfg"):
            if method != "tab.get_cfg":
                self._edit(
                    params if method == "tab.edit_cfg" else {**params, "edits": []}
                )
            return deepcopy(self.publication)
        if method == "tab.run_start":
            assert params == {"tab_id": "t", "expected": self.publication["cfg_ref"]}
            assert not self.ran
            self.ran = True
            return {"operation_id": 71}
        if method == "tab.save_data":
            assert params == {"tab_id": "t", "run_operation_id": 71}
            return {"operation_id": 82, "data_path": "/actual/raw.h5"}
        if method == "operation.await":
            assert 0 < params["timeout"] <= 0.25
            assert params["operation_id"] in (71, 82, 93)
            if params["operation_id"] == 82:
                self.raw_saved = True
            return {"reason": "completed", "status": "finished"}
        return self._analysis(method, params)

    def _analysis(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "tab.analyze":
            assert self.raw_saved
            assert params == {"tab_id": "t", "updates": {}, "run_operation_id": 71}
            return {
                "operation_id": 93,
                "interactive": False,
                "params": {"threshold": 0.5},
                "invalidated_on_success": [],
            }
        if method == "tab.get_analyze_result":
            assert params == {"tab_id": "t", "operation_id": 93}
            return {
                "summary": {"offset": 0.24},
                "params": {"threshold": 0.5},
                "operation_state": {"analysis_state": {"figure_names": ["trace"]}},
            }
        if method == "tab.save_image":
            assert params["operation_id"] == 93
            assert params["figure_name"] == "trace"
            return {"image_path": "/actual/trace.png"}
        if method == "tab.get_figure":
            assert params["operation_id"] == 93
            return {"png_b64": base64.b64encode(_PNG).decode()}
        if method == "tab.writeback_preview":
            assert params == {
                "tab_id": "t",
                "subtab_id": "analysis",
                "operation_id": 93,
            }
            return {
                "has_draft": True,
                "items": [{"id": "md-1", "proposed": 0.24}],
                "destination_context": {"active_label": "sample"},
            }
        raise AssertionError(method)


def test_lookback_initial_wait_returns_while_the_same_execution_continues(
    tmp_path, monkeypatch
):
    gui = LookbackGui()
    run_waiting = Event()
    release_run = Event()
    returned = Event()
    replies: list[ToolReply] = []

    def respond(method, params):
        if method == "operation.await" and params["operation_id"] == 71:
            run_waiting.set()
            if not release_run.wait(0.02):
                return {"reason": "timeout"}
        return gui(method, params)

    client = make_client(tmp_path, respond)
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)

    def call_recipe():
        replies.append(client.call("lookback", {"frequency_mhz": 6020.0}))
        returned.set()

    caller = Thread(target=call_recipe)
    caller.start()
    try:
        assert run_waiting.wait(1)
        assert returned.wait(1), "Initial wait must not wait for Run completion"
        initial = replies[0].data
        assert initial["status"] == "running"
        execution = initial["execution"]
        initial["actual"]["fields"].clear()
        before = len(client.transport.sent)
        status = client.call("status", {"execution": execution})
        waiting = client.call("wait", {"execution": execution, "timeout": 0})
        assert status["actual"]["fields"]
        assert waiting.data["execution"] == execution
        assert waiting.data["status"] == "running"
        assert len(client.transport.sent) == before
        release_run.set()
        completed = client.call("wait", {"execution": execution, "timeout": 2})
        assert completed.data["status"] == "finished", completed.data
        assert completed.data["run_op"] == initial["run_op"]
        assert completed.data["raw_save"]["path"] == "/actual/raw.h5"
        assert completed.images[0].data == _PNG
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == 1
        assert methods.count("tab.analyze") == 1
    finally:
        release_run.set()
        caller.join(2)
        client.context.session.close()
    assert not caller.is_alive()


@pytest.fixture
def interactive_recipe(tmp_path, handoff_failure):
    gui = LookbackGui()
    done = Event()
    writeback_read = Event()
    initial_handoff = True

    def respond(method, params):
        nonlocal initial_handoff
        if method == "tab.analyze":
            return {**gui(method, params), "interactive": True}
        if (
            method == "operation.await"
            and params["operation_id"] == 93
            and not done.is_set()
        ):
            return {"reason": "user_feedback"}
        if method == "tab.interact":
            assert params["tab_id"] == "t"
            failure = handoff_failure if initial_handoff else None
            initial_handoff = False
            if params.get("payload", {}).get("command") == "done":
                done.set()
            return {
                "operation_id": 94 if failure == "replaced" else 93,
                "figure": {
                    "png_b64": "invalid"
                    if failure == "png"
                    else base64.b64encode(_PNG).decode()
                },
                "state": {"offset": 0.24},
                "commands": [{"name": "done"}],
                "prompt": "Confirm offset",
            }
        if method == "tab.writeback_preview":
            writeback_read.set()
        return gui(method, params)

    client = make_client(tmp_path, respond)
    if handoff_failure == "query":

        def reject_initial_query(params):
            nonlocal initial_handoff
            initial_handoff = False
            del client.transport.replies["tab.interact"]
            return {
                "ok": False,
                "error": {
                    "code": "precondition_failed",
                    "reason": "render_failed",
                    "message": "Preview unavailable",
                },
            }

        client.transport.replies["tab.interact"] = reject_initial_query
    try:
        yield client, writeback_read
    finally:
        done.set()
        client.context.session.close()


@pytest.mark.parametrize("handoff_failure", [None, "png", "query", "replaced"])
def test_lookback_interaction_handoff_keeps_the_original_pipeline_alive(
    interactive_recipe, handoff_failure
):
    client, writeback_read = interactive_recipe
    handoff = client.call("lookback", {"frequency_mhz": 6020.0})
    assert handoff.data["status"] == "interactive"
    assert handoff.data["raw_save"]["path"] == "/actual/raw.h5"
    assert handoff.data["analysis"]["op"] == handoff.data["op"]
    interaction = handoff.data["analysis"]["interaction"]
    assert interaction is not None
    if handoff_failure:
        assert handoff.is_error
        assert interaction["delivery_error"]
        assert interaction["figure"] is None
        assert not handoff.images
    else:
        assert not handoff.is_error
        assert handoff.images[0].data == _PNG
        assert Path(interaction["figure"]).read_bytes() == _PNG
    if handoff_failure not in ("query", "replaced"):
        assert interaction["state"] == {"offset": 0.24}
        assert interaction["commands"] == [{"name": "done"}]
    analysis_execution = handoff.data["analysis"]["execution"]
    execution = handoff.data["execution"]
    read = client.call("tab_interact", {"tab": "t"})
    assert read.data["prompt"] == "Confirm offset"
    assert read.images[0].data == _PNG
    finished_analysis = client.call(
        "tab_interact", {"tab": "t", "payload": {"command": "done"}}
    )
    assert finished_analysis.data["status"] == "finished"
    assert writeback_read.wait(2), "Recipe must resume after interactive analysis"
    finished = client.call("wait", {"execution": execution, "timeout": 2})
    assert finished.data["status"] == "finished", finished.data
    assert finished.data["analysis"]["execution"] == analysis_execution
    assert read.data["execution"] == analysis_execution
    assert finished.data["writeback"]["has_draft"]
    assert finished.images[0].data == _PNG
    methods = [method for method, _ in client.transport.sent]
    assert methods.count("tab.run_start") == methods.count("tab.analyze") == 1


@pytest.mark.parametrize("held_op", [71, 82, 93])
def test_session_close_drains_recipe_work_and_rejects_new_admission(
    tmp_path, monkeypatch, held_op
):
    gui = LookbackGui()
    pending = Event()
    disconnected = Event()

    def respond(method, params):
        if method == "operation.await" and params["operation_id"] == held_op:
            pending.set()
            assert disconnected.wait(2), "Session must disconnect before joining"
        return gui(method, params)

    client = make_client(tmp_path, respond)
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    close_transport = client.transport.close

    def disconnect():
        close_transport()
        disconnected.set()

    monkeypatch.setattr(client.transport, "close", disconnect)
    try:
        initial = client.call("lookback", {"frequency_mhz": 6020.0})
        assert pending.wait(1)
        client.context.session.close()
        before = len(client.transport.sent)
        result = client.call("status", {"execution": initial.data["execution"]})
        assert result["status"] == "failed", result
        assert result["phase"] == "terminal"
        assert result["error"]["reason"] in ("session_closed", "connection_lost")
        if held_op == 82:
            assert result["raw_save"]["status"] == "unknown"
            assert result["raw_save"]["reserved_path"] == "/actual/raw.h5"
            assert result["raw_save"]["path"] is None
        with pytest.raises(GuiRpcError, match="closed"):
            client.call("lookback", {"frequency_mhz": 6020.0})
        client.context.session.close()
        assert len(client.transport.sent) == before
    finally:
        disconnected.set()
        client.context.session.close()


def test_lookback_cancel_latches_one_control_and_stops_after_original_run(
    tmp_path, monkeypatch
):
    gui = LookbackGui()
    stopped = Event()

    def respond(method, params):
        if method == "operation.await" and params["operation_id"] == 71:
            if stopped.is_set():
                return {"reason": "completed", "status": "cancelled"}
            return {"reason": "timeout"}
        if method == "operation.cancel":
            assert params == {"operation_id": 71}
            stopped.set()
            return {"status": "cancelling"}
        return gui(method, params)

    client = make_client(tmp_path, respond)
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    try:
        initial = client.call("lookback", {"frequency_mhz": 6020.0})
        execution = initial.data["execution"]
        cancelled = client.call("cancel", {"execution": execution})
        assert cancelled.data["cancel_requested"]
        assert cancelled.data["execution"] == execution
        client.call("cancel", {"op": initial.data["run_op"]})
        terminal = client.call("wait", {"execution": execution, "timeout": 2})
        assert terminal.data["status"] == "cancelled"
        assert terminal.data["run_outcome"]["status"] == "cancelled"
        before = client.call("status", {"execution": execution})
        client.call("cancel", {"execution": execution})
        assert client.call("status", {"execution": execution}) == before
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("operation.cancel") == 1
        assert methods.count("tab.run_start") == 1
        assert "tab.save_data" not in methods
        assert "tab.analyze" not in methods
    finally:
        stopped.set()
        client.context.session.close()


@pytest.mark.parametrize(
    ("partial_available", "cancel_after", "expected"),
    [(True, False, "finished"), (False, False, "failed"), (True, True, "cancelled")],
)
def test_lookback_finish_early_uses_partial_data_unless_cancel_wins(
    tmp_path, monkeypatch, partial_available, cancel_after, expected
):
    gui = LookbackGui()
    release_run = Event()

    def respond(method, params):
        if method == "operation.await" and params["operation_id"] == 71:
            return (
                {"reason": "completed", "status": "cancelled"}
                if release_run.is_set()
                else {"reason": "timeout"}
            )
        if method == "operation.cancel":
            assert params == {"operation_id": 71}
            return {"status": "cancelling"}
        reply = gui(method, params)
        if method == "tab.snapshot" and gui.ran:
            reply["tabs"][0]["result_state"]["available"] = partial_available
        return reply

    client = make_client(tmp_path, respond)
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    try:
        initial = client.call("lookback", {"frequency_mhz": 6020.0})
        execution = initial.data["execution"]
        early = client.call("finish_early", {"op": initial.data["run_op"]})
        assert early.data["finish_early_requested"]
        assert not early.data["cancel_requested"]
        client.call("finish_early", {"execution": execution})
        if cancel_after:
            client.call("cancel", {"execution": execution})
        release_run.set()
        result = client.call("wait", {"execution": execution, "timeout": 2})
        assert result.data["status"] == expected, result.data
        assert result.data["run_outcome"]["status"] == "cancelled"
        if expected == "finished":
            assert result.data["raw_save"]["path"] == "/actual/raw.h5"
            assert result.data["analysis"]["status"] == "finished"
        elif expected == "failed":
            assert result.data["error"]["reason"] == "run_result_unavailable"
            assert result.data["raw_save"]["status"] == "not_started"
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("operation.cancel") == 1
        assert methods.count("tab.run_start") == 1
        assert methods.count("tab.save_data") == int(expected == "finished")
        assert methods.count("tab.analyze") == int(expected == "finished")
    finally:
        release_run.set()
        client.context.session.close()


@pytest.mark.parametrize("save_succeeds", [True, False])
def test_lookback_cancel_during_raw_save_waits_for_the_true_save_outcome(
    tmp_path, monkeypatch, save_succeeds
):
    gui = LookbackGui()
    saving = Event()
    release_save = Event()

    def respond(method, params):
        if method == "operation.await" and params["operation_id"] == 82:
            saving.set()
            if not release_save.is_set():
                return {"reason": "timeout"}
            if not save_succeeds:
                return {"reason": "completed", "status": "failed", "error": "disk full"}
        return gui(method, params)

    client = make_client(tmp_path, respond)
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    try:
        initial = client.call("lookback", {"frequency_mhz": 6020.0})
        assert saving.wait(1)
        execution = initial.data["execution"]
        early = client.call("finish_early", {"execution": execution})
        assert early.data["status"] == "not_applicable"
        cancelled = client.call("cancel", {"execution": execution})
        assert cancelled.data["gui_cancel"]["status"] == "not_cancellable"
        assert cancelled.data["cancel_requested"]
        assert cancelled.data["raw_save"]["status"] == "saving"
        assert cancelled.data["raw_save"]["path"] is None
        release_save.set()
        final = client.call("wait", {"execution": execution, "timeout": 2})
        assert not final.is_error  # Successful query, even when execution failed.
        assert final.data["status"] == ("cancelled" if save_succeeds else "failed")
        assert final.data["raw_save"]["status"] == (
            "saved" if save_succeeds else "failed"
        )
        assert final.data["raw_save"]["path"] == (
            "/actual/raw.h5" if save_succeeds else None
        )
        if not save_succeeds:
            assert final.data["error"]["reason"] == "raw_save_failed"
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.save_data") == 1
        assert "operation.cancel" not in methods
        assert "tab.analyze" not in methods
    finally:
        release_save.set()
        client.context.session.close()


@pytest.mark.parametrize("outcome", ["cancelled", "failed"])
def test_recipe_cancel_before_initial_handoff_joins_the_true_analysis_outcome(
    tmp_path, monkeypatch, outcome
):
    gui = LookbackGui()
    handoff_waiting = Event()
    release_handoff = Event()
    allow_terminal = Event()
    original_send = MeasureMcpSession.GuiConnection.send_gui_rpc

    def delay_handoff(self, method, params, *args, **kwargs):
        # Schedule the public connection call before its real admission check.
        if method == "tab.interact":
            handoff_waiting.set()
            assert release_handoff.wait(2)
        return original_send(self, method, params, *args, **kwargs)

    def respond(method, params):
        if method == "tab.analyze":
            return {**gui(method, params), "interactive": True}
        if method == "operation.await" and params["operation_id"] == 93:
            if allow_terminal.is_set():
                return {
                    "reason": "completed",
                    "status": outcome,
                    "error": "Analysis failed after cancellation",
                }
            return {"reason": "user_feedback"}
        if method == "operation.cancel":
            assert params == {"operation_id": 93}
            return {"status": "cancelling"}
        return gui(method, params)

    client = make_client(tmp_path, respond)
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    monkeypatch.setattr(
        MeasureMcpSession.GuiConnection, "send_gui_rpc", delay_handoff
    )
    try:
        initial = client.call("lookback", {"frequency_mhz": 6020.0})
        assert handoff_waiting.wait(1)
        execution = initial.data["execution"]
        analysis_receipt, = client.context.session.executions.snapshots()
        cancelled = client.call("cancel", {"execution": execution})
        assert cancelled.data["cancel_requested"]
        assert cancelled.data["gui_cancel"]["status"] == "requested"
        release_handoff.set()
        allow_terminal.set()
        for _ in range(100):
            terminal = client.call("wait", {"execution": execution, "timeout": 0.01})
            if terminal.data["phase"] == "terminal":
                break
            sleep(0.01)
        assert terminal.data["status"] == outcome
        assert terminal.data["analysis"]["execution"] == analysis_receipt.execution
        assert terminal.data["analysis"]["op"] == analysis_receipt.op
        assert terminal.data["analysis"]["status"] == outcome
        assert terminal.data["analysis"]["cancel_requested"]
        assert terminal.data["analysis"]["operation_outcome"]["status"] == outcome
        assert terminal.data["raw_save"]["path"] == "/actual/raw.h5"
        if outcome == "failed":
            assert terminal.data["error"]["reason"] == "analysis_failed"
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == 1
        assert methods.count("tab.analyze") == 1
        assert methods.count("operation.cancel") == 1
        assert not {
            "tab.interact", "tab.get_analyze_result", "tab.save_image",
            "tab.get_figure", "tab.writeback_preview",
        }.intersection(methods)
    finally:
        release_handoff.set()
        allow_terminal.set()
        client.context.session.close()


@pytest.mark.parametrize("selector", ["execution", "run_op", "op"])
def test_recipe_cancel_delegates_to_the_existing_analysis_owner(
    tmp_path, monkeypatch, selector
):
    gui = LookbackGui()
    analyzing = Event()
    stopped = Event()

    def respond(method, params):
        if method == "operation.await" and params["operation_id"] == 93:
            analyzing.set()
            return (
                {"reason": "completed", "status": "cancelled"}
                if stopped.is_set()
                else {"reason": "timeout"}
            )
        if method == "operation.cancel":
            assert params == {"operation_id": 93}
            stopped.set()
            return {"status": "cancelling"}
        return gui(method, params)

    client = make_client(tmp_path, respond)
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    try:
        initial = client.call("lookback", {"frequency_mhz": 6020.0})
        assert analyzing.wait(1)
        execution = initial.data["execution"]
        state = client.call("status", {"execution": execution})
        key = "execution" if selector == "execution" else "op"
        arguments = {key: state[selector]}
        early = client.call("finish_early", arguments)
        assert early.data["status"] == "not_applicable"
        assert not stopped.is_set()
        cancelled = client.call("cancel", arguments)
        assert cancelled.data["cancel_requested"]
        assert cancelled.data["gui_cancel"]["status"] == "requested"
        client.call("cancel", {"execution": execution})
        terminal = client.call("wait", {"execution": execution, "timeout": 2})
        assert terminal.data["status"] == "cancelled"
        assert terminal.data["analysis"]["status"] == "cancelled"
        assert terminal.data["analysis"]["cancel_requested"]
        assert terminal.data["raw_save"]["path"] == "/actual/raw.h5"
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("operation.cancel") == 1
        assert methods.count("tab.analyze") == 1
        assert "tab.writeback_preview" not in methods
    finally:
        stopped.set()
        client.context.session.close()


@pytest.mark.parametrize(
    ("pending_method", "forbidden_method"),
    [
        ("tab.snapshot", "tab.save_data"),
        ("operation.await", "tab.analyze"),
        ("tab.get_figure", "tab.writeback_preview"),
    ],
)
def test_recipe_cancel_blocks_the_next_phase_while_an_admitted_read_finishes(
    tmp_path, monkeypatch, pending_method, forbidden_method
):
    gui = LookbackGui()
    pending = Event()
    release = Event()
    control_replies = []

    def respond(method, params):
        at_boundary = method == pending_method and (
            (method == "tab.snapshot" and gui.ran)
            or (method == "operation.await" and params["operation_id"] == 82)
            or method == "tab.get_figure"
        )
        if at_boundary:
            pending.set()
            assert release.wait(2)
        if method == "operation.cancel":
            return {"status": "finished"}
        return gui(method, params)

    client = make_client(tmp_path, respond)
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    controller = None
    try:
        initial = client.call("lookback", {"frequency_mhz": 6020.0})
        assert pending.wait(1)
        execution = initial.data["execution"]
        controller = Thread(
            target=lambda: control_replies.append(
                client.call("cancel", {"execution": execution})
            )
        )
        controller.start()
        state = client.call("status", {"execution": execution})
        for _ in range(100):
            state = client.call("status", {"execution": execution})
            if state["cancel_requested"]:
                break
            release.wait(0.01)
        assert state["cancel_requested"]
        assert forbidden_method not in [method for method, _ in client.transport.sent]
        release.set()
        controller.join(2)
        assert not controller.is_alive()
        assert control_replies[0].data["cancel_requested"]
        terminal = client.call("wait", {"execution": execution, "timeout": 2})
        assert terminal.data["status"] == "cancelled"
        methods = [method for method, _ in client.transport.sent]
        assert forbidden_method not in methods
        assert methods.count("tab.run_start") == 1
        if pending_method != "tab.snapshot":
            assert terminal.data["raw_save"]["path"] == "/actual/raw.h5"
    finally:
        release.set()
        if controller is not None:
            controller.join(2)
        client.context.session.close()


@pytest.mark.parametrize(
    ("method_pending", "wire_op", "control"),
    [
        ("tab.run_start", 71, "cancel"),
        ("tab.run_start", 71, "finish_early"),
        ("tab.analyze", 93, "cancel"),
    ],
)
def test_recipe_control_reaches_the_original_operation_after_a_late_receipt(
    tmp_path, monkeypatch, method_pending, wire_op, control
):
    gui = LookbackGui()
    pending = Event()
    release = Event()
    stopped = Event()

    def respond(method, params):
        if method == method_pending:
            pending.set()
            assert release.wait(2)
        if method == "operation.cancel":
            assert params == {"operation_id": wire_op}
            stopped.set()
            return {"status": "cancelling"}
        if method == "operation.await" and params["operation_id"] == wire_op:
            return (
                {"reason": "completed", "status": "cancelled"}
                if stopped.is_set()
                else {"reason": "timeout"}
            )
        return gui(method, params)

    client = make_client(tmp_path, respond)
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    try:
        initial = client.call("lookback", {"frequency_mhz": 6020.0})
        assert pending.wait(1)
        execution = initial.data["execution"]
        for _ in range(2):
            reply = client.call(control, {"execution": execution})
            assert reply.data[f"{control}_requested"]
        assert not stopped.is_set()
        release.set()
        terminal = client.call("wait", {"execution": execution, "timeout": 2})
        assert stopped.is_set()
        assert terminal.data["status"] == (
            "finished" if control == "finish_early" else "cancelled"
        )
        methods = [method for method, _ in client.transport.sent]
        assert methods.count(method_pending) == 1
        assert methods.count("operation.cancel") == 1
        if method_pending == "tab.analyze":
            assert terminal.data["analysis"]["cancel_requested"]
            assert terminal.data["raw_save"]["path"] == "/actual/raw.h5"
            assert "tab.get_analyze_result" not in methods
        elif control == "cancel":
            assert "tab.save_data" not in methods
        else:
            assert terminal.data["raw_save"]["path"] == "/actual/raw.h5"
    finally:
        release.set()
        stopped.set()
        client.context.session.close()


def test_recipe_cancel_during_preparation_wins_over_a_missing_parameter_handoff(
    tmp_path, monkeypatch
):
    gui = LookbackGui()
    pending = Event()
    release = Event()

    def respond(method, params):
        if method == "tab.edit_cfg":
            pending.set()
            assert release.wait(2)
        return gui(method, params)

    client = make_client(tmp_path, respond)
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    try:
        initial = client.call("lookback", {})
        assert pending.wait(1)
        execution = initial.data["execution"]
        cancelled = client.call("cancel", {"execution": execution})
        assert cancelled.data["cancel_requested"]
        release.set()
        terminal = client.call("wait", {"execution": execution, "timeout": 2})
        assert terminal.data["status"] == "cancelled"
        assert terminal.data["cancel_requested"]
        assert terminal.data["tab"] == "t"
        assert terminal.data["missing"] == []
        assert not gui.ran
    finally:
        release.set()
        client.context.session.close()


def test_recipe_worker_start_failure_returns_a_terminal_receipt_and_closes_safely(
    tmp_path, monkeypatch
):
    def fail_start(self):
        raise RuntimeError("no thread resources")

    client = make_client(tmp_path, LookbackGui())
    monkeypatch.setattr(Thread, "start", fail_start)
    try:
        reply = client.call("lookback", {"frequency_mhz": 6020.0})
        assert reply.is_error
        assert reply.data["status"] == "failed"
        assert reply.data["error"]["phase"] == "preparing"
        assert reply.data["error"]["reason"] == "worker_start_failed"
        assert reply.data["error"]["message"] == "no thread resources"
        execution = reply.data["execution"]
        state = client.call("status", {"execution": execution})
        assert state["status"] == "failed"
        cancelled = client.call("cancel", {"execution": execution})
        assert cancelled.data["status"] == "failed"
        assert not cancelled.data["cancel_requested"]
        assert "tab.new" not in [method for method, _ in client.transport.sent]
    finally:
        client.context.session.close()


def test_recipe_cancel_during_admitted_writeback_preserves_result_and_intent(
    tmp_path, monkeypatch
):
    gui = LookbackGui()
    reading = Event()
    release = Event()

    def respond(method, params):
        if method == "tab.writeback_preview":
            reading.set()
            assert release.wait(2)
        return gui(method, params)

    client = make_client(tmp_path, respond)
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    try:
        initial = client.call("lookback", {"frequency_mhz": 6020.0})
        assert reading.wait(1)
        execution = initial.data["execution"]
        cancelled = client.call("cancel", {"execution": execution})
        assert cancelled.data["cancel_requested"]
        release.set()
        terminal = client.call("wait", {"execution": execution, "timeout": 2})
        assert terminal.data["status"] == "cancelled"
        assert terminal.data["cancel_requested"]
        assert terminal.data["writeback"]["items"][0]["proposed"] == 0.24
        assert terminal.data["raw_save"]["path"] == "/actual/raw.h5"
        assert terminal.data["analysis"]["status"] == "finished"
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.writeback_preview") == 1
        assert "operation.cancel" not in methods
    finally:
        release.set()
        client.context.session.close()


@pytest.mark.parametrize("replacement_source", [99, None])
@pytest.mark.parametrize("available", [False, True])
def test_lookback_rejects_a_post_run_snapshot_from_another_source(
    tmp_path, replacement_source, available
):
    gui = LookbackGui()

    def respond(method, params):
        reply = gui(method, params)
        if method == "tab.snapshot" and gui.ran:
            reply["tabs"][0]["result_state"] = {
                "available": available,
                "revision": 99,
                "source_operation_id": replacement_source,
            }
        return reply

    client = make_client(tmp_path, respond)
    try:
        reply = client.call("lookback", {"frequency_mhz": 6020.0})
        assert reply.is_error
        assert reply.data["status"] == "failed"
        assert reply.data["error"]["reason"] == "result_superseded"
        assert reply.data["run_outcome"]["status"] == "finished"
        assert reply.data["result_state"] is None
        assert reply.data["raw_save"]["status"] == "not_started"
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == 1
        assert "tab.save_data" not in methods
        assert "tab.analyze" not in methods
    finally:
        client.context.session.close()


@pytest.mark.parametrize("reuse", [False, True])
def test_lookback_saves_original_run_then_analysis_and_delivers_complete_reply(
    tmp_path,
    reuse,
):
    gui = LookbackGui()
    client = make_client(tmp_path, gui)
    try:
        reply = client.call(
            "lookback",
            {
                "reuse_tab_id": "t" if reuse else None,
                "frequency_mhz": 6020.0,
                "readout_length_us": 4.0,
                "trigger_offset_us": 0.2,
                "rounds": 7,
            },
        )
        assert isinstance(reply, ToolReply)
        data = reply.data
        assert data["status"] == "finished", data
        assert not reply.is_error
        assert data["tab"] == "t"
        assert data["run_op"] != 71
        assert data["run_outcome"]["status"] == "finished"
        assert data["raw_save"]["path"] == "/actual/raw.h5"
        assert data["analysis"]["result"]["summary"] == {"offset": 0.24}
        assert data["analysis"]["saved_images"] == [
            {"figure_name": "trace", "image_path": "/actual/trace.png"}
        ]
        assert data["writeback"]["items"] == [{"id": "md-1", "proposed": 0.24}]
        assert reply.images[0].data == _PNG
        assert data["elapsed_s"] >= 0
        actual = data["actual"]
        assert actual["cfg_ref"] == gui.publication["cfg_ref"]
        assert actual["fields"]["modules.readout.pulse_cfg.freq"]["value"] == 6020.0
        assert actual["fields"]["modules.readout.ro_cfg.ro_freq"]["value"] == 6020.0
        assert actual["fields"]["modules.readout.ro_cfg.ro_length"]["value"] == 4.0
        assert actual["fields"]["modules.readout.ro_cfg.trig_offset"]["value"] == 0.2
        assert actual["fields"]["rounds"]["value"] == 7
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == 1
        assert methods.count("tab.reset_cfg") == int(reuse)
        assert methods.count("tab.new") == int(not reuse)
        assert methods.count("tab.save_data") == 1
        assert methods.count("tab.analyze") == 1
        assert methods.index("device.snapshot") < methods.index("tab.run_start")
        assert methods.index("soc.info") < methods.index("tab.run_start")
        assert methods.index("tab.save_data") < methods.index("tab.analyze")
        assert methods.index("tab.save_image") < methods.index("tab.writeback_preview")
        edits = [
            edit
            for method, params in client.transport.sent
            if method == "tab.edit_cfg"
            for edit in params["edits"]
        ]
        by_path = {tuple(edit["path"]): edit["value"] for edit in edits}
        assert by_path["modules", "reset"] is None
        assert by_path["modules", "init_pulse"] is None
    finally:
        client.context.session.close()


@pytest.mark.parametrize("reuse_tab_id", [None, "kept"])
def test_lookback_missing_frequency_does_not_run_a_blind_default(
    tmp_path, reuse_tab_id
):
    tab = reuse_tab_id or "new"
    publication = {
        "cfg_ref": {"cfg_id": "cfg", "revision": "2"},
        "status": "Valid",
        "source_basis": [],
        "diagnostics": [],
        "tree": {
            "kind": "section",
            "valid": True,
            "children": {
                "modules": {
                    "kind": "section",
                    "valid": True,
                    "children": {
                        "readout": {
                            "kind": "reference",
                            "ref": None,
                            "valid": True,
                            "children": {},
                        },
                    },
                },
            },
        },
    }

    def respond(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "context.snapshot":
            return {"label": "sample", "md": {}, "ml": {"modules": {}, "waveforms": {}}}
        if method == "tab.new":
            return {"tab_id": tab}
        if method == "tab.snapshot":
            return {
                "tabs": [
                    {
                        "tab_id": tab,
                        "adapter_name": "lookback",
                        "interaction": {
                            "is_running": False,
                            "is_analyzing": False,
                            "is_saving_data": False,
                        },
                    }
                ]
            }
        if method in ("tab.get_cfg", "tab.reset_cfg", "tab.edit_cfg"):
            return deepcopy(publication)
        raise AssertionError(f"Missing frequency must not start work: {method}")

    client = make_client(tmp_path, respond)
    try:
        arguments = {} if reuse_tab_id is None else {"reuse_tab_id": reuse_tab_id}
        reply = client.call("lookback", arguments)
        assert isinstance(reply, ToolReply)
        assert reply.data["status"] == "needs_parameters"
        assert reply.data["tab"] == tab
        assert [item["parameter"] for item in reply.data["missing"]] == [
            "frequency_mhz"
        ]
        assert reply.is_error is False
        methods = [method for method, _ in client.transport.sent]
        assert "tab.run_start" not in methods
        assert ("tab.new" in methods) is (reuse_tab_id is None)
        assert ("tab.reset_cfg" in methods) is (reuse_tab_id is not None)
    finally:
        client.context.session.close()


@pytest.mark.parametrize(
    "source", ["r_f", "library", "library_error", "library_invalid"]
)
def test_lookback_uses_only_valid_frequency_sources_and_keeps_gui_defaults(
    tmp_path, source
):
    gui = LookbackGui()
    gui.md["r_f"] = 6500.0
    readout = gui.publication["tree"]["children"]["modules"]["children"]["readout"]
    pulse = readout["children"]["pulse_cfg"]["children"]["freq"]
    adc = readout["children"]["ro_cfg"]["children"]["ro_freq"]
    pulse["input"]["resolved"] = 6100.0
    adc["input"]["resolved"] = 6110.0
    arguments = {} if source == "r_f" else {"readout_ref": "calibrated"}
    if source == "library_error":
        pulse["input"]["error"] = "unresolved"
    if source == "library_invalid":
        pulse["valid"] = False
        pulse["input"]["validation_error"] = "outside range"
    client = make_client(tmp_path, gui)
    try:
        reply = client.call("lookback", arguments)
        assert reply.data["status"] == "finished", reply.data
        fields = reply.data["actual"]["fields"]
        expected_pulse = 6100.0 if source == "library" else 6500.0
        expected_adc = 6500.0 if source == "r_f" else 6110.0
        assert fields["modules.readout.pulse_cfg.freq"]["value"] == expected_pulse
        assert fields["modules.readout.ro_cfg.ro_freq"]["value"] == expected_adc
        assert fields["modules.readout.pulse_cfg.freq"]["source"] == (
            "library:calibrated" if source == "library" else "r_f"
        )
        assert fields["modules.readout.ro_cfg.ro_length"]["value"] == 2.0
        assert fields["modules.readout.ro_cfg.trig_offset"]["value"] == 0.1
        assert fields["rounds"]["value"] == 3
    finally:
        client.context.session.close()


@pytest.mark.parametrize(
    "arguments",
    [
        {"frequency_mhz": True},
        {"frequency_mhz": float("nan")},
        {"readout_length_us": float("inf")},
        {"trigger_offset_us": "0.1"},
        {"rounds": True},
        {"rounds": 2.5},
        {"reuse_tab_id": ""},
        {"readout_ref": ""},
        {"use_reset": False},
        {"init_pulse_ref": 1},
    ],
)
def test_lookback_invalid_explicit_inputs_never_fall_back_or_create_tab(
    tmp_path, arguments
):
    gui = LookbackGui()
    gui.md["r_f"] = 6500.0
    client = make_client(tmp_path, gui)
    try:
        reply = client.call("lookback", arguments)
        assert reply.is_error
        assert reply.data["status"] == "failed"
        assert reply.data["tab"] is None
        assert not gui.ran
        assert not any(method.startswith("tab.") for method, _ in client.transport.sent)
    finally:
        client.context.session.close()


@pytest.mark.parametrize("partial_data", [False, True])
def test_manual_gui_run_cancel_never_starts_the_recipe_save_pipeline(
    tmp_path, partial_data
):
    gui = LookbackGui()

    def respond(method, params):
        if method == "operation.await" and params["operation_id"] == 71:
            gui.ran = partial_data
            return {"reason": "completed", "status": "cancelled"}
        return gui(method, params)

    client = make_client(tmp_path, respond)
    try:
        reply = client.call("lookback", {"frequency_mhz": 6000.0})
        assert reply.data["status"] == "cancelled"
        assert not reply.data["cancel_requested"]
        assert not reply.data["finish_early_requested"]
        assert reply.data["run_outcome"]["status"] == "cancelled"
        assert reply.data["raw_save"]["status"] == "not_started"
        methods = [method for method, _ in client.transport.sent]
        assert "tab.save_data" not in methods
        assert "tab.analyze" not in methods
        assert "operation.cancel" not in methods
    finally:
        client.context.session.close()


@pytest.mark.parametrize(
    ("argument", "module"),
    [
        ("readout_ref", "readout"),
        ("use_reset", "reset"),
        ("init_pulse_ref", "init_pulse"),
    ],
)
def test_explicit_invalid_library_reference_is_not_disabled_or_replaced(
    tmp_path, argument, module
):
    gui = LookbackGui()
    gui.md["r_f"] = 6500.0
    gui.publication["tree"]["children"]["modules"]["children"][module].update(
        valid=False, error="missing library entry"
    )
    client = make_client(tmp_path, gui)
    try:
        reply = client.call("lookback", {argument: "missing"})
        assert reply.is_error
        assert reply.data["status"] == "failed"
        assert reply.data["error"]["reason"] == "invalid_cfg"
        assert reply.data["tab"] == "t"
        edits = [
            params
            for method, params in client.transport.sent
            if method == "tab.edit_cfg"
        ]
        assert len(edits) == 1
        assert {"path": ["modules", module], "value": {"__ref": "missing"}} in edits[0][
            "edits"
        ]
        assert "tab.run_start" not in [method for method, _ in client.transport.sent]
    finally:
        client.context.session.close()


@pytest.mark.parametrize(
    ("pending_method", "wire_op", "phase", "raw_status"),
    [
        ("operation.await", 71, "run", "not_started"),
        ("tab.save_data", None, "raw_save", "unknown"),
        ("operation.await", 82, "raw_save", "unknown"),
        ("operation.await", 93, "analysis", "saved"),
    ],
)
def test_recipe_connection_loss_retains_known_prefix_without_reconnecting(
    tmp_path, monkeypatch, pending_method, wire_op, phase, raw_status
):
    client = make_client(tmp_path, LookbackGui(), port_is_open=lambda port: True)
    send_line = client.transport.send_line
    pending = Event()
    reconnects = []

    def send(payload):
        if payload["method"] == pending_method and (
            wire_op is None or payload["params"]["operation_id"] == wire_op
        ):
            client.transport.sent.append((payload["method"], payload["params"]))
            pending.set()
            return
        send_line(payload)

    def unexpected_connect(*args, **kwargs):
        reconnects.append((args, kwargs))
        raise AssertionError("Recipe must not reconnect")

    monkeypatch.setattr(client.transport, "send_line", send)
    monkeypatch.setattr(client.context.bridge, "connect", unexpected_connect)
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    try:
        initial = client.call("lookback", {"frequency_mhz": 6000.0})
        assert pending.wait(1)
        client.transport.close()
        assert client.transport.on_closed is not None
        client.transport.on_closed(None)
        execution = initial.data["execution"]
        terminal = client.call("wait", {"execution": execution, "timeout": 2})
        assert terminal.data["status"] == "failed"
        assert terminal.data["error"]["reason"] == "connection_lost"
        assert terminal.data["error"]["phase"] == phase
        assert terminal.data["raw_save"]["status"] == raw_status
        if wire_op == 82:
            assert terminal.data["raw_save"]["reserved_path"] == "/actual/raw.h5"
            assert terminal.data["raw_save"]["path"] is None
        if raw_status == "saved":
            assert terminal.data["raw_save"]["path"] == "/actual/raw.h5"
        state = client.call("status", {"execution": execution})
        assert state["error"] == terminal.data["error"]
        assert not reconnects
        assert client.transport.sent[-1][0] == pending_method
        assert [method for method, _ in client.transport.sent].count(
            "tab.run_start"
        ) == 1
    finally:
        client.context.session.close()


@pytest.mark.parametrize("failure", ["wrong_experiment", "busy", "missing", "stale"])
def test_lookback_reuse_errors_do_not_create_replacement_or_retry(tmp_path, failure):
    gui = LookbackGui()
    client = make_client(tmp_path, gui)
    if failure == "stale":
        client.transport.replies["tab.reset_cfg"] = {
            "ok": False,
            "error": {
                "code": "precondition_failed",
                "reason": "stale_cfg",
                "message": "changed",
            },
        }
    else:
        snapshot = gui("tab.snapshot", {"tab_id": "t"})
        if failure == "missing":
            snapshot["tabs"] = []
        elif failure == "busy":
            snapshot["tabs"][0]["interaction"]["is_analyzing"] = True
        else:
            snapshot["tabs"][0]["adapter_name"] = "other"
        client.transport.replies["tab.snapshot"] = {"ok": True, "result": snapshot}
    try:
        reply = client.call("lookback", {"reuse_tab_id": "t", "frequency_mhz": 6000.0})
        assert reply.is_error
        assert reply.data["tab"] == "t"
        methods = [method for method, _ in client.transport.sent]
        assert "tab.new" not in methods
        assert "tab.run_start" not in methods
        assert methods.count("tab.reset_cfg") == int(failure == "stale")
    finally:
        client.context.session.close()


@pytest.mark.parametrize(
    "failure, phase, raw_status",
    [
        ("run", "run", "not_started"),
        ("raw_start", "raw_save", "failed"),
        ("raw_finish", "raw_save", "failed"),
        ("analysis", "analysis", "saved"),
        ("superseded_raw", "raw_save", "failed"),
        ("superseded_analysis", "analysis", "saved"),
        ("superseded_writeback", "writeback_read", "saved"),
        ("image", "analysis", "saved"),
        ("writeback", "writeback_read", "saved"),
    ],
)
def test_lookback_failure_preserves_completed_prefix_without_retry(
    tmp_path, failure, phase, raw_status
):
    gui = LookbackGui()

    def respond(method, params):
        fail_operation = {"run": 71, "raw_finish": 82, "analysis": 93}.get(failure)
        if method == "operation.await" and params["operation_id"] == fail_operation:
            return {
                "reason": "completed",
                "status": "failed",
                "error": "injected failure",
            }
        return gui(method, params)

    client = make_client(tmp_path, respond)
    failed_method = {
        "raw_start": "tab.save_data",
        "superseded_raw": "tab.save_data",
        "superseded_analysis": "tab.analyze",
        "superseded_writeback": "tab.writeback_preview",
        "image": "tab.save_image",
        "writeback": "tab.writeback_preview",
    }.get(failure)
    if failed_method:
        client.transport.replies[failed_method] = {
            "ok": False,
            "error": {
                "code": "precondition_failed",
                "reason": "result_superseded"
                if failure.startswith("superseded")
                else "injected",
                "message": "injected failure",
            },
        }
    try:
        reply = client.call("lookback", {"frequency_mhz": 6000.0})
        data = reply.data
        assert data["status"] == "failed", data
        assert reply.is_error
        assert data["error"]["phase"] == phase
        if failure.startswith("superseded"):
            assert data["error"]["reason"] == "result_superseded"
        assert data["raw_save"]["status"] == raw_status
        assert data["tab"] == "t"
        assert data["run_op"] is not None
        if raw_status == "saved":
            assert data["raw_save"]["path"] == "/actual/raw.h5"
        if failure == "raw_finish":
            assert data["raw_save"]["reserved_path"] == "/actual/raw.h5"
            assert data["raw_save"]["operation_outcome"]["status"] == "failed"
        if failure in ("image", "writeback"):
            assert data["analysis"]["result"]["summary"] == {"offset": 0.24}
        if failure == "writeback":
            assert data["analysis"]["saved_images"] == [
                {"figure_name": "trace", "image_path": "/actual/trace.png"}
            ]
            assert reply.images[0].data == _PNG
        methods = [method for method, _ in client.transport.sent]
        for method in (
            "tab.run_start",
            "tab.save_data",
            "tab.analyze",
            "tab.writeback_preview",
        ):
            assert methods.count(method) <= 1
        if raw_status != "saved":
            assert "tab.analyze" not in methods
    finally:
        client.context.session.close()
