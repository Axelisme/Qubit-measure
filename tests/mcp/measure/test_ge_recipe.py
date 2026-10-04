"""GE calibration behavior through shipped tools and the GUI wire boundary."""

import base64
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event
from typing import Any

import pytest
from zcu_tools.mcp.core.bridge import GuiTransportTimeoutError
from zcu_tools.mcp.core.reply import ToolReply

from ._recipe_support import PNG, LookbackGui, scalar
from ._support import MeasureClient, full_execution_reply, make_client


def start_ge(client: MeasureClient) -> ToolReply:
    """Start GE through public admission with a short local wait for blocked RPCs."""
    return client.context.session.recipes.start(
        client.context, "singleshot_ge", {"pi_ref": "pi"}
    ).wait(0.01)


def skip_writeback(client: MeasureClient, question: ToolReply) -> ToolReply:
    """Observe both captured stages and answer without draft writes."""
    assert question.data["status"] == "awaiting_answer", question.data
    assert tuple(question.data["question_items"]) == ("predict_offset", "classifier")
    before = list(client.transport.sent)
    reply = client.call(
        "answer", {"recipe": question.data["execution"], "decision": "skipped"}
    )
    assert reply.data["status"] == "finished", reply.data
    assert client.transport.sent == before
    return reply


class GeGui(LookbackGui):
    def __init__(self, *, interactive=False):
        super().__init__()
        self.interactive = interactive
        self.current_stage = "primary"
        self.done = {"primary": Event(), "post": Event()}
        self.writes: list[dict[str, object]] = []
        self.calls = []
        self.library = {"pi": {}, "readout": {}}
        self.md = {"r_f": 5100.0}
        self.publication["tree"]["children"]["shots"] = scalar(7000)
        self.publication["tree"]["children"]["reps"] = scalar(1)
        self.publication["tree"]["children"]["rounds"] = scalar(1)
        modules = self.publication["tree"]["children"]["modules"]["children"]
        modules["probe_pulse"] = {
            "kind": "reference",
            "valid": True,
            "ref": "<Custom:Pulse>",
            "error": None,
            "children": {"freq": scalar(6100.0)},
        }

    def _observations(self):
        observations = super()._observations()
        observations["context.snapshot"]["ml"]["modules"] = self.library
        observations["tab.snapshot"]["tabs"][0]["adapter_name"] = "singleshot/ge"
        return observations

    def __call__(self, method, params) -> dict[str, Any]:
        self.calls.append((method, deepcopy(params)))
        if method == "tab.interact":
            if params.get("payload", {}).get("command") == "done":
                self.done[self.current_stage].set()
            return {
                "operation_id": 93 if self.current_stage == "primary" else 104,
                "plugin": "ge-picker",
                "state": {"stage": self.current_stage},
                "info": {},
                "commands": [{"name": "done"}],
                "preview_active": True,
                "figure": {"png_b64": base64.b64encode(PNG).decode()}
                if params.get("include_figure", True)
                else None,
            }
        if method == "operation.await" and params["operation_id"] in (93, 104):
            stage = "primary" if params["operation_id"] == 93 else "post"
            if self.interactive and not self.done[stage].is_set():
                return {"reason": "timeout", "status": "interactive"}
        if method == "tab.writeback_write":
            self.writes.append(deepcopy(params))
            return {
                "written": [
                    {
                        "id": item["id"],
                        "kind": "md",
                        "target": item["id"],
                        "before": {"value": 0.0},
                        "after": {"value": 1.0},
                    }
                    for item in params["write"]
                ]
            }
        if "operation_id" not in params and method in (
            "tab.get_analyze_result",
            "tab.get_post_analyze_result",
            "tab.writeback_preview",
        ):
            return self._current_draft(method, params)
        return self._native_reply(method, params)

    def _current_draft(self, method, params):
        if method in ("tab.get_analyze_result", "tab.get_post_analyze_result"):
            return {"summary": {}}
        if method == "tab.writeback_preview":
            if params["subtab_id"] == "post_analysis":
                return self._post_result(method, params)
            return {
                "has_draft": True,
                "items": [
                    {
                        "id": "md-1",
                        "kind": "metadict",
                        "target_name": "predict_offset",
                        "proposed": 0.24,
                        "current": 0.0,
                        "selected": False,
                    }
                ],
                "destination_context": {"active_label": "sample"},
            }
        raise AssertionError(method)

    def _native_reply(self, method, params):
        if method == "tab.post_analyze":
            self.current_stage = "post"
            assert params == {
                "tab_id": "t",
                "updates": {},
                "operation_id": 93,
                "run_operation_id": 71,
            }
            return {
                "operation_id": 104,
                "interactive": self.interactive,
                "params": {"bins": 64},
                "invalidated_on_success": [],
            }
        if method == "operation.await" and params["operation_id"] == 104:
            return {"reason": "completed", "status": "finished"}
        if method == "tab.get_post_analyze_result":
            assert params == {"tab_id": "t", "operation_id": 104}
            return {
                "summary": {"fidelity": 0.98},
                "invalid": [],
                "params": {"bins": 64},
                "operation_state": {"post_analysis_state": {"figure_names": ["cloud"]}},
            }
        if params.get("operation_id") == 104:
            return self._post_result(method, params)
        if method == "tab.new":
            assert params == {"adapter_name": "singleshot/ge"}
            return {"tab_id": "t"}
        reply = super().__call__(method, params)
        if method == "tab.analyze":
            self.current_stage = "primary"
            reply["interactive"] = self.interactive
        if method == "tab.edit_cfg":
            self._edit_references(reply, params)
        if method == "tab.writeback_preview":
            reply["items"][0].update(
                kind="metadict",
                target_name="predict_offset",
                current=0.0,
                selected=False,
            )
        return reply

    def _edit_references(self, reply, params):
        modules = self.publication["tree"]["children"]["modules"]["children"]
        for edit in params["edits"]:
            if edit["path"][0] == "modules" and len(edit["path"]) == 2:
                name = edit["value"]["__ref"]
                if name is not None and name not in self.library:
                    modules[edit["path"][1]].update(
                        valid=False, error="unknown library"
                    )
                    reply["tree"]["children"]["modules"]["children"][
                        edit["path"][1]
                    ].update(valid=False, error="unknown library")

    def _post_result(self, method, params):
        assert params["subtab_id"] == "post_analysis"
        if method == "tab.save_image":
            assert params["figure_name"] in ("cloud", "histogram")
            return {"image_path": f"/actual/{params['figure_name']}.png"}
        if method == "tab.get_figure":
            return {"png_b64": base64.b64encode(PNG).decode()}
        if method == "tab.writeback_preview":
            return {
                "has_draft": True,
                "items": [
                    {
                        "id": "classifier",
                        "kind": "metadict",
                        "target_name": "classifier",
                        "proposed": 0.98,
                        "current": 0.0,
                        "selected": False,
                    }
                ],
                "destination_context": {"active_label": "sample"},
            }
        raise AssertionError(method)


@pytest.fixture()
def ge_client(tmp_path):
    gui = GeGui()
    client = make_client(tmp_path, gui)
    try:
        yield gui, client
    finally:
        client.context.session.close()


def test_ge_estimates_preserve_three_native_fit_stages_without_post_refit(ge_client):
    gui, client = ge_client
    quality = {
        "joint": {
            "r2": None,
            "normalized_residual_rms": 0.1,
            "relative_parameter_errors": {"sigma": 0.2},
            "invalid": [
                {"path": "summary.fit_quality.joint.r2", "reason": "non_finite"}
            ],
        },
        "ground": {
            "r2": 0.85,
            "normalized_residual_rms": 0.02,
            "relative_parameter_errors": {"p0": None},
            "invalid": [
                {
                    "path": "summary.fit_quality.ground.relative_parameter_errors.p0",
                    "reason": "zero_parameter",
                }
            ],
        },
        "excited": {
            "r2": 0.9,
            "normalized_residual_rms": None,
            "relative_parameter_errors": {"p0": 0.3},
            "invalid": [
                {
                    "path": "summary.fit_quality.excited.normalized_residual_rms",
                    "reason": "zero_range",
                }
            ],
        },
    }

    def result(params):
        reply = gui("tab.get_analyze_result", params)
        reply["summary"] = {
            "fidelity": 0.98,
            "theta": 0.2,
            "threshold": -0.1,
            "ge_s": 0.42,
            "init_pops": [[0.9, 0.1], [0.05, 0.95]],
            "fit_quality": quality,
        }
        reply["invalid"] = quality["joint"]["invalid"]
        return {"ok": True, "result": reply}

    client.transport.replies["tab.get_analyze_result"] = result
    initial = skip_writeback(client, client.call("singleshot_ge", {"pi_ref": "pi"}))
    assert initial.data["status"] == "finished", initial.data
    key = initial.data["execution"]
    before = len(client.transport.sent)
    summary = client.call("status", {"execution": key})
    full = client.call("status", {"execution": key, "detail": "full"})
    assert len(client.transport.sent) == before
    assert full["analysis"]["result"]["summary"]["fit_quality"] == quality
    primary = summary["analysis"]["primary"]
    assert set(primary["estimates"]) == {"fidelity", "theta", "threshold", "ge_s"}
    for name, estimate in primary["estimates"].items():
        stages = estimate["quality"]
        assert set(stages) == {"joint", "ground", "excited"}
        assert stages["joint"]["r2"] is None
        assert stages["ground"]["r2"] == 0.85
        assert stages["excited"]["relative_parameter_errors"]["p0"] == 0.3
        issue = {
            "path": f"analysis.primary.estimates.{name}.quality.joint.r2",
            "reason": "non_finite",
        }
        assert stages["joint"]["invalid"] == [issue]
        assert summary["invalid"].count(issue) == 1
        assert all(
            issue["path"].startswith(f"analysis.primary.estimates.{name}.quality.")
            for stage in stages.values()
            for issue in stage["invalid"]
        )
    assert len(summary["invalid"]) == 12
    assert primary["details"] == {"init_pops": [[0.9, 0.1], [0.05, 0.95]]}
    assert summary["analysis"]["post"]["estimates"]["fidelity"]["quality"] is None
    json.dumps(summary, allow_nan=False)
    json.dumps(full, allow_nan=False)


@pytest.fixture()
def background_ge_client(ge_client):
    return ge_client


@pytest.mark.parametrize("stage", ["primary", "post"])
@pytest.mark.parametrize("receipt", ["delayed", "lost", "timeout"])
def test_ge_unconfirmed_analysis_start_retains_unknown_and_saved_prefix(
    background_ge_client, monkeypatch, stage, receipt
):
    gui, client = background_ge_client
    pending, release = Event(), Event()
    method = "tab.analyze" if stage == "primary" else "tab.post_analyze"
    send_line = client.transport.send_line

    def send(payload):
        if payload["method"] != method:
            return send_line(payload)
        if receipt == "delayed":
            pending.set()
            assert release.wait(2)
            return send_line(payload)
        client.transport.sent.append((payload["method"], payload["params"]))
        gui(method, payload["params"])
        pending.set()
        if receipt == "timeout":
            raise GuiTransportTimeoutError(method, 0.01)
        return None

    monkeypatch.setattr(client.transport, "send_line", send)
    try:
        initial = start_ge(client)
        assert pending.wait(1)
        execution = initial.data["execution"]
        before = len(client.transport.sent)
        summary = client.call("status", {"execution": execution})
        full = client.call("status", {"execution": execution, "detail": "full"})
        assert summary["steps"]["analysis"][stage]["status"] == "unknown"
        assert full["analysis_starts"][stage]["status"] == "unknown"
        assert summary["artifacts"]["raw"]["data"]["members"]["data"] == [
            {"path": "/actual/raw.h5", "status": "saved"}
        ]
        if stage == "post":
            assert summary["artifacts"]["analysis"]["trace"]["status"] == "saved"
        assert len(client.transport.sent) == before
        if receipt == "delayed":
            release.set()
        elif receipt == "lost":
            client.transport.close()
            assert client.transport.on_closed is not None
            client.transport.on_closed(None)
        completed = client.call("wait", {"execution": execution, "timeout": 2})
        confirmed = client.call("status", {"execution": execution})
        if receipt == "delayed":
            assert completed.data["status"] == "awaiting_answer", completed.data
            skip_writeback(client, completed)
            assert confirmed["steps"]["analysis"][stage]["status"] == "finished"
        else:
            assert completed.data["status"] == "failed", completed.data
            assert confirmed["steps"]["analysis"][stage]["status"] == "unknown"
            assert confirmed["error"]["reason"] == (
                "connection_lost" if receipt == "lost" else "gui_transport_timeout"
            )
        assert [name for name, _ in client.transport.sent].count(method) == 1
    finally:
        release.set()


@pytest.mark.parametrize("stage", ["primary", "post"])
def test_ge_cancel_during_writeback_retains_stage_and_prevents_next_admission(
    background_ge_client, stage
):
    gui, client = background_ge_client
    pending, release = Event(), Event()
    operation = 93 if stage == "primary" else 104

    def writeback(params):
        if params["operation_id"] == operation:
            pending.set()
            assert release.wait(2)
        return {"ok": True, "result": gui("tab.writeback_preview", params)}

    client.transport.replies["tab.writeback_preview"] = writeback
    client.transport.replies["operation.cancel"] = {
        "ok": True,
        "result": {"status": "finished"},
    }
    try:
        initial = full_execution_reply(client, start_ge(client))
        assert pending.wait(1)
        execution = initial.data["execution"]
        status = client.call("status", {"execution": execution, "detail": "full"})
        assert status["analysis_stage"] == stage
        with ThreadPoolExecutor(max_workers=1) as pool:
            request = pool.submit(client.call, "cancel", {"execution": execution})
            try:
                for _ in range(100):
                    if client.call("status", {"execution": execution})[
                        "cancel_requested"
                    ]:
                        break
                    release.wait(0.01)
                else:
                    pytest.fail("Cancel intent did not reach the public snapshot")
            finally:
                release.set()
            assert (
                request.result(timeout=2).data["gui_cancel"]["status"] == "not_needed"
            )
        terminal = full_execution_reply(
            client, client.call("wait", {"execution": execution, "timeout": 2})
        )
        data = terminal.data
        assert data["status"] == "cancelled", data
        assert data["writeback"]["items"][0]["id"] == "md-1"
        assert data["analysis"]["status"] == (
            "cancelled" if stage == "primary" else "finished"
        )
        assert data["raw_save"]["path"] == "/actual/raw.h5"
        assert len(terminal.images) == (1 if stage == "primary" else 2)
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.post_analyze") == (0 if stage == "primary" else 1)
        assert methods.count("operation.cancel") == 1
        if stage == "post":
            assert data["post_writeback"]["items"][0]["id"] == "classifier"
    finally:
        release.set()


@pytest.mark.parametrize("stage", ["primary", "post"])
@pytest.mark.parametrize("outcome", ["cancelled", "failed"])
def test_ge_cancel_targets_late_stage_receipt_and_joins_true_outcome(
    background_ge_client, stage, outcome
):
    gui, client = background_ge_client
    pending, release, stopped = Event(), Event(), Event()
    start_method = "tab.analyze" if stage == "primary" else "tab.post_analyze"
    operation = 93 if stage == "primary" else 104

    def start(params):
        pending.set()
        assert release.wait(2)
        return {"ok": True, "result": gui(start_method, params)}

    def cancel(params):
        assert params == {"operation_id": operation}
        stopped.set()
        return {"ok": True, "result": {"status": "cancelling"}}

    def await_operation(params):
        if params["operation_id"] != operation:
            return {"ok": True, "result": gui("operation.await", params)}
        result = (
            {"reason": "completed", "status": outcome, "error": "failure after cancel"}
            if stopped.is_set()
            else {"reason": "timeout"}
        )
        return {"ok": True, "result": result}

    client.transport.replies.update(
        {
            start_method: start,
            "operation.cancel": cancel,
            "operation.await": await_operation,
        }
    )
    try:
        initial = full_execution_reply(client, start_ge(client))
        assert pending.wait(1)
        execution = initial.data["execution"]
        for _ in range(2):
            assert client.call("cancel", {"execution": execution}).data[
                "cancel_requested"
            ]
        assert not stopped.is_set()
        release.set()
        terminal = full_execution_reply(
            client, client.call("wait", {"execution": execution, "timeout": 2})
        )
        data = terminal.data
        assert stopped.is_set()
        assert data["status"] == outcome, data
        current = data["analysis" if stage == "primary" else "post_analysis"]
        assert current["status"] == outcome
        assert current["cancel_requested"]
        assert current["operation_outcome"]["status"] == outcome
        assert data["raw_save"]["path"] == "/actual/raw.h5"
        assert len(terminal.images) == (1 if stage == "post" else 0)
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("operation.cancel") == 1
        assert methods.count("tab.post_analyze") == (1 if stage == "post" else 0)
        assert methods.count("tab.get_post_analyze_result") == 0
        if stage == "post":
            assert data["analysis"]["status"] == "finished"
            assert data["writeback"]["items"][0]["id"] == "md-1"
    finally:
        release.set()
        stopped.set()


@pytest.mark.parametrize("save_failed", [False, True])
def test_ge_cancel_during_post_save_preserves_real_save_outcome(
    background_ge_client, save_failed
):
    gui, client = background_ge_client
    pending, release = Event(), Event()

    def save(params):
        if params["operation_id"] == 104:
            pending.set()
            assert release.wait(2)
            if save_failed:
                return {
                    "ok": False,
                    "error": {
                        "code": "io_error",
                        "reason": "save_failed",
                        "message": "save failed after cancellation",
                    },
                }
        return {"ok": True, "result": gui("tab.save_image", params)}

    client.transport.replies["tab.save_image"] = save
    client.transport.replies["operation.cancel"] = {
        "ok": True,
        "result": {"status": "finished"},
    }
    try:
        initial = full_execution_reply(client, start_ge(client))
        assert pending.wait(1)
        execution = initial.data["execution"]
        with ThreadPoolExecutor(max_workers=1) as pool:
            request = pool.submit(client.call, "cancel", {"execution": execution})
            try:
                for _ in range(100):
                    if client.call("status", {"execution": execution})[
                        "cancel_requested"
                    ]:
                        break
                    release.wait(0.01)
                else:
                    pytest.fail("Cancel intent did not reach the public snapshot")
            finally:
                release.set()
            cancelled = request.result(timeout=2)
        assert cancelled.data["gui_cancel"]["status"] == "not_needed"
        terminal = full_execution_reply(
            client, client.call("wait", {"execution": execution, "timeout": 2})
        )
        data = terminal.data
        assert data["status"] == ("failed" if save_failed else "cancelled"), data
        assert data["post_analysis"]["save_status"] == (
            "incomplete" if save_failed else "saved"
        )
        assert data["post_analysis"]["saved_images"] == (
            []
            if save_failed
            else [{"figure_name": "cloud", "image_path": "/actual/cloud.png"}]
        )
        assert data["analysis"]["status"] == "finished"
        assert data["raw_save"]["path"] == "/actual/raw.h5"
        assert len(terminal.images) == 1
        assert data["post_writeback"] is None
        assert not any(
            method in ("tab.get_figure", "tab.writeback_preview")
            and params.get("operation_id") == 104
            for method, params in client.transport.sent
        )
    finally:
        release.set()


@pytest.mark.parametrize(
    "method_pending,operation",
    [
        ("tab.writeback_preview", 93),
        ("tab.post_analyze", None),
        ("tab.save_image", 104),
    ],
)
def test_ge_close_stops_stage_admission_without_reconnect(
    background_ge_client, monkeypatch, method_pending, operation
):
    gui, client = background_ge_client
    pending, disconnected = Event(), Event()
    original_close = client.transport.close

    def respond(params):
        if operation is None or params.get("operation_id") == operation:
            pending.set()
            assert disconnected.wait(2), "Session must disconnect before joining"
        return {"ok": True, "result": gui(method_pending, params)}

    def disconnect():
        original_close()
        disconnected.set()

    client.transport.replies[method_pending] = respond
    monkeypatch.setattr(client.transport, "close", disconnect)
    try:
        initial = start_ge(client)
        assert pending.wait(1)
        before = len(client.transport.sent)
        client.context.session.close()
        data = client.call(
            "status", {"execution": initial.data["execution"], "detail": "full"}
        )
        assert data["status"] == "cancelled", data
        assert data["phase"] == "terminal"
        assert data["error"] is None
        assert data["raw_save"]["path"] == "/actual/raw.h5"
        assert data["analysis"]["status"] == (
            "failed" if method_pending == "tab.writeback_preview" else "finished"
        )
        if method_pending == "tab.writeback_preview":
            assert data["analysis"]["error"]["phase"] == "writeback_read"
            assert data["analysis"]["writeback"] is None
        assert data["analysis"]["saved_images"] == [
            {"figure_name": "trace", "image_path": "/actual/trace.png"}
        ]
        assert len(client.transport.sent) == before
        client.context.session.close()
    finally:
        disconnected.set()


@pytest.mark.parametrize("reuse", [False, True])
def test_ge_reports_all_missing_calibration_without_running(ge_client, reuse):
    gui, client = ge_client
    gui.md.clear()
    data = client.call("singleshot_ge", {"reuse_tab_id": "t"} if reuse else {}).data
    assert data["status"] == "needs_parameters", data
    assert {item["parameter"] for item in data["missing"]} == {"pi_ref", "readout_ref"}
    assert not gui.ran
    methods = [method for method, _ in gui.calls]
    assert ("tab.reset_cfg" in methods) == reuse
    assert ("tab.new" in methods) != reuse


@pytest.mark.parametrize("explicit", [False, True])
def test_ge_preserves_calibrated_library_and_gui_shots_defaults(ge_client, explicit):
    gui, client = ge_client
    gui.md.clear()
    modules = gui.publication["tree"]["children"]["modules"]["children"]
    if not explicit:
        modules["probe_pulse"]["ref"] = "pi"
        modules["readout"]["ref"] = "readout"
    arguments = {"pi_ref": "pi", "readout_ref": "readout"} if explicit else {}
    data = full_execution_reply(
        client, skip_writeback(client, client.call("singleshot_ge", arguments))
    ).data
    assert data["status"] == "finished", data
    fields = data["actual"]["fields"]
    assert fields["shots"]["value"] == 7000
    assert fields["shots"]["source"] == "gui_default"
    assert fields["modules.readout.ro_cfg.ro_freq"]["value"] == 5000.0
    assert fields["modules.readout.ro_cfg.ro_freq"]["source"] == "library:readout"
    assert fields["modules.probe_pulse"]["value"] == "pi"
    assert fields["modules.probe_pulse"]["source"] == (
        "explicit" if explicit else "gui_default"
    )


@pytest.mark.parametrize("reuse", [False, True])
def test_ge_optional_refs_are_explicit_and_omission_resets_them(ge_client, reuse):
    gui, client = ge_client
    gui.library.update(reset={}, init={})
    modules = gui.publication["tree"]["children"]["modules"]["children"]
    modules["reset"]["ref"] = "old_reset"
    modules["init_pulse"]["ref"] = "old_init"
    arguments = {
        "pi_ref": "pi",
        "reuse_tab_id": "t" if reuse else None,
        "use_reset": None if reuse else "reset",
        "init_pulse_ref": None if reuse else "init",
    }
    data = skip_writeback(client, client.call("singleshot_ge", arguments)).data
    assert data["status"] == "finished", data
    assert modules["reset"]["ref"] == (None if reuse else "reset")
    assert modules["init_pulse"]["ref"] == (None if reuse else "init")
    assert sum(method == "tab.run_start" for method, _ in gui.calls) == 1


@pytest.mark.parametrize(
    "parameter", ["pi_ref", "readout_ref", "use_reset", "init_pulse_ref"]
)
def test_ge_does_not_replace_missing_explicit_library_refs(ge_client, parameter):
    gui, client = ge_client
    data = client.call("singleshot_ge", {"pi_ref": "pi", parameter: "missing"}).data
    assert data["status"] == "failed", data
    assert data["error"]["reason"] == "invalid_cfg"
    assert not gui.ran


@pytest.mark.parametrize("stage", ["primary", "post"])
@pytest.mark.parametrize(
    "failure", ["start", "analysis", "result", "save", "png", "writeback"]
)
def test_ge_stage_failure_preserves_completed_prefix(ge_client, stage, failure):
    gui, client = ge_client
    operation = 93 if stage == "primary" else 104
    result_method = (
        "tab.get_analyze_result"
        if stage == "primary"
        else "tab.get_post_analyze_result"
    )
    failing_method = {
        "start": "tab.analyze" if stage == "primary" else "tab.post_analyze",
        "analysis": "operation.await",
        "result": result_method,
        "save": "tab.save_image",
        "png": "tab.get_figure",
        "writeback": "tab.writeback_preview",
    }[failure]

    def result_reply(params):
        result = gui(result_method, params)
        if stage == "post":
            result["operation_state"]["post_analysis_state"]["figure_names"] = [
                "cloud",
                "histogram",
            ]
        return {"ok": True, "result": result}

    def failure_reply(params):
        is_target = failure == "start" or params.get("operation_id") == operation
        if failure == "save" and stage == "post":
            is_target = is_target and params["figure_name"] == "histogram"
        if not is_target:
            return {"ok": True, "result": gui(failing_method, params)}
        if failure == "analysis":
            return {
                "ok": True,
                "result": {
                    "reason": "completed",
                    "status": "failed",
                    "error": "analysis failed",
                },
            }
        return {
            "ok": False,
            "error": {
                "code": "precondition_failed",
                "reason": "result_superseded",
                "message": "injected failure",
            },
        }

    client.transport.replies[result_method] = result_reply
    client.transport.replies[failing_method] = failure_reply
    reply = full_execution_reply(client, client.call("singleshot_ge", {"pi_ref": "pi"}))
    data = reply.data
    assert data["status"] == "failed", data
    assert data["tab"] == "t"
    assert data["raw_save"]["path"] == "/actual/raw.h5"
    assert data["analysis_stage"] == stage
    methods = [method for method, _ in client.transport.sent]
    assert methods.count("tab.run_start") == 1
    assert methods.count("tab.post_analyze") == (1 if stage == "post" else 0)
    assert data["post_writeback"] is None
    if stage == "post":
        assert data["analysis"]["status"] == "finished"
        assert data["analysis"]["saved_images"] == [
            {"figure_name": "trace", "image_path": "/actual/trace.png"}
        ]
        assert data["writeback"]["items"][0]["id"] == "md-1"
    else:
        assert data["writeback"] is None
        assert data["post_analysis"] is None
    current = data["analysis" if stage == "primary" else "post_analysis"]
    if failure == "start":
        assert current["status"] == "failed"
        assert current["start"] == {
            "status": "not_started",
            "reason": "result_superseded",
        }
        assert current["op"] is None
    elif failure == "save" and stage == "post":
        assert current["saved_images"] == [
            {"figure_name": "cloud", "image_path": "/actual/cloud.png"}
        ]
        assert current["remaining_images"] == ["histogram"]
    elif failure in ("png", "writeback"):
        assert len(current["saved_images"]) == (2 if stage == "post" else 1)
    assert len(reply.images) == (1 if stage == "post" else 0) + (
        1 if failure == "writeback" else 0
    )


@pytest.mark.parametrize("failure", ["edit", "invalid"])
def test_ge_gui_cfg_rejection_stops_before_run(tmp_path, failure):
    gui = GeGui()

    def responder(method, params):
        result = gui(method, params)
        if method == "tab.edit_cfg" and failure == "invalid":
            result["status"] = "Invalid"
        return result

    client = make_client(tmp_path, responder)
    if failure == "edit":
        client.transport.replies["tab.edit_cfg"] = {
            "ok": False,
            "error": {
                "code": "precondition_failed",
                "reason": "invalid_cfg",
                "message": "Reference is not applicable",
            },
        }
    try:
        data = client.call("singleshot_ge", {"pi_ref": "pi"}).data
        assert data["status"] == "failed", data
        assert data["error"]["reason"] == "invalid_cfg"
        assert not gui.ran
    finally:
        client.context.session.close()


@pytest.mark.parametrize(
    "arguments",
    [
        {"shots": value}
        for value in (0, -1, True, 1.0, 2.5, float("inf"), float("nan"), "10")
    ]
    + [
        {name: value}
        for name in (
            "reuse_tab_id",
            "readout_ref",
            "pi_ref",
            "use_reset",
            "init_pulse_ref",
        )
        for value in ("", " ", [], False)
    ],
)
def test_ge_rejects_invalid_arguments_before_preparing(tmp_path, arguments):
    gui = GeGui()
    client = make_client(tmp_path, gui)
    try:
        data = client.call("singleshot_ge", {"pi_ref": "pi", **arguments}).data
        assert data["status"] == "failed", data
        assert not any(method == "tab.run_start" for method, _ in client.transport.sent)
        assert not gui.ran
        assert not any(method == "context.snapshot" for method, _ in gui.calls)
    finally:
        client.context.session.close()


def test_ge_requires_calibrated_pi_instead_of_custom_template(tmp_path):
    gui = GeGui()
    client = make_client(tmp_path, gui)
    try:
        reply = client.call("singleshot_ge", {})
        assert reply.data["status"] == "needs_parameters", reply.data
        assert {item["parameter"] for item in reply.data["missing"]} == {"pi_ref"}
        assert not gui.ran
    finally:
        client.context.session.close()


def test_ge_preserves_invalid_analysis_and_saved_paths_without_accepting(ge_client):
    gui, client = ge_client
    for method in ("tab.get_analyze_result", "tab.get_post_analyze_result"):

        def result(params, result_method=method):
            observed = gui(result_method, params)
            observed["summary"]["stderr"] = None
            observed["invalid"] = [{"path": "summary.stderr", "reason": "non_finite"}]
            return {"ok": True, "result": observed}

        client.transport.replies[method] = result
    reply = full_execution_reply(client, client.call("singleshot_ge", {"pi_ref": "pi"}))
    reply = full_execution_reply(client, skip_writeback(client, reply))
    data = json.loads(json.dumps(reply.data, allow_nan=False))
    assert data["status"] == "finished"
    assert data["raw_save"]["path"] == "/actual/raw.h5"
    for stage, summary, image in (
        ("analysis", {"offset": 0.24, "stderr": None}, "trace"),
        ("post_analysis", {"fidelity": 0.98, "stderr": None}, "cloud"),
    ):
        assert data[stage]["result"]["summary"] == summary
        assert data[stage]["result"]["invalid"] == [
            {"path": "summary.stderr", "reason": "non_finite"}
        ]
        assert data[stage]["saved_images"] == [
            {"figure_name": image, "image_path": f"/actual/{image}.png"}
        ]
    assert "tab.writeback_apply" not in [method for method, _ in client.transport.sent]


@pytest.mark.parametrize("decision", ["accepted", "skipped"])
@pytest.mark.parametrize("interactive", [False, True])
def test_ge_writeback_waits_for_both_stages_and_reports_actual_receipts(
    tmp_path, decision, interactive
):
    gui = GeGui(interactive=interactive)
    client = make_client(tmp_path, gui)
    try:
        reply = client.call("singleshot_ge", {"pi_ref": "pi"})
        execution = reply.data["execution"]
        if interactive:
            assert reply.data["status"] == "interactive"
            post = client.call(
                "tab_interact", {"tab": "t", "payload": {"command": "done"}}
            )
            assert post.data["execution"] == execution
            assert post.data["status"] == "interactive"
            assert post.data["analysis"]["stage"] == "post"
            reply = client.call(
                "tab_interact", {"tab": "t", "payload": {"command": "done"}}
            )
        assert reply.data["status"] == "awaiting_answer", reply.data
        assert reply.data["question_items"] == ("predict_offset", "classifier")
        assert reply.data["writeback"]["receipts"] == ()
        assert len(reply.images) == 2
        assert (
            len(reply.data["previews"]["primary"])
            == len(reply.data["previews"]["post"])
            == 1
        )
        assert gui.writes == []
        before = list(client.transport.sent)
        summary = client.call("status", {"execution": execution})
        full = client.call("status", {"execution": execution, "detail": "full"})
        assert summary["question_items"] == full["question_items"]
        assert client.transport.sent == before
        result = client.call("answer", {"recipe": execution, "decision": decision})
        assert result.data["status"] == "finished", result.data
        assert len(result.images) == 2
        receipts = result.data["writeback"]["receipts"]
        if decision == "accepted":
            assert [request["subtab_id"] for request in gui.writes] == [
                "analysis",
                "post_analysis",
            ]
            assert [request["write"] for request in gui.writes] == [
                [{"id": "md-1"}],
                [{"id": "classifier"}],
            ]
            assert len(receipts) == 1
            assert [stage["stage"] for stage in receipts[0]["completed"]] == [
                "primary",
                "post",
            ]
        else:
            assert gui.writes == [] and receipts == ()
    finally:
        client.context.session.close()


def test_ge_answer_failure_keeps_confirmed_primary_write_and_all_previews(ge_client):
    gui, client = ge_client
    question = client.call("singleshot_ge", {"pi_ref": "pi"})
    assert question.data["status"] == "awaiting_answer", question.data

    def write(params):
        if params["subtab_id"] == "post_analysis":
            return {
                "ok": False,
                "error": {
                    "code": "io_error",
                    "reason": "write_failed",
                    "message": "Post write failed",
                },
            }
        return {"ok": True, "result": gui("tab.writeback_write", params)}

    client.transport.replies["tab.writeback_write"] = write
    failed = client.call(
        "answer", {"recipe": question.data["execution"], "decision": "accepted"}
    )
    assert failed.data["status"] == "failed" and failed.is_error
    assert len(failed.images) == 2
    (receipt,) = failed.data["writeback"]["receipts"]
    assert [stage["stage"] for stage in receipt["completed"]] == ["primary"]
    assert receipt["failed_stage"] == "post"
    assert receipt["failed_stage_may_have_partial_writes"]
    assert len(gui.writes) == 1
    full = client.call(
        "status", {"execution": failed.data["execution"], "detail": "full"}
    )
    assert full["raw_save"]["path"] == "/actual/raw.h5"
    assert full["analysis"]["status"] == full["post_analysis"]["status"] == "finished"


def test_ge_saves_and_delivers_primary_then_post_without_rerun(tmp_path):
    gui = GeGui()
    modules = gui.publication["tree"]["children"]["modules"]["children"]
    modules["reset"]["ref"] = "old_reset"
    modules["init_pulse"]["ref"] = "old_init"

    def respond(method, params):
        reply = gui(method, params)
        if method == "tab.get_analyze_result":
            reply["summary"] = {
                "centers": {"ground": [0.1, 0.2], "excited": [0.8, 0.9]}
            }
        elif method == "tab.get_post_analyze_result":
            reply["summary"] = {
                "fidelity": 0.98,
                "populations": {"ground": 0.97, "excited": 0.03},
            }
        return reply

    client = make_client(tmp_path, respond)
    try:
        reply = full_execution_reply(
            client,
            skip_writeback(
                client, client.call("singleshot_ge", {"pi_ref": "pi", "shots": 1234})
            ),
        )
        data = reply.data
        assert data["status"] == "finished", data
        assert data["analysis_mode"] == "primary_post"
        assert data["analysis_stage"] == "post"
        assert data["analysis"]["result"]["summary"] == {
            "centers": {"ground": [0.1, 0.2], "excited": [0.8, 0.9]}
        }
        assert data["post_analysis"]["result"]["summary"] == {
            "fidelity": 0.98,
            "populations": {"ground": 0.97, "excited": 0.03},
        }
        before = len(client.transport.sent)
        summary = client.call("status", {"execution": data["execution"]})
        assert len(client.transport.sent) == before
        assert summary["analysis"]["primary"]["details"] == {
            "centers": {"ground": [0.1, 0.2], "excited": [0.8, 0.9]}
        }
        assert summary["analysis"]["post"]["estimates"]["fidelity"] == {
            "value": 0.98,
            "stderr": None,
            "unit": None,
            "quality": None,
        }
        assert summary["analysis"]["post"]["details"] == {
            "populations": {"ground": 0.97, "excited": 0.03},
        }
        assert summary["artifacts"]["analysis"]["trace"]["members"]["image"] == [
            {"path": "/actual/trace.png", "status": "saved"}
        ]
        assert summary["artifacts"]["post_analysis"]["cloud"]["members"]["image"] == [
            {"path": "/actual/cloud.png", "status": "saved"}
        ]
        assert data["analysis"]["saved_images"] == [
            {"figure_name": "trace", "image_path": "/actual/trace.png"}
        ]
        assert data["post_analysis"]["saved_images"] == [
            {"figure_name": "cloud", "image_path": "/actual/cloud.png"}
        ]
        assert data["writeback"]["items"][0]["id"] == "md-1"
        assert data["post_writeback"]["items"][0]["id"] == "classifier"
        assert len(reply.images) == 2
        assert data["actual"]["fields"]["shots"]["value"] == 1234
        assert modules["probe_pulse"]["ref"] == "pi"
        assert modules["reset"]["ref"] is None
        assert modules["init_pulse"]["ref"] is None
        stages = {
            "tab.run_start",
            "tab.save_data",
            "tab.analyze",
            "tab.post_analyze",
            "tab.save_image",
            "tab.get_figure",
            "tab.writeback_preview",
        }
        assert [method for method, _ in gui.calls if method in stages] == [
            "tab.run_start",
            "tab.save_data",
            "tab.analyze",
            "tab.save_image",
            "tab.get_figure",
            "tab.writeback_preview",
            "tab.post_analyze",
            "tab.save_image",
            "tab.get_figure",
            "tab.writeback_preview",
        ]
    finally:
        client.context.session.close()
