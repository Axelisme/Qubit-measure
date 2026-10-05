"""Captured recipe reply projection and failure-prefix contracts."""

from copy import deepcopy
from threading import Event
from typing import Any

import pytest
from zcu_tools.mcp.core.bridge import GuiTransportTimeoutError
from zcu_tools.mcp.measure import tools_recipes

from ._recipe_support import PNG, LookbackGui, recipe_client
from ._support import full_execution_reply, make_client


@pytest.mark.parametrize("outcome", ["awaiting_answer", "missing", "failed"])
def test_recipe_initial_wait_and_status_share_the_same_summary(tmp_path, outcome):
    gui = LookbackGui()

    def respond(method, params):
        if outcome == "failed" and method == "operation.await":
            return {"reason": "completed", "status": "failed", "error": "Run failed"}
        return gui(method, params)

    with recipe_client(tmp_path, respond) as client:
        initial = client.call(
            "lookback", {} if outcome == "missing" else {"frequency_mhz": 6020.0}
        )
        key = initial.data["execution"]
        before = len(client.transport.sent)
        summary = client.call("status", {"execution": key})
        waited = client.call("wait", {"execution": key, "timeout": 0})
        full = client.call("status", {"execution": key, "detail": "full"})
        assert {k: v for k, v in initial.data.items() if k != "elapsed_s"} == summary
        assert {k: v for k, v in waited.data.items() if k != "elapsed_s"} == summary
        assert len(client.transport.sent) == before
        assert summary["status"] == (
            "needs_parameters" if outcome == "missing" else outcome
        )
        assert summary["run_id"] is None
        assert summary["steps"]["run"]["status"] == (
            "not_started"
            if outcome == "missing"
            else "finished"
            if outcome == "awaiting_answer"
            else outcome
        )
        assert summary["previews"] == {
            "run": [],
            "primary": [full["analysis"]["figure"]]
            if outcome == "awaiting_answer"
            else [],
            "post": [],
        }
        if outcome == "awaiting_answer":
            assert summary["question_items"] == ("trigger_offset",)
            assert not summary["writeback"]["receipts"]
            assert initial.images
            assert waited.images
            assert summary["artifacts"]["raw"]["data"]["members"]["data"] == [
                {"path": "/actual/raw.h5", "status": "saved"}
            ]
            assert (
                full["actual"]["publication"]["cfg_ref"] == summary["actual"]["cfg_ref"]
            )
        elif outcome == "missing":
            assert summary["missing"] == full["missing"]
        else:
            assert summary["error"] == full["error"]


def test_writeback_destination_summary_keeps_context_and_project_identity(tmp_path):
    gui = LookbackGui()
    destination = {
        "active_label": "sample",
        "has_active_context": True,
        "chip_name": "chip",
        "qub_name": "q",
        "res_name": "r",
        "database_path": "/resolved/experiment/data",
        "result_dir": "/resolved/experiment/result",
    }

    def respond(method, params):
        response = gui(method, params)
        if method == "tab.writeback_preview":
            response["destination_context"] = destination
        return response

    with recipe_client(tmp_path, respond) as client:
        completed = client.call("lookback", {"frequency_mhz": 6020.0})
        assert completed.data["status"] == "awaiting_answer", completed.data
        key = completed.data["execution"]
        summary = client.call("status", {"execution": key})
        full = client.call("status", {"execution": key, "detail": "full"})
        assert summary["writeback"]["destination"] == {
            "context": {"active_label": "sample"},
            "project": {"chip_name": "chip", "qub_name": "q", "res_name": "r"},
        }
        assert full["writeback"]["destination_context"] == destination
        assert summary["artifacts"]["raw"]["data"]["members"]["data"] == [
            {"path": "/actual/raw.h5", "status": "saved"}
        ]


def test_pulse_readout_candidate_summary_keeps_nested_values_and_captured_full(
    tmp_path,
):
    gui = LookbackGui()
    current: dict[str, Any] = {
        "type": "readout/pulse",
        "cloned_from": "calibrated_readout",
        "pulse_cfg": {
            "type": "pulse",
            "freq": 5000.0,
            "gain": 0.1,
            "phase": 0.0,
            "waveform": {"style": "const", "length": 2.0},
            "ch": 0,
            "nqz": 1,
        },
        "ro_cfg": {
            "type": "readout/direct",
            "ro_freq": 5000.0,
            "ro_length": 2.0,
            "trig_offset": 0.1,
            "ro_ch": 0,
            "gen_ch": 0,
        },
    }
    proposed = deepcopy(current)
    proposed["pulse_cfg"].update(freq=5020.0, gain=0.15)
    proposed["pulse_cfg"]["waveform"]["length"] = 2.5
    proposed["ro_cfg"].update(ro_freq=5020.0, ro_length=2.5, trig_offset=0.2)

    def respond(method, params):
        response = gui(method, params)
        if method == "tab.writeback_preview":
            response["items"] = [
                {
                    "id": "ml-readout",
                    "kind": "module",
                    "target_name": "readout_rf",
                    "proposed": proposed,
                    "current": current,
                }
            ]
        return response

    with recipe_client(tmp_path, respond) as client:
        completed = client.call("lookback", {"frequency_mhz": 6020.0})
        assert completed.data["status"] == "awaiting_answer", completed.data
        sent = list(client.transport.sent)
        summary = client.call("status", {"execution": completed.data["execution"]})
        full = client.call(
            "status", {"execution": completed.data["execution"], "detail": "full"}
        )
        assert client.transport.sent == sent
        candidate = summary["writeback"]["stages"]["primary"][0]
        assert candidate == completed.data["writeback"]["stages"]["primary"][0]
        assert candidate["target"] == "readout_rf"
        assert candidate["cfg_ref"] == summary["actual"]["cfg_ref"]
        for source, expected in (("current", current), ("proposed", proposed)):
            projected = candidate[source]
            assert projected["type"] == "readout/pulse"
            assert projected["cloned_from"] == "calibrated_readout"
            assert projected["pulse_cfg"] == {
                key: expected["pulse_cfg"][key]
                for key in ("type", "freq", "gain", "phase", "waveform")
            }
            assert projected["ro_cfg"] == {
                key: expected["ro_cfg"][key]
                for key in ("type", "ro_freq", "ro_length", "trig_offset")
            }
            assert full["writeback"]["items"][0][source] == expected
        assert set(candidate["changes"]) == {
            "pulse_cfg.freq",
            "pulse_cfg.gain",
            "pulse_cfg.waveform.length",
            "ro_cfg.ro_freq",
            "ro_cfg.ro_length",
            "ro_cfg.trig_offset",
        }


def test_module_candidate_summary_keeps_source_changes_and_full_proposal(tmp_path):
    gui = LookbackGui()
    current: dict[str, Any] = {
        "type": "pulse",
        "freq": 6100.0,
        "gain": 0.1,
        "phase": 0.0,
        "waveform": {"style": "const", "length": 0.1},
        "ch": 0,
        "nqz": 2,
        "pre_delay": 0.0,
        "post_delay": 0.0,
        "cloned_from": "calibrated_drive",
    }
    proposed: dict[str, Any] = {**deepcopy(current), "gain": 0.15}
    proposed["waveform"]["length"] = 0.24

    def respond(method, params):
        response = gui(method, params)
        if method == "tab.writeback_preview":
            response["items"] = [
                {
                    "id": "ml-1",
                    "kind": "module",
                    "target_name": "pi_len",
                    "selected": False,
                    "proposed": proposed,
                    "current": current,
                }
            ]
        return response

    with recipe_client(tmp_path, respond) as client:
        completed = client.call("lookback", {"frequency_mhz": 6020.0})
        assert completed.data["status"] == "awaiting_answer", completed.data
        key = completed.data["execution"]
        summary = client.call("status", {"execution": key})
        full = client.call("status", {"execution": key, "detail": "full"})
        candidate = summary["writeback"]["stages"]["primary"][0]
        assert candidate["kind"] == "module"
        assert candidate["target"] == "pi_len"
        assert candidate["resolved_target"] is None
        assert full["writeback"]["items"][0]["selected"] is False
        assert candidate["cfg_ref"] == summary["actual"]["cfg_ref"]
        assert candidate["proposed"] == {
            "type": "pulse",
            "freq": 6100.0,
            "gain": 0.15,
            "phase": 0.0,
            "waveform": {"style": "const", "length": 0.24},
            "cloned_from": "calibrated_drive",
        }
        assert candidate["current"]["cloned_from"] == "calibrated_drive"
        assert set(candidate["changes"]) == {"gain", "waveform.length"}
        assert full["writeback"]["items"][0]["proposed"] == proposed
        assert full["writeback"]["items"][0]["current"] == current


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
def test_run_start_failure_preserves_confirmed_rejection_or_ambiguity(
    tmp_path, monkeypatch, receipt, reason, step_status
):
    gui = LookbackGui()
    pending = Event()
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    with recipe_client(tmp_path, gui) as client:
        if receipt in {"handler_timeout", "internal", "stale", "busy"}:
            code = {
                "handler_timeout": "timeout",
                "internal": "internal",
                "stale": "precondition_failed",
                "busy": "busy",
            }[receipt]
            client.transport.replies["tab.run_start"] = {
                "ok": False,
                "error": {
                    "code": code,
                    "reason": None if receipt == "handler_timeout" else reason,
                    "message": "Run start did not provide a handle",
                },
            }
        else:
            send_line = client.transport.send_line

            def send(payload):
                if payload["method"] != "tab.run_start":
                    return send_line(payload)
                client.transport.sent.append((payload["method"], payload["params"]))
                gui("tab.run_start", payload["params"])
                pending.set()
                if receipt == "timeout":
                    raise GuiTransportTimeoutError("tab.run_start", 0.01)
                return None

            monkeypatch.setattr(client.transport, "send_line", send)
        initial = client.call("lookback", {"frequency_mhz": 6020.0})
        execution = initial.data["execution"]
        if receipt == "lost":
            assert pending.wait(1)
            client.transport.close()
            assert client.transport.on_closed is not None
            client.transport.on_closed(None)
        completed = client.call("wait", {"execution": execution, "timeout": 2})
        assert completed.data["status"] == "failed", completed.data
        before = len(client.transport.sent)
        summary = client.call("status", {"execution": execution})
        full = client.call("status", {"execution": execution, "detail": "full"})
        assert summary["steps"]["run"]["status"] == step_status
        assert full["run_start"]["status"] == step_status
        assert summary["error"]["reason"] == reason
        assert summary["run_op"] is None
        assert summary["actual"]["cfg_ref"] == full["actual"]["publication"]["cfg_ref"]
        assert summary["steps"]["raw_save"]["status"] == "not_started"
        assert len(client.transport.sent) == before
        assert [method for method, _ in client.transport.sent].count(
            "tab.run_start"
        ) == 1


def test_unconfirmed_run_receipt_stays_unknown_until_the_original_start_returns(
    tmp_path, monkeypatch
):
    gui = LookbackGui()
    pending = Event()
    release = Event()

    def respond(method, params):
        if method == "tab.run_start":
            pending.set()
            assert release.wait(2)
        return gui(method, params)

    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    with recipe_client(tmp_path, respond) as client:
        try:
            initial = client.call("lookback", {"frequency_mhz": 6020.0})
            assert pending.wait(1)
            execution = initial.data["execution"]
            before = len(client.transport.sent)
            summary = client.call("status", {"execution": execution})
            full = client.call("status", {"execution": execution, "detail": "full"})
            assert summary["steps"]["run"]["status"] == "unknown"
            assert summary["run_op"] is None
            assert full["run_start"]["status"] == "unknown"
            assert len(client.transport.sent) == before
            release.set()
            completed = client.call("wait", {"execution": execution, "timeout": 2})
            assert completed.data["status"] == "awaiting_answer", completed.data
            confirmed = client.call("status", {"execution": execution})
            assert confirmed["steps"]["run"] == {
                "status": "finished",
                "reason": "completed",
            }
            assert confirmed["artifacts"]["raw"]["data"]["members"]["data"] == [
                {"path": "/actual/raw.h5", "status": "saved"}
            ]
            assert [method for method, _ in client.transport.sent].count(
                "tab.run_start"
            ) == 1
        finally:
            release.set()


def test_summary_status_reports_the_finished_recipe_facts(tmp_path):
    gui = LookbackGui()

    def respond(method, params):
        response = gui(method, params)
        if method == "tab.writeback_preview":
            response["items"] = [
                {
                    "id": "md-1",
                    "kind": "metadict",
                    "target_name": "trigger_offset",
                    "selected": False,
                    "proposed": 0.24,
                    "current": 0.1,
                }
            ]
        return response

    with recipe_client(tmp_path, respond) as client:
        question = client.call("lookback", {"frequency_mhz": 6020.0, "rounds": 7})
        assert question.data["status"] == "awaiting_answer", question.data
        execution = question.data["execution"]
        client.call("answer", {"recipe": execution, "decision": "skipped"})
        completed = client.call("wait", {"execution": execution, "timeout": 2})
        assert completed.data["status"] == "finished", completed.data
        before = len(client.transport.sent)
        summary = client.call("status", {"execution": execution})
        full = client.call("status", {"execution": execution, "detail": "full"})
        assert summary["execution"] == full["execution"] == execution
        assert summary["run_id"] is None
        assert summary["actual"]["cfg_ref"] == full["actual"]["cfg_ref"]
        assert summary["actual"]["parameters"]["frequency_mhz"] == {
            "value": 6020.0,
            "source": "frequency_mhz",
            "unit": "MHz",
        }
        assert summary["actual"]["parameters"]["rounds"] == {
            "value": 7,
            "source": "rounds",
        }
        assert summary["actual"]["modules"]["reset"] == {
            "value": None,
            "source": "disabled",
        }
        assert summary["steps"]["run"] == {"status": "finished", "reason": "completed"}
        assert summary["steps"]["raw_save"]["status"] == "saved"
        assert summary["steps"]["analysis"]["primary"]["status"] == "finished"
        assert summary["steps"]["analysis_save"]["primary"]["status"] == "saved"
        assert summary["analysis"]["primary"]["params"] == {"threshold": 0.5}
        assert summary["analysis"]["primary"]["details"] == {"offset": 0.24}
        assert summary["artifacts"]["raw"]["data"] == {
            "status": "saved",
            "lifetime": "persistent",
            "members": {"data": [{"path": "/actual/raw.h5", "status": "saved"}]},
        }
        assert summary["artifacts"]["analysis"]["trace"]["members"] == {
            "image": [{"path": "/actual/trace.png", "status": "saved"}]
        }
        candidate = summary["writeback"]["stages"]["primary"][0]
        assert candidate == {
            "id": "md-1",
            "kind": "parameter",
            "target": "trigger_offset",
            "resolved_target": None,
            "proposed": 0.24,
            "current": 0.1,
        }
        assert summary["writeback"]["destination"] == {
            "context": {"active_label": "sample"}
        }
        assert not summary["missing"]
        assert not summary["invalid"]
        assert summary["question_items"] is None
        assert not summary["writeback"]["receipts"]
        assert summary["error"] is None
        assert len(client.transport.sent) == before


@pytest.mark.parametrize(
    "wire_op,phase,raw_status",
    [(71, "run", "not_started"), (82, "raw_save", "failed"), (93, "analysis", "saved")],
)
@pytest.mark.parametrize(
    "outcome",
    [
        {"reason": "completed", "status": "running"},
        {"reason": "unexpected", "status": "finished"},
        {"reason": "completed"},
    ],
)
def test_malformed_completion_fails_at_its_owner_without_losing_run_capture(
    tmp_path, wire_op, phase, raw_status, outcome
):
    gui = LookbackGui()

    def respond(method, params):
        if method == "operation.await" and params["operation_id"] == wire_op:
            return outcome
        return gui(method, params)

    with recipe_client(tmp_path, respond) as client:
        reply = full_execution_reply(
            client, client.call("lookback", {"frequency_mhz": 6020.0})
        )
        assert reply.is_error
        assert reply.data["status"] == "failed"
        assert reply.data["error"]["reason"] == "incompatible_wire"
        assert reply.data["error"]["phase"] == phase
        assert reply.data["actual"]["fields"]["modules.readout.pulse_cfg.freq"] == {
            "value": 6020.0,
            "input": gui.publication["tree"]["children"]["modules"]["children"][
                "readout"
            ]["children"]["pulse_cfg"]["children"]["freq"]["input"],
            "source": "frequency_mhz",
        }
        assert reply.data["raw_save"]["status"] == raw_status
        assert reply.data["raw_save"]["path"] == (
            "/actual/raw.h5" if wire_op == 93 else None
        )
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == 1
        assert methods.count("tab.save_data") == (0 if wire_op == 71 else 1)
        assert methods.count("tab.analyze") == (1 if wire_op == 93 else 0)
        assert "tab.get_analyze_result" not in methods


def test_run_capture_keeps_calibrated_raw_input_after_live_cfg_changes(tmp_path):
    gui = LookbackGui()
    gui.md["r_f"] = 6120.0

    def respond(method, params):
        response = gui(method, params)
        if method == "tab.run_start":
            gui.md["r_f"] = 7000.0
            readout = gui.publication["tree"]["children"]["modules"]["children"][
                "readout"
            ]
            state = readout["children"]["pulse_cfg"]["children"]["freq"]["input"]
            state.update(raw="another_source", resolved=7000.0)
        return response

    with recipe_client(tmp_path, respond) as client:
        completed = client.call("lookback", {})
        assert completed.data["status"] == "awaiting_answer", completed.data
        before = len(client.transport.sent)
        full = client.call(
            "status", {"execution": completed.data["execution"], "detail": "full"}
        )
        frequency = full["actual"]["fields"]["modules.readout.pulse_cfg.freq"]
        assert frequency["value"] == 6120.0
        assert frequency["source"] == "r_f"
        assert frequency["input"]["raw"] == "r_f"
        assert frequency["input"]["mode"] == "expression"
        assert frequency["input"]["resolved"] == 6120.0
        assert (
            full["actual"]["publication"]["tree"]["children"]["modules"]["children"][
                "readout"
            ]["children"]["pulse_cfg"]["children"]["freq"]["input"]
            == (frequency["input"])
        )
        assert len(client.transport.sent) == before


def test_full_query_keeps_the_publication_used_before_run(tmp_path):
    gui = LookbackGui()
    gui.publication["source_basis"] = [{"label": "sample", "revision": 3}]
    captured: dict[str, Any] = {}

    def respond(method, params):
        response = gui(method, params)
        if method == "tab.run_start":
            captured.update(deepcopy(gui.publication))
            gui.publication["source_basis"][0]["revision"] = 99
            gui.publication["cfg_ref"]["revision"] = "99"
            gui.publication["tree"]["children"]["rounds"] = {
                "kind": "scalar",
                "input": {"resolved": 999},
            }
        return response

    with recipe_client(tmp_path, respond) as client:
        completed = client.call("lookback", {"frequency_mhz": 6020.0, "rounds": 7})
        assert completed.data["status"] == "awaiting_answer", completed.data
        execution = completed.data["execution"]
        before = len(client.transport.sent)
        full = client.call("status", {"execution": execution, "detail": "full"})
        assert full["actual"]["publication"] == captured
        assert full["actual"]["source_basis"] == [{"label": "sample", "revision": 3}]
        assert full["actual"]["fields"]["rounds"]["value"] == 7
        assert full["actual"]["fields"]["rounds"]["source"] == "rounds"
        assert (
            full["actual"]["publication"]["tree"]["children"]["rounds"]["input"][
                "resolved"
            ]
            == 7
        )
        full["actual"]["publication"]["tree"]["children"].clear()
        full["actual"]["fields"]["rounds"]["value"] = -1
        full["actual"]["source_basis"].clear()
        repeated = client.call("status", {"execution": execution, "detail": "full"})
        assert repeated["actual"]["publication"] == captured
        assert repeated["actual"]["fields"]["rounds"]["value"] == 7
        assert repeated["actual"]["source_basis"] == captured["source_basis"]
        assert len(client.transport.sent) == before


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
        initial = full_execution_reply(
            client, client.call("lookback", {"frequency_mhz": 6000.0})
        )
        assert pending.wait(1)
        client.transport.close()
        assert client.transport.on_closed is not None
        client.transport.on_closed(None)
        execution = initial.data["execution"]
        terminal = full_execution_reply(
            client, client.call("wait", {"execution": execution, "timeout": 2})
        )
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
        reply = full_execution_reply(
            client, client.call("lookback", {"frequency_mhz": 6000.0})
        )
        data = reply.data
        before = len(client.transport.sent)
        summary = client.call("status", {"execution": data["execution"]})
        assert len(client.transport.sent) == before
        assert summary["status"] == data["status"] == "failed", data
        assert summary["error"] == data["error"]
        assert summary["steps"]["raw_save"]["status"] == raw_status
        assert summary["artifacts"]["raw"]["data"]["members"]["data"] == (
            [{"path": "/actual/raw.h5", "status": "saved"}]
            if raw_status == "saved"
            else [{"path": "/actual/raw.h5", "status": "reserved"}]
            if failure == "raw_finish"
            else []
        )
        if failure == "writeback":
            assert summary["artifacts"]["analysis"]["trace"]["members"]["image"] == [
                {"path": "/actual/trace.png", "status": "saved"}
            ]
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
            assert reply.images[0].data == PNG
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
