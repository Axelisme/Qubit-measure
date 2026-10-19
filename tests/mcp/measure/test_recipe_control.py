"""Recipe tool controls and lifecycle through the recording GUI transport."""

from dataclasses import asdict
from threading import Event, Thread

import pytest
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.measure.session import GuiRpcError

from ._recipe_stage_support import RAW_REPORT
from ._recipe_support import ExecutionGui, recipe_client
from ._recipe_sync_support import cancel_while_rpc_held, start_recipe_at, wait_for_event
from ._support import full_execution_reply, make_client


@pytest.mark.parametrize("held_op", [71, 82, 93])
def test_session_close_drains_recipe_work_and_rejects_new_admission(
    tmp_path, monkeypatch, held_op
):
    gui = ExecutionGui()
    pending = Event()
    disconnected = Event()

    def respond(method, params):
        if (held_op == 82 and method == "tab.save_data") or (
            held_op != 82
            and method == "operation.await"
            and params["operation_id"] == held_op
        ):
            pending.set()
            wait_for_event(disconnected, "Test must release the admitted RPC")
        return gui(method, params)

    client = make_client(tmp_path, respond)
    close_transport = client.transport.close

    def disconnect():
        close_transport()
        disconnected.set()

    monkeypatch.setattr(client.transport, "close", disconnect)
    try:
        initial = start_recipe_at(client, pending, {"frequency_mhz": 6020.0})
        client.context.session.close()
        before = len(client.transport.sent)
        result = client.call(
            "status", {"execution": initial.data["execution"], "detail": "full"}
        )
        assert result["status"] == "cancelled", result
        assert result["phase"] == "terminal"
        assert result["error"] is None
        if held_op == 82:
            assert result["raw_save"] is None
        with pytest.raises(GuiRpcError, match="closed"):
            client.call("pipeline", {"frequency_mhz": 6020.0})
        client.context.session.close()
        assert len(client.transport.sent) == before
    finally:
        disconnected.set()
        client.context.session.close()


def test_lookback_cancel_latches_one_control_and_stops_after_original_run(
    tmp_path, monkeypatch
):
    gui = ExecutionGui()
    stopped = Event()
    running = Event()

    def respond(method, params):
        if method == "operation.await" and params["operation_id"] == 71:
            running.set()
            if stopped.is_set():
                return {"reason": "completed", "status": "cancelled"}
            return {"reason": "timeout"}
        if method == "operation.cancel":
            assert params == {"operation_id": 71}
            stopped.set()
            return {"status": "cancelling"}
        return gui(method, params)

    client = make_client(tmp_path, respond)
    try:
        initial = full_execution_reply(
            client, start_recipe_at(client, running, {"frequency_mhz": 6020.0})
        )
        execution = initial.data["execution"]
        started = client.call("status", {"execution": execution})
        cancelled = client.call("cancel", {"execution": execution})
        assert cancelled.data["cancel_requested"]
        assert cancelled.data["execution"] == execution
        client.call("cancel", {"op": started["run_op"]})
        terminal = full_execution_reply(
            # Released work must reach a handoff; timeout only guards deadlock.
            client,
            client.call("wait", {"execution": execution, "timeout": 30}),
        )
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
    gui = ExecutionGui()
    release_run = Event()
    running = Event()

    def respond(method, params):
        if method == "operation.await" and params["operation_id"] == 71:
            running.set()
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
    try:
        initial = full_execution_reply(
            client, start_recipe_at(client, running, {"frequency_mhz": 6020.0})
        )
        execution = initial.data["execution"]
        started = client.call("status", {"execution": execution})
        early = client.call("finish_early", {"op": started["run_op"]})
        assert early.data == {
            "execution": execution,
            "op": started["op"],
            "run_op": started["run_op"],
            "status": "running",
            "phase": "run",
            "cancel_requested": False,
            "finish_early_requested": True,
            "gui_cancel": {"status": "requested", "error": None},
        }
        client.call("finish_early", {"execution": execution})
        if cancel_after:
            client.call("cancel", {"execution": execution})
        release_run.set()
        result = full_execution_reply(
            # Released work must reach a handoff; timeout only guards deadlock.
            client,
            client.call("wait", {"execution": execution, "timeout": 30}),
        )
        assert result.data["status"] == (
            "awaiting_answer" if expected == "finished" else expected
        ), result.data
        if expected == "finished":
            answered = client.call(
                "answer", {"recipe": execution, "decision": "skipped"}
            )
            assert answered.data["status"] == "finished"
            assert not answered.data["writeback"]["receipts"]
        assert result.data["run_outcome"]["status"] == "cancelled"
        frozen = client.call("status", {"execution": execution, "detail": "full"})
        with pytest.raises(GuiRpcError) as rejected:
            client.call("finish_early", {"execution": execution})
        assert rejected.value.reason == "not_running"
        assert (
            client.call("status", {"execution": execution, "detail": "full"}) == frozen
        )
        assert_partial_run_products(result, expected)
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("operation.cancel") == 1
        assert methods.count("tab.run_start") == 1
        assert methods.count("tab.save_data") == int(expected == "finished")
        assert methods.count("tab.analyze") == int(expected == "finished")
    finally:
        release_run.set()
        client.context.session.close()


def assert_partial_run_products(result: ToolReply, expected: str) -> None:
    if expected == "finished":
        assert result.data["raw_save"]["run_id"] == "run-a"
        assert result.data["raw_save"]["artifacts"]["raw"]["data"]["status"] == "Saved"
        assert result.data["analysis"]["status"] == "finished"
    elif expected == "failed":
        assert result.data["error"]["reason"] == "run_result_unavailable"
        assert result.data["raw_save"] is None


@pytest.fixture(params=[True, False], ids=["saved", "incomplete"])
def admitted_raw_save(tmp_path, monkeypatch, request):
    """Hold a real recipe at its admitted raw-save transport until released."""
    save_succeeds = request.param
    gui = ExecutionGui()
    expected_report = asdict(RAW_REPORT)
    if not save_succeeds:
        artifact = expected_report["artifacts"]["raw"]["data"]
        artifact["status"] = "Not saved"
        for output in artifact["members"]["data"]:
            output["status"] = "failed"
            output["error"] = "disk full"
    saving = Event()
    release_save = Event()

    def respond(method, params):
        if method == "tab.save_data":
            saving.set()
            wait_for_event(release_save, "Test must release the admitted RPC")
        reply = gui(method, params)
        return (
            expected_report
            if method == "tab.save_data" and not save_succeeds
            else reply
        )

    client = make_client(tmp_path, respond)
    try:
        initial = full_execution_reply(
            client, start_recipe_at(client, saving, {"frequency_mhz": 6020.0})
        )
        yield (
            client,
            initial.data["execution"],
            expected_report,
            release_save,
            save_succeeds,
        )
    finally:
        release_save.set()
        client.context.session.close()


def test_lookback_cancel_during_raw_save_has_no_report_until_save_completes(
    admitted_raw_save,
):
    client, execution, _report, _release_save, _save_succeeds = admitted_raw_save
    with pytest.raises(GuiRpcError) as rejected:
        client.call("finish_early", {"execution": execution})
    assert rejected.value.reason == "not_running"
    cancelled = client.call("cancel", {"execution": execution})
    assert cancelled.data["gui_cancel"]["status"] == "not_needed"
    assert cancelled.data["cancel_requested"]
    during_save = client.call("status", {"execution": execution, "detail": "full"})
    assert during_save["raw_save"] is None
    saving_summary = client.call("status", {"execution": execution})
    assert saving_summary["steps"]["raw_save"] is None
    assert saving_summary["artifacts"] == {}


def test_lookback_cancel_during_raw_save_waits_for_the_true_save_outcome(
    admitted_raw_save,
):
    client, execution, expected_report, release_save, save_succeeds = admitted_raw_save
    cancelled = client.call("cancel", {"execution": execution})
    assert cancelled.data["gui_cancel"]["status"] == "not_needed"
    assert cancelled.data["cancel_requested"]
    release_save.set()
    final = full_execution_reply(
        # Released work must reach a handoff; timeout only guards deadlock.
        client,
        client.call("wait", {"execution": execution, "timeout": 30}),
    )
    assert not final.is_error  # Successful query, even when execution failed.
    assert final.data["status"] == "cancelled"
    assert final.data["raw_save"] == expected_report
    assert final.data["error"] is None
    summary = client.call("status", {"execution": execution})
    assert summary["status"] == final.data["status"]
    assert not summary["cancel_requested"]
    assert summary["steps"]["raw_save"]["status"] == (
        "saved" if save_succeeds else "incomplete"
    )
    assert summary["artifacts"]["raw"] == expected_report["artifacts"]["raw"]
    assert summary["error"] == final.data["error"]
    methods = [method for method, _ in client.transport.sent]
    assert methods.count("tab.save_data") == 1
    assert "operation.cancel" not in methods
    assert "tab.analyze" not in methods


@pytest.mark.parametrize("outcome", ["cancelled", "failed"])
def test_recipe_cancel_after_analysis_admission_joins_the_true_outcome(
    tmp_path, outcome
):
    # Replaces the retired pre-handoff timing projection, not its scheduling.
    gui = ExecutionGui()
    stopped = Event()

    def respond(method, params):
        if method == "tab.analyze":
            return {**gui(method, params), "interactive": True}
        if method == "tab.interact":
            return {
                "operation_id": 93,
                "figure": None,
                "state": {},
                "commands": [{"name": "done"}],
            }
        if method == "operation.await" and params["operation_id"] == 93:
            if stopped.is_set():
                return {
                    "reason": "completed",
                    "status": outcome,
                    "error": "Analysis failed after cancellation",
                }
            return {"reason": "timeout"}
        if method == "operation.cancel":
            assert params == {"operation_id": 93}
            stopped.set()
            return {"status": "cancelling"}
        return gui(method, params)

    client = make_client(tmp_path, respond)
    try:
        initial = full_execution_reply(
            client, client.call("pipeline", {"frequency_mhz": 6020.0})
        )
        assert initial.data["status"] == "interactive"
        execution = initial.data["execution"]
        (analysis_receipt,) = client.context.session.executions.snapshots()
        cancelled = client.call("cancel", {"execution": execution})
        assert cancelled.data["cancel_requested"]
        assert cancelled.data["gui_cancel"]["status"] == "requested"
        terminal = full_execution_reply(
            # Released work must reach a handoff; timeout only guards deadlock.
            client,
            client.call("wait", {"execution": execution, "timeout": 30}),
        )
        assert terminal.data["phase"] == "terminal"
        assert terminal.data["status"] == outcome
        assert terminal.data["analysis"]["execution"] == analysis_receipt.execution
        assert terminal.data["analysis"]["op"] == analysis_receipt.op
        assert terminal.data["analysis"]["status"] == outcome
        assert terminal.data["analysis"]["cancel_requested"]
        assert terminal.data["analysis"]["operation_outcome"]["status"] == outcome
        assert terminal.data["raw_save"] == asdict(RAW_REPORT)
        if outcome == "failed":
            assert terminal.data["error"]["reason"] == "analysis_failed"
        methods = [method for method, _ in client.transport.sent]
        assert all(
            methods.count(method) == 1
            for method in ("tab.run_start", "tab.analyze", "operation.cancel")
        )
        assert not {
            "tab.get_analyze_result",
            "tab.save_artifacts",
            "tab.get_figure",
            "tab.writeback_preview",
        }.intersection(methods)
    finally:
        stopped.set()
        client.context.session.close()


@pytest.mark.parametrize("selector", ["execution", "run_op", "op"])
def test_recipe_cancel_delegates_to_the_existing_analysis_owner(
    tmp_path, monkeypatch, selector
):
    gui = ExecutionGui()
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
    try:
        initial = full_execution_reply(
            client, start_recipe_at(client, analyzing, {"frequency_mhz": 6020.0})
        )
        execution = initial.data["execution"]
        state = client.call("status", {"execution": execution})
        key = "execution" if selector == "execution" else "op"
        arguments = {key: state[selector]}
        with pytest.raises(GuiRpcError) as rejected:
            client.call("finish_early", arguments)
        assert rejected.value.reason == "not_running"
        assert not stopped.is_set()
        cancelled = client.call("cancel", arguments)
        assert cancelled.data["cancel_requested"]
        assert cancelled.data["gui_cancel"]["status"] == "requested"
        client.call("cancel", {"execution": execution})
        terminal = full_execution_reply(
            # Released work must reach a handoff; timeout only guards deadlock.
            client,
            client.call("wait", {"execution": execution, "timeout": 30}),
        )
        assert terminal.data["status"] == "cancelled"
        assert terminal.data["analysis"]["status"] == "cancelled"
        assert terminal.data["analysis"]["cancel_requested"]
        assert terminal.data["raw_save"] == asdict(RAW_REPORT)
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
        ("tab.snapshot", "tab.analyze"),
        ("tab.save_data", "tab.analyze"),
        ("tab.get_figure", "tab.writeback_preview"),
    ],
)
def test_recipe_cancel_blocks_the_next_phase_while_an_admitted_read_finishes(
    tmp_path, monkeypatch, pending_method, forbidden_method
):
    gui = ExecutionGui()
    pending = Event()
    release = Event()

    def respond(method, params):
        at_boundary = method == pending_method and (
            (method == "tab.snapshot" and gui.ran)
            or method == "tab.save_data"
            or method == "tab.get_figure"
        )
        if at_boundary:
            pending.set()
            wait_for_event(release, "Test must release the admitted RPC")
        if method == "operation.cancel":
            return {"status": "finished"}
        return gui(method, params)

    client = make_client(tmp_path, respond)
    try:
        initial = full_execution_reply(
            client, start_recipe_at(client, pending, {"frequency_mhz": 6020.0})
        )
        execution = initial.data["execution"]
        control_reply = cancel_while_rpc_held(client, execution, release)
        assert control_reply.data["execution"] == execution
        assert not control_reply.is_error
        # RPC released, cancel caller joined: worker can finish; 30s guards deadlock.
        terminal = full_execution_reply(
            client, client.call("wait", {"execution": execution, "timeout": 30})
        )
        assert terminal.data["status"] == "cancelled"
        assert not terminal.data["cancel_requested"]
        methods = [method for method, _ in client.transport.sent]
        assert forbidden_method not in methods
        assert methods.count("tab.run_start") == 1
        assert terminal.data["raw_save"] == asdict(RAW_REPORT)
    finally:
        release.set()
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
    gui = ExecutionGui()
    pending = Event()
    release = Event()
    stopped = Event()

    def respond(method, params):
        if method == method_pending:
            pending.set()
            wait_for_event(release, "Test must release the admitted RPC")
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
    try:
        initial = full_execution_reply(
            client, start_recipe_at(client, pending, {"frequency_mhz": 6020.0})
        )

        execution = initial.data["execution"]
        for _ in range(2):
            reply = client.call(control, {"execution": execution})
            assert reply.data[f"{control}_requested"]
        assert not stopped.is_set()
        release.set()
        terminal = full_execution_reply(
            # Released work must reach a handoff; timeout only guards deadlock.
            client,
            client.call("wait", {"execution": execution, "timeout": 30}),
        )
        assert stopped.is_set()
        assert terminal.data["status"] == (
            "awaiting_answer" if control == "finish_early" else "cancelled"
        )
        if control == "finish_early":
            answered = client.call(
                "answer", {"recipe": execution, "decision": "skipped"}
            )
            assert answered.data["status"] == "finished"
            assert not answered.data["writeback"]["receipts"]
        methods = [method for method, _ in client.transport.sent]
        assert methods.count(method_pending) == 1
        assert methods.count("operation.cancel") == 1
        if method_pending == "tab.analyze":
            assert terminal.data["analysis"]["cancel_requested"]
            assert terminal.data["raw_save"] == asdict(RAW_REPORT)
            assert "tab.get_analyze_result" not in methods
        elif control == "cancel":
            assert "tab.save_data" not in methods
        else:
            assert terminal.data["raw_save"] == asdict(RAW_REPORT)
    finally:
        release.set()
        stopped.set()
        client.context.session.close()


def test_recipe_cancel_during_preparation_keeps_the_missing_parameter_handoff(
    tmp_path, monkeypatch
):
    gui = ExecutionGui()
    pending = Event()
    release = Event()

    def respond(method, params):
        if method == "tab.select_recipe_module" and params["path"] == [
            "modules",
            "reset",
        ]:
            assert params["action"] == "disable"
            pending.set()
            wait_for_event(release, "Test must release the admitted RPC")
        return gui(method, params)

    client = make_client(tmp_path, respond)
    try:
        initial = start_recipe_at(client, pending, {})
        execution = initial.data["execution"]
        cancelled = client.call("cancel", {"execution": execution})
        assert cancelled.data["cancel_requested"]
        release.set()
        # Released work must reach a handoff; timeout only guards deadlock.
        terminal = client.call("wait", {"execution": execution, "timeout": 30})
        assert terminal.data["status"] == "needs_parameters"
        assert terminal.data["cancel_requested"]
        assert terminal.data["tab"] == "t"
        assert [item["parameter"] for item in terminal.data["missing"]] == [
            "frequency_mhz"
        ]
        assert not gui.ran
    finally:
        release.set()
        client.context.session.close()


def test_recipe_worker_start_failure_returns_a_terminal_receipt_and_closes_safely(
    tmp_path, monkeypatch
):
    def fail_start(self):
        raise RuntimeError("no thread resources")

    with recipe_client(tmp_path, ExecutionGui()) as client:
        monkeypatch.setattr(Thread, "start", fail_start)
        reply = client.call("pipeline", {"frequency_mhz": 6020.0})
        assert reply.is_error
        assert reply.data["status"] == "failed"
        assert reply.data["error"]["phase"] == "preparing"
        assert reply.data["error"]["reason"] == "recipe_worker_start_failed"
        assert reply.data["error"]["message"] == "no thread resources"
        execution = reply.data["execution"]
        state = client.call("status", {"execution": execution})
        assert state["status"] == "failed"
        cancelled = client.call("cancel", {"execution": execution})
        assert cancelled.data["status"] == "failed"
        assert not cancelled.data["cancel_requested"]
        assert "tab.new" not in [method for method, _ in client.transport.sent]


def test_recipe_cancel_during_admitted_writeback_preserves_result_and_intent(
    tmp_path, monkeypatch
):
    gui = ExecutionGui()
    reading = Event()
    release = Event()

    def respond(method, params):
        if method == "tab.writeback_preview":
            reading.set()
            wait_for_event(release, "Test must release the admitted RPC")
        if method == "operation.cancel":
            assert params == {"operation_id": 93}
            return {"status": "finished"}
        return gui(method, params)

    client = make_client(tmp_path, respond)
    try:
        initial = full_execution_reply(
            client, start_recipe_at(client, reading, {"frequency_mhz": 6020.0})
        )
        execution = initial.data["execution"]
        control_reply = cancel_while_rpc_held(client, execution, release)
        assert control_reply.data["execution"] == execution
        assert control_reply.data["gui_cancel"]["status"] == "not_needed"
        # RPC released, cancel caller joined: worker can finish; 30s guards deadlock.
        terminal = full_execution_reply(
            client, client.call("wait", {"execution": execution, "timeout": 30})
        )
        assert terminal.data["status"] == "cancelled"
        assert not terminal.data["cancel_requested"]
        assert terminal.data["writeback"]["items"][0]["proposed"] == 0.24
        assert terminal.data["raw_save"] == asdict(RAW_REPORT)
        assert terminal.data["analysis"]["operation_outcome"]["status"] == "finished"
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.writeback_preview") == 1
        assert methods.count("operation.cancel") == 1
    finally:
        release.set()
        client.context.session.close()


@pytest.mark.parametrize("partial_data", [False, True])
def test_manual_gui_run_cancel_never_starts_the_recipe_save_pipeline(
    tmp_path, partial_data
):
    gui = ExecutionGui()

    def respond(method, params):
        if method == "operation.await" and params["operation_id"] == 71:
            gui.ran = partial_data
            return {"reason": "completed", "status": "cancelled"}
        return gui(method, params)

    with recipe_client(tmp_path, respond) as client:
        reply = full_execution_reply(
            client, client.call("pipeline", {"frequency_mhz": 6000.0})
        )
        assert reply.data["status"] == "cancelled"
        assert not reply.data["cancel_requested"]
        assert not reply.data["finish_early_requested"]
        assert reply.data["run_outcome"]["status"] == "cancelled"
        assert reply.data["raw_save"] is None
        methods = [method for method, _ in client.transport.sent]
        assert "tab.save_data" not in methods
        assert "tab.analyze" not in methods
        assert "operation.cancel" not in methods
