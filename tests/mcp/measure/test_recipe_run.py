"""Author Run/source capture and raw-save contracts through the GUI transport."""

from copy import deepcopy
from threading import Condition, Event

import pytest
from zcu_tools.mcp.measure.recipe import (
    RecipeRun,
    RecipeSession,
)
from zcu_tools.mcp.measure.session import GuiRpcError

from ._recipe_support import LookbackGui, recipe_client, scalar


def test_run_starts_immediately_with_detached_conditions_and_original_ref(tmp_path):
    gui = LookbackGui()
    gui.md["r_f"] = 6100.0
    gui.publication["source_basis"] = [{"key": "r_f", "revision": "7"}]
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        tab.set("rounds", 7)
        tab.disable_library("modules.reset")
        tab.set_frequency(
            "modules.readout", calibration="resonator", required="frequency_mhz"
        )
        expected = deepcopy(gui.publication["cfg_ref"])
        operation = tab.run()
        assert gui.ran
        assert not any(
            method == "operation.await" for method, _ in client.transport.sent
        )
        before = operation.snapshot()
        assert before is not None and before.start_status == "running"
        assert before.actual["cfg_ref"] == expected
        assert before.actual["fields"]["rounds"] == {
            "value": 7,
            "source": "explicit",
            "input": scalar(7)["input"],
        }
        assert before.actual["fields"]["modules.reset"] == {
            "value": None,
            "source": "disabled",
        }
        expected_frequency = scalar(6100.0)["input"]
        expected_frequency.update(mode="expression", raw="r_f")
        assert before.actual["fields"]["modules.readout.pulse_cfg.freq"] == {
            "value": 6100.0,
            "source": "r_f",
            "input": expected_frequency,
        }
        assert before.actual["source_basis"] == [{"key": "r_f", "revision": "7"}]
        gui.publication["source_basis"].clear()
        tab.set("rounds", 11)
        before.actual["fields"]["rounds"]["value"] = -1
        run, status = operation.complete_in_current_worker()
        assert isinstance(run, RecipeRun) and status == "completed"
        captured = run.snapshot()
        assert captured is not None
        assert captured.actual["fields"]["rounds"]["value"] == 7
        assert captured.actual["cfg_ref"] == expected
        assert captured.actual["source_basis"] == [{"key": "r_f", "revision": "7"}]
        assert captured.result_state == {
            "available": True,
            "revision": 1,
            "source_operation_id": 71,
        }
        assert captured.outcome == {"reason": "completed", "status": "finished"}
        receipt = run.save_raw()
        assert receipt.status == "saved" and receipt.path == "/actual/raw.h5"
        captured = operation.snapshot()
        assert captured is not None and captured.raw_save == receipt
        assert (
            "tab.run_start",
            {"tab_id": "t", "expected": expected},
        ) in client.transport.sent
        assert (
            "tab.save_data",
            {"tab_id": "t", "run_operation_id": 71},
        ) in client.transport.sent


@pytest.mark.parametrize(
    "available,source,closed_tab",
    [(True, 71, False), (False, 71, False), (False, None, False), (False, None, True)],
)
def test_cancelled_run_delivers_partial_handle_and_only_saves_available_data(
    tmp_path, available, source, closed_tab
):
    gui = LookbackGui()

    def respond(method, params):
        reply = gui(method, params)
        if method == "operation.await" and params["operation_id"] == 71:
            reply["status"] = "cancelled"
        if method == "tab.snapshot" and gui.ran:
            if closed_tab:
                reply["tabs"] = []
            else:
                reply["tabs"][0]["result_state"].update(
                    available=available, source_operation_id=source
                )
        return reply

    with recipe_client(tmp_path, respond) as client:
        operation = RecipeSession(client.context).open_tab("lookback").run()
        run, status = operation.complete_in_current_worker()
        assert isinstance(run, RecipeRun) and status == "cancelled"
        if available:
            assert run.save_raw().path == "/actual/raw.h5"
        else:
            with pytest.raises(GuiRpcError) as error:
                run.save_raw()
            assert error.value.reason == "run_result_unavailable"
            assert not any(
                method == "tab.save_data" for method, _ in client.transport.sent
            )


@pytest.mark.parametrize("outcome", ["finished", "cancelled"])
def test_run_rejects_a_superseded_source_without_saving_it(tmp_path, outcome):
    gui = LookbackGui()

    def respond(method, params):
        reply = gui(method, params)
        if method == "operation.await":
            reply["status"] = outcome
        if method == "tab.snapshot" and gui.ran:
            reply["tabs"][0]["result_state"]["source_operation_id"] = 99
        return reply

    with recipe_client(tmp_path, respond) as client:
        operation = RecipeSession(client.context).open_tab("lookback").run()
        with pytest.raises(GuiRpcError) as error:
            operation.complete_in_current_worker()
        assert error.value.reason == "result_superseded"
        capture = operation.snapshot()
        assert capture is not None and capture.outcome is not None
        assert capture.outcome["status"] == outcome
        assert capture.result_state is not None
        assert capture.result_state["source_operation_id"] == 99
        assert capture.raw_save.status == "not_started"


def test_failed_run_preserves_its_native_outcome_before_throwing(tmp_path):
    gui = LookbackGui()

    def respond(method, params):
        if method == "operation.await":
            return {
                "reason": "completed",
                "status": "failed",
                "error": "acquisition failed",
            }
        return gui(method, params)

    with recipe_client(tmp_path, respond) as client:
        operation = RecipeSession(client.context).open_tab("lookback").run()
        with pytest.raises(GuiRpcError, match="acquisition failed") as error:
            operation.complete_in_current_worker()
        assert error.value.reason == "run_failed"
        capture = operation.snapshot()
        assert capture is not None and capture.outcome is not None
        assert capture.outcome["status"] == "failed"
        assert capture.result_state is None


def test_between_yield_cancel_suppresses_only_one_start_not_fast_edits(tmp_path):
    gui = LookbackGui()
    pending = Event()

    def consume_cancel():
        if pending.is_set():
            pending.clear()
            return True
        return False

    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context, consume_cancel=consume_cancel).open_tab(
            "lookback"
        )
        pending.set()
        tab.set("rounds", 8)
        assert pending.is_set()
        suppressed = tab.run()
        assert suppressed.complete_in_current_worker() == (None, "cancelled")
        assert suppressed.snapshot() is None
        assert not gui.ran
        assert not pending.is_set()
        run, status = tab.run().complete_in_current_worker()
        assert isinstance(run, RecipeRun) and status == "completed"
        capture = run.snapshot()
        assert capture is not None and capture.actual["fields"]["rounds"]["value"] == 8


def test_cancel_at_native_start_admission_does_not_send_run(tmp_path):
    gui = LookbackGui()
    pending = Event()

    def consume_cancel():
        if pending.is_set():
            pending.clear()
            return True
        return False

    def respond(method, params):
        if method == "device.snapshot":
            pending.set()
        return gui(method, params)

    with recipe_client(tmp_path, respond) as client:
        tab = RecipeSession(client.context, consume_cancel=consume_cancel).open_tab(
            "lookback"
        )
        assert tab.run().complete_in_current_worker() == (None, "cancelled")
        assert not any(method == "tab.run_start" for method, _ in client.transport.sent)


def test_start_rejection_retains_the_attempted_cfg_without_retry(tmp_path):
    gui = LookbackGui()
    with recipe_client(tmp_path, gui) as client:
        recipe = RecipeSession(client.context)
        tab = recipe.open_tab("lookback")
        tab.set("rounds", 6)
        client.transport.replies["tab.run_start"] = {
            "ok": False,
            "error": {
                "code": "precondition_failed",
                "reason": "stale",
                "message": "cfg changed",
            },
        }
        with pytest.raises(GuiRpcError) as error:
            tab.run()
        assert error.value.reason == "stale"
        capture = recipe.run_snapshot()
        assert capture is not None and capture.start_status == "not_started"
        assert capture.start_reason == "stale" and capture.op is None
        assert capture.actual["fields"]["rounds"]["value"] == 6
        assert (
            sum(method == "tab.run_start" for method, _ in client.transport.sent) == 1
        )
        assert sum(method == "tab.get_cfg" for method, _ in client.transport.sent) == 1


def test_uncertain_start_does_not_infer_not_started_from_a_missing_handle(tmp_path):
    gui = LookbackGui()

    def respond(method, params):
        if method == "tab.run_start":
            raise GuiRpcError("lost receipt", reason="connection_lost")
        return gui(method, params)

    with recipe_client(tmp_path, respond) as client:
        recipe = RecipeSession(client.context)
        tab = recipe.open_tab("lookback")
        with pytest.raises(GuiRpcError) as error:
            tab.run()
        assert error.value.reason == "connection_lost"
        capture = recipe.run_snapshot()
        assert capture is not None and capture.start_status == "unknown"
        assert (
            capture.op is None
            and capture.actual["cfg_ref"] == gui.publication["cfg_ref"]
        )
        assert (
            sum(method == "tab.run_start" for method, _ in client.transport.sent) == 1
        )


@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
def test_raw_save_failure_retains_reserved_path_and_prior_confirmed_save(
    tmp_path, outcome
):
    gui = LookbackGui()
    with recipe_client(tmp_path, gui) as client:
        operation = RecipeSession(client.context).open_tab("lookback").run()
        run, _ = operation.complete_in_current_worker()
        assert isinstance(run, RecipeRun)
        assert run.save_raw().path == "/actual/raw.h5"
        client.transport.replies["tab.save_data"] = {
            "ok": True,
            "result": {"operation_id": 83, "data_path": "/actual/second.h5"},
        }
        client.transport.replies["operation.await"] = {
            "ok": True,
            "result": {
                "reason": "completed",
                "status": outcome,
                "error": "save interrupted",
            },
        }
        with pytest.raises(GuiRpcError) as error:
            run.save_raw()
        assert error.value.reason == "raw_save_failed"
        capture = run.snapshot()
        assert capture is not None
        receipt = capture.raw_save
        assert receipt.status == "failed"
        assert receipt.reserved_path == "/actual/second.h5"
        assert receipt.path == "/actual/raw.h5"
        assert receipt.operation_outcome is not None
        assert receipt.operation_outcome["status"] == outcome


def test_uncertain_second_raw_save_retains_only_confirmed_prefix(tmp_path):
    gui = LookbackGui()
    attempts = 0

    def respond(method, params):
        nonlocal attempts
        if method == "tab.save_data":
            attempts += 1
            if attempts == 2:
                raise GuiRpcError("lost save receipt", reason="connection_lost")
        return gui(method, params)

    with recipe_client(tmp_path, respond) as client:
        operation = RecipeSession(client.context).open_tab("lookback").run()
        run, _ = operation.complete_in_current_worker()
        assert isinstance(run, RecipeRun)
        assert run.save_raw().path == "/actual/raw.h5"
        with pytest.raises(GuiRpcError) as error:
            run.save_raw()
        assert error.value.reason == "connection_lost"
        receipt = run.snapshot().raw_save
        assert receipt.status == "unknown" and receipt.path == "/actual/raw.h5"
        assert receipt.operation_outcome is None
        assert attempts == 2


@pytest.mark.parametrize("close_during_save", [False, True])
def test_raw_save_observes_true_outcome_without_consuming_pending_cancel(
    tmp_path, close_during_save
):
    gui = LookbackGui()
    pending = Event()
    closed = Event()
    calls = 0

    def consume_cancel():
        if pending.is_set():
            pending.clear()
            return True
        return False

    def respond(method, params):
        nonlocal calls
        if method == "operation.await" and params["operation_id"] == 82:
            calls += 1
            pending.set()
            if close_during_save:
                closed.set()
            if calls == 1:
                return {"reason": "timeout", "status": "running"}
        return gui(method, params)

    with recipe_client(tmp_path, respond) as client:
        tab = RecipeSession(
            client.context,
            closed=closed,
            condition=Condition(),
            consume_cancel=consume_cancel,
        ).open_tab("lookback")
        operation = tab.run()
        run, _ = operation.complete_in_current_worker()
        assert isinstance(run, RecipeRun)
        if close_during_save:
            with pytest.raises(GuiRpcError) as error:
                run.save_raw()
            assert error.value.reason == "session_closed"
            capture = run.snapshot()
            assert capture is not None
            receipt = capture.raw_save
            assert receipt.status == "unknown"
            assert receipt.reserved_path == "/actual/raw.h5" and receipt.path is None
            assert calls == 1
        else:
            assert run.save_raw().path == "/actual/raw.h5"
            assert calls == 2
            assert tab.run().complete_in_current_worker() == (None, "cancelled")
        assert pending.is_set() == close_during_save
