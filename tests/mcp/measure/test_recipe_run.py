"""Author Run/source capture and raw-save contracts through the GUI transport."""

from threading import Event

import pytest
from zcu_tools.mcp.measure.execution_reply import SummaryParameter
from zcu_tools.mcp.measure.recipe import (
    RecipeRun,
    RecipeSession,
)
from zcu_tools.mcp.measure.session import GuiRpcError

from ._recipe_support import LookbackGui, recipe_client


def test_run_starts_immediately_with_detached_conditions(tmp_path):
    gui = LookbackGui()
    gui.md["$resonator.freq"] = 6100.0
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(
            client.context,
            summary_parameters=(
                SummaryParameter("rounds", ("rounds",)),
                SummaryParameter("reset", ("modules", "reset")),
                SummaryParameter("frequency", ("modules", "readout", "freq")),
            ),
        ).open_tab("lookback")
        tab.set(("rounds",), 7)
        tab.disable_library(("modules", "reset"))
        tab.read_value("$resonator.freq")
        for path in (("modules", "readout", "freq"),):
            tab.set(path, {"expr": "$resonator.freq"}, source="$resonator.freq")
        operation = tab.run()
        assert gui.ran
        assert not any(
            method == "operation.await" for method, _ in client.transport.sent
        )
        before = operation.snapshot()
        assert before is not None and before.start_status == "running"
        assert before.actual["fields"]["rounds"] == {
            "value": 7,
            "source": "explicit",
        }
        assert before.actual["fields"]["reset"] == {
            "value": None,
            "source": "disabled",
        }
        assert before.actual["fields"]["frequency"] == {
            "value": 6100.0,
            "source": "$resonator.freq",
        }
        tab.set(("rounds",), 11)
        before.actual["fields"]["rounds"]["value"] = -1
        run, status = operation.complete_in_current_worker()
        assert isinstance(run, RecipeRun) and status == "completed"
        captured = run.snapshot()
        assert captured is not None
        assert captured.actual["fields"]["rounds"]["value"] == 7
        assert captured.result_state == {
            "available": True,
            "revision": 1,
            "source_operation_id": 71,
        }
        assert captured.outcome == {"reason": "completed", "status": "finished"}
        receipt = run.save_raw()
        assert receipt.run_id == "run-a"
        assert receipt.artifacts["raw"]["data"].status == "Saved"
        assert [
            output.path for output in receipt.artifacts["raw"]["data"].members["data"]
        ] == ["/runs/run-a/data.h5", "/labber/raw.hdf5"]
        captured = operation.snapshot()
        assert captured is not None and captured.raw_save == receipt
        assert (
            "tab.run_start",
            {
                "tab_id": "t",
            },
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
            report = run.save_raw()
            assert report.run_id == "run-a"
            assert report.artifacts["raw"]["data"].status == "Saved"
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
        assert capture.raw_save is None


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
        tab = RecipeSession(
            client.context,
            consume_cancel=consume_cancel,
            summary_parameters=(SummaryParameter("rounds", ("rounds",)),),
        ).open_tab("lookback")
        pending.set()
        tab.set(("rounds",), 8)
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
        recipe = RecipeSession(
            client.context,
            summary_parameters=(SummaryParameter("rounds", ("rounds",)),),
        )
        tab = recipe.open_tab("lookback")
        tab.set(("rounds",), 6)
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
        assert (
            sum(method == "tab.reset_cfg" for method, _ in client.transport.sent) == 1
        )


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
            sum(method == "tab.run_start" for method, _ in client.transport.sent) == 1
        )
