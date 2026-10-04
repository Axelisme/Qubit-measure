"""Failed/uncertain generator handoffs retain owner facts and run Python finally."""

from threading import Event

import pytest
from zcu_tools.mcp.measure.recipe import RecipeGenerator, RecipeRun, RecipeSession
from zcu_tools.mcp.measure.session import GuiRpcError

from ._recipe_analysis_support import AnalysisRecipeGui
from ._recipe_execution_support import ControlledRunGui, definition, registry
from ._recipe_support import LookbackGui, recipe_client


@pytest.mark.parametrize("uncertain", [False, True])
def test_unyielded_run_start_failure_retains_fixed_conditions(tmp_path, uncertain):
    finally_calls: list[str] = []

    def sample(session: RecipeSession) -> RecipeGenerator:
        try:
            tab = session.open_tab("lookback")
            tab.set("rounds", 8)
            _, _ = yield tab.run()
        finally:
            finally_calls.append("closed")

    gui = LookbackGui()

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.run_start" and uncertain:
            raise GuiRpcError("lost receipt", reason="connection_lost")
        return gui(method, params)

    with (
        recipe_client(tmp_path, respond) as client,
        registry(client.context, definition(sample)) as executions,
    ):
        if not uncertain:
            client.transport.replies["tab.run_start"] = {
                "ok": False,
                "error": {
                    "code": "precondition_failed",
                    "reason": "stale",
                    "message": "cfg changed",
                },
            }
        execution = executions.start(client.context, "sample", {})
        reply = execution.wait(5)
        assert reply.data["status"] == "failed"
        assert reply.data["actual"]["fields"]["rounds"]["value"] == 8
        assert reply.data["run_op"] is None
        assert reply.data["run_start"]["status"] == (
            "unknown" if uncertain else "not_started"
        )
        assert reply.data["error"]["reason"] == (
            "connection_lost" if uncertain else "stale"
        )
        assert (
            sum(method == "tab.run_start" for method, _ in client.transport.sent) == 1
        )
        execution.close()
        execution.join()
        assert finally_calls == ["closed"]


@pytest.mark.parametrize("reuse", [None, "t"])
def test_pre_run_cfg_failure_retains_prepared_or_requested_locator(tmp_path, reuse):
    def sample(session: RecipeSession) -> RecipeGenerator:
        tab = session.open_tab("lookback", reuse=reuse)
        _, _ = yield tab.run()

    gui = LookbackGui()
    with (
        recipe_client(tmp_path, gui) as client,
        registry(client.context, definition(sample)) as executions,
    ):
        client.transport.replies["tab.get_cfg"] = {
            "ok": False,
            "error": {
                "code": "precondition_failed",
                "reason": "stale_cfg",
                "message": "Changed",
            },
        }
        execution = executions.start(client.context, "sample", {})
        reply = execution.wait(5)
        assert reply.data["status"] == "failed"
        assert reply.data["tab"] == "t"
        assert reply.data["actual"] is None
        assert reply.data["run_start"]["status"] == "not_started"
        assert not gui.ran
        before = list(client.transport.sent)
        assert execution.snapshot().tab == "t"
        assert client.transport.sent == before
        execution.close()
        execution.join()


@pytest.mark.parametrize("control", ["cancel", "finish_early"])
def test_stop_intent_does_not_relabel_failed_native_run(tmp_path, control):
    caught: list[str | None] = []

    def sample(session: RecipeSession) -> RecipeGenerator:
        try:
            _, _ = yield session.open_tab("lookback").run()
        except GuiRpcError as error:
            caught.append(error.reason)
            raise

    gui = ControlledRunGui()

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "operation.cancel":
            gui.outcome = "failed"
            gui.settled.set()
            return {"status": "cancelling"}
        return gui(method, params)

    with (
        recipe_client(tmp_path, respond) as client,
        registry(client.context, definition(sample)) as executions,
    ):
        execution = executions.start(client.context, "sample", {})
        assert gui.awaited.wait(2)
        if control == "cancel":
            execution.cancel()
        else:
            execution.finish_early()
        reply = execution.wait(5)
        assert reply.data["status"] == "failed"
        assert reply.data["run_outcome"]["status"] == "failed"
        assert caught == ["run_failed"]
        assert not gui.raw_saved


def test_close_question_runs_finally_once_without_delivering_answer(tmp_path):
    finally_calls: list[str] = []
    resumed = Event()

    def sample(session: RecipeSession) -> RecipeGenerator:
        try:
            run, _ = yield session.open_tab("lookback").run()
            assert isinstance(run, RecipeRun)
            _, _ = yield run.analyze("primary")
            _, _ = yield run.propose_writeback()
            resumed.set()
        finally:
            finally_calls.append("closed")

    with (
        recipe_client(tmp_path, AnalysisRecipeGui()) as client,
        registry(client.context, definition(sample)) as executions,
    ):
        execution = executions.start(client.context, "sample", {})
        assert execution.wait(5).data["status"] == "awaiting_answer"
        execution.close()
        execution.join()
        assert execution.snapshot().status == "cancelled"
        assert execution.snapshot().question_preview is None
        assert not resumed.is_set()
        execution.close()
        execution.join()
        assert finally_calls == ["closed"]


@pytest.mark.parametrize("uncertain", [False, True])
def test_failed_second_save_preserves_confirmed_prefix_and_failure_phase(
    tmp_path, uncertain
):
    def sample(session: RecipeSession) -> RecipeGenerator:
        run, _ = yield session.open_tab("lookback").run()
        assert isinstance(run, RecipeRun)
        run.save_raw()
        run.save_raw()

    gui = LookbackGui()
    saves = 0

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        nonlocal saves
        if method == "tab.save_data":
            saves += 1
            if saves == 2 and uncertain:
                raise GuiRpcError("lost receipt", reason="connection_lost")
            return {
                "operation_id": 82 if saves == 1 else 84,
                "data_path": "/first.h5" if saves == 1 else "/second.h5",
            }
        if method == "operation.await" and params["operation_id"] == 84:
            return {"reason": "completed", "status": "failed", "error": "disk full"}
        return gui(method, params)

    with (
        recipe_client(tmp_path, respond) as client,
        registry(client.context, definition(sample)) as executions,
    ):
        execution = executions.start(client.context, "sample", {})
        reply = execution.wait(5)
        assert reply.data["status"] == "failed"
        assert reply.data["error"]["phase"] == "raw_save"
        assert reply.data["raw_save"]["path"] == "/first.h5"
        assert reply.data["raw_save"]["status"] == (
            "unknown" if uncertain else "failed"
        )
        assert saves == 2


def test_worker_start_failure_returns_a_queryable_terminal_record(
    tmp_path, monkeypatch
):
    def sample(session: RecipeSession) -> RecipeGenerator:
        yield session.open_tab("lookback").run()

    def fail_start(thread):
        raise RuntimeError("thread unavailable")

    with (
        recipe_client(tmp_path, LookbackGui()) as client,
        registry(client.context, definition(sample)) as executions,
    ):
        monkeypatch.setattr("threading.Thread.start", fail_start)
        execution = executions.start(client.context, "sample", {})
        reply = execution.wait(0)
        assert reply.data["status"] == "failed" and reply.is_error
        assert reply.data["error"]["reason"] == "recipe_worker_start_failed"
        assert executions.get(reply.data["execution"]) is execution
        execution.close()
        execution.join()
        assert not any(method == "tab.run_start" for method, _ in client.transport.sent)


def test_generator_close_failure_is_isolated_with_a_terminal_record(tmp_path):
    def sample(session: RecipeSession) -> RecipeGenerator:
        try:
            _, _ = yield session.open_tab("lookback").run()
        finally:
            raise RuntimeError("finally failed")

    gui = ControlledRunGui()
    with (
        recipe_client(tmp_path, gui) as client,
        registry(client.context, definition(sample)) as executions,
    ):
        execution = executions.start(client.context, "sample", {})
        assert gui.awaited.wait(2)
        execution.close()
        execution.join()
        reply = execution.wait(0)
        assert reply.data["status"] == "failed" and reply.is_error
        assert reply.data["error"]["reason"] == "recipe_close_failed"
        assert reply.data["error"]["message"] == "finally failed"
