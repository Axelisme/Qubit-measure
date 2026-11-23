"""Composite scope paths cross the recipe/GUI wire without losing ambiguity."""

from dataclasses import asdict
from pathlib import Path
from threading import Event

import pytest
from zcu_tools.experiment import Scope
from zcu_tools.mcp.measure.recipe_api import RecipeSession, recipe

from ._script_recipe_support import (
    Fit,
    Inputs,
    Report,
    SyntheticGui,
    start,
    wait_for_terminal,
)
from ._support import make_client


@pytest.fixture
def gui_fork():
    with (
        Scope("run") as root,
        root.branch("schedule") as first,
        root.branch("schedule") as second,
        first.branch("acquire") as left,
        second.branch("acquire") as right,
    ):
        yield root, left, right


@pytest.mark.parametrize("selector", ["acquire", "acquire#1"])
def test_recipe_projects_fork_leaves_candidates_and_exact_selector(
    tmp_path: Path, selector: str, gui_fork
) -> None:
    gui = SyntheticGui()
    root, left, right = gui_fork
    waiting, released = Event(), Event()
    prefix = "recipe:synthetic/operation#1"
    paths = [f"{prefix}/schedule#1/acquire#1", f"{prefix}/schedule#2/acquire#1"]

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "operation.status_scope":
            assert params == {"operation_id": 1}
            return {"active_path": root.active_path, "live_paths": root.live_paths}
        if method == "operation.cancel_scope":
            assert params["operation_id"] == 1
            scope = params["scope"]
            assert isinstance(scope, str)
            expected = params.get("expected")
            assert expected is None or isinstance(expected, int)
            return asdict(root.cancel(scope, expected=expected))
        if method == "operation.await" and params["operation_id"] == 1:
            waiting.set()
            return (
                {"reason": "completed", "status": "finished"}
                if released.is_set()
                else {"reason": "timeout"}
            )
        if method == "operation.cancel":
            released.set()
            return {"status": "requested"}
        return gui.respond(method, params)

    @recipe("synthetic", "Project a GUI fork and select its exact leaf")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        await session.open_tab("synthetic").run()
        return Report(count=inputs.count)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    execution = start(client, "synthetic")
    try:
        # The submitted Run necessarily enters operation.await before it can finish.
        assert waiting.wait(30)
        execution_id = execution.snapshot().execution
        status = client.call("status", {"execution": execution_id})
        assert status["active_path"] == prefix
        assert status["live_paths"] == paths
        rejected = client.call("cancel", {"execution": execution_id, "scope": selector})
        assert (rejected.data["cancelled"], rejected.data["reason"]) == (
            False,
            "ambiguous",
        )
        assert rejected.data["candidates"] == paths
        assert rejected.data["active_path"] == prefix
        assert not left.cancelled and not right.cancelled
        accepted = client.call("cancel", {"execution": execution_id, "scope": paths[1]})
        assert accepted.data["cancelled"] is True
        assert not left.cancelled and right.cancelled
        released.set()
        execution.join()
        assert execution.snapshot().status == "completed"
    finally:
        released.set()
        execution.join()
        client.context.session.close()


def test_execution_projects_only_the_current_run_progress(tmp_path: Path) -> None:
    gui = SyntheticGui()
    run_waiting, release_run = Event(), Event()
    analysis_waiting, release_analysis = Event(), Event()
    bars = [
        {"label": "schedule", "percent": 25.0, "eta_s": 20.0},
        {"label": "read", "percent": 50.0, "eta_s": 8.0},
        {"label": "shots", "percent": None, "eta_s": None},
    ]

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "operation.status_scope":
            return {"active_path": "run", "live_paths": ["run"]}
        if method == "operation.await" and params["operation_id"] == 1:
            run_waiting.set()
            if not release_run.is_set():
                return {"reason": "timeout"}
        if method == "operation.await" and params["operation_id"] == 100:
            analysis_waiting.set()
            if not release_analysis.is_set():
                return {"reason": "timeout"}
        if method == "operation.progress":
            if params["operation_id"] == 1:
                return {"active": True, "bars": bars}
            # Analysis telemetry also exists on the wire; it is not Run progress.
            return {"active": True, "bars": [{"label": "fit", "eta_s": 3.0}]}
        return gui.respond(method, params)

    @recipe("synthetic", "Run, analyze, then ask for a writeback decision")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        await tab.analyze("primary", result=Fit)
        await tab.ask_writeback()
        return Report(count=inputs.count)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    execution = start(client, "synthetic")
    selector = {"execution": execution.snapshot().execution}
    try:
        assert run_waiting.wait(3)
        status = client.call("status", {**selector, "detail": "full"})
        op = status["details"][0]["op"]
        operation = client.call("wait", {"op": op, "timeout": 0})
        assert operation["progress"] == bars
        assert operation["eta_s"] == 20.0
        replies = (
            status,
            client.call("status", selector),
            client.call("wait", {**selector, "timeout": 0}).data,
        )
        for reply in replies:
            assert reply["progress"] == operation["progress"]
            assert reply["eta_s"] == operation["eta_s"]

        release_run.set()
        assert analysis_waiting.wait(3)
        after_run = (
            client.call("status", selector),
            client.call("wait", {**selector, "timeout": 0}).data,
        )
        release_analysis.set()
        waiting = client.call("wait", {**selector, "timeout": 3}).data
        assert waiting["status"] == "waiting"
        waiting_status = client.call("status", selector)
        client.call(
            "answer",
            {
                **selector,
                "request_id": waiting["pending_question"]["request_id"],
                "answer": {"accepted": False, "items": []},
            },
        )
        wait_for_terminal(execution)
        terminal = client.call("wait", {**selector, "timeout": 0}).data
        assert terminal["status"] == "completed"
        for reply in (
            *after_run,
            waiting,
            waiting_status,
            terminal,
            client.call("status", selector),
        ):
            assert "progress" not in reply
            assert "eta_s" not in reply
    finally:
        release_run.set()
        release_analysis.set()
        client.context.session.close()
