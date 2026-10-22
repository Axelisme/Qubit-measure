"""Composite scope paths cross the recipe/GUI wire without losing ambiguity."""

from dataclasses import asdict
from pathlib import Path
from threading import Event

import pytest
from zcu_tools.experiment import Scope
from zcu_tools.mcp.measure.recipe_api import RecipeSession, recipe

from ._script_recipe_support import Inputs, Report, SyntheticGui, start
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
