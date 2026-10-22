"""Recipe cancellation at the public session/execution and MCP seams."""

import sys
from pathlib import Path
from threading import Event
from typing import Literal

import pytest
from zcu_tools.mcp.measure import recipe_api
from zcu_tools.mcp.measure.recipe_api import RecipeSession, recipe

from ._script_recipe_support import (
    ConfirmQuestion,
    Fit,
    Inputs,
    Report,
    SyntheticGui,
    start,
)
from ._support import make_client


@pytest.mark.parametrize("scope", ["acquire", "schedule"])
def test_recipe_proxies_fine_scope_by_run_handle_and_projects_active_path(
    tmp_path: Path, scope: str
) -> None:
    gui = SyntheticGui(partial_first=True)
    waiting = Event()
    released = Event()
    requests: list[tuple[str, dict[str, object]]] = []
    partials: list[bool] = []
    path = "recipe:synthetic/iteration#1/operation#1/schedule#1/acquire#2"

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        requests.append((method, params))
        if method == "operation.status_scope":
            assert params == {"operation_id": 1}
            return {"active_path": "run/schedule#1/acquire#2"}
        if method == "operation.cancel_scope":
            assert params["operation_id"] == 1
            expected = params.get("expected")
            accepted = expected is None or expected == (2 if scope == "acquire" else 1)
            if accepted:
                released.set()
            return {
                "cancelled": accepted,
                "reason": "scope cancellation requested"
                if accepted
                else "sequence mismatch",
                "active_path": "run/schedule#1/acquire#2",
            }
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

    @recipe("synthetic", "Cancel a GUI-owned scope, then run again")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        partials.append((await tab.run()).partial_result)
        partials.append((await tab.run()).partial_result)
        return Report(count=inputs.count)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        assert waiting.wait(30), "The first GUI operation must enter its await"
        execution_id = execution.snapshot().execution
        assert client.call("status", {"execution": execution_id})["active_path"] == path
        rejected = client.call(
            "cancel", {"execution": execution_id, "scope": scope, "expected": 99}
        )
        assert rejected.data["cancelled"] is False
        assert rejected.data["reason"] == "sequence mismatch"
        assert rejected.data["active_path"] == path
        accepted = client.call("cancel", {"execution": execution_id, "scope": scope})
        assert accepted.data["cancelled"] is True
        assert accepted.data["active_path"] == path
        execution.join()
        assert execution.snapshot().status == "completed"
        assert partials == [True, False]
        assert [
            params["operation_id"]
            for method, params in requests
            if method == "operation.cancel_scope"
        ] == [1, 1]
        assert not [method for method, _ in requests if method == "operation.cancel"]
    finally:
        released.set()
        client.context.session.close()


@pytest.mark.parametrize("scope", ["acquire", "schedule"])
def test_fine_scope_old_recipe_op_cannot_cancel_the_next_run(
    tmp_path: Path, scope: str
) -> None:
    gui = SyntheticGui()
    waiting = Event()
    released = Event()
    scoped_requests: list[int] = []
    partials: list[bool] = []

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "operation.status_scope":
            assert params == {"operation_id": 2}
            return {"active_path": "run/schedule#1/acquire#1"}
        if method == "operation.cancel_scope":
            assert params["operation_id"] == 2
            scoped_requests.append(2)
            gui.cancelled.set()
            released.set()
            return {
                "cancelled": True,
                "reason": "scope cancellation requested",
                "active_path": "run/schedule#1/acquire#1",
            }
        if method == "operation.await" and params["operation_id"] == 2:
            waiting.set()
            return (
                {"reason": "completed", "status": "finished"}
                if released.is_set()
                else {"reason": "timeout"}
            )
        return gui.respond(method, params)

    @recipe("synthetic", "Keep fine cancellation bound to the observed Run")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        partials.append((await tab.run()).partial_result)
        partials.append((await tab.run()).partial_result)
        return Report(count=inputs.count)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        assert waiting.wait(3), "The second Run must be active"
        first, second = execution.snapshot().details
        path = "recipe:synthetic/iteration#1/operation#2/schedule#1/acquire#1"
        stale = client.call("cancel", {"op": first.op, "scope": scope, "expected": 1})
        assert stale.data["cancelled"] is False
        assert stale.data["reason"] == "scope not active"
        assert stale.data["active_path"] == path
        assert scoped_requests == []
        assert not released.is_set()
        accepted = client.call(
            "cancel", {"op": second.op, "scope": scope, "expected": 1}
        )
        assert accepted.data["cancelled"] is True
        execution.join()
        assert execution.snapshot().status == "completed"
        assert partials == [False, True]
        assert scoped_requests == [2]
    finally:
        released.set()
        client.context.session.close()


def test_recipe_reads_partial_completion_without_stop_intent_or_cancel_ack(
    tmp_path: Path,
) -> None:
    gui = SyntheticGui(partial_first=True)
    partials: list[bool] = []

    @recipe("synthetic", "Observe GUI completion independently of control intent")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        partials.append((await tab.run()).partial_result)
        partials.append((await tab.run()).partial_result)
        return Report(count=inputs.count)

    client = make_client(tmp_path, gui.respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.join()
        assert execution.snapshot().status == "completed"
        assert partials == [True, False]
        assert not [
            method
            for method, _ in client.transport.sent
            if method == "operation.cancel_scope"
        ]
    finally:
        client.context.session.close()


@pytest.mark.parametrize("await_kind", ["analyze", "ask", "writeback"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_stopped_script_await_unwinds_author_and_preserves_cleanup_error(
    tmp_path: Path,
    await_kind: Literal["analyze", "ask", "writeback"],
    cleanup_fails: bool,
) -> None:
    gui = SyntheticGui(hold=True)
    unwinding: list[BaseException | None] = []
    continued: list[bool] = []

    @recipe("synthetic", "Stop a value-bearing await after partial acquisition")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        try:
            assert (await tab.run()).partial_result
            if await_kind == "analyze":
                await tab.analyze("primary", result=Fit)
            elif await_kind == "ask":
                await session.ask(ConfirmQuestion(evidence="never answered"))
            else:
                await tab.ask_writeback()
            continued.append(True)
            return Report(count=inputs.count)
        finally:
            unwinding.append(sys.exception())
            tab.save()
            if cleanup_fails:
                raise ValueError("cleanup failed")

    client = make_client(tmp_path, gui.respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        assert gui.started.wait(3)
        client.call("cancel", {"execution": execution.snapshot().execution})
        execution.join()
        snapshot = execution.snapshot()
        assert continued == []
        assert len(unwinding) == 1
        assert isinstance(unwinding[0], recipe_api.RecipeCancelled)
        assert isinstance(unwinding[0], Exception)
        assert snapshot.status == "cancelled"
        assert snapshot.outcome is not None
        assert snapshot.outcome.report is None
        assert len(snapshot.outcome.runs) == 1
        assert snapshot.outcome.runs[0].partial_result
        assert len(snapshot.outcome.runs[0].saves) == 1
        if cleanup_fails:
            assert snapshot.error is not None
            assert (snapshot.error.type, snapshot.error.message) == (
                "ValueError",
                "cleanup failed",
            )
        else:
            assert snapshot.error is None
    finally:
        client.context.session.close()


@pytest.mark.parametrize("await_kind", ["analyze", "ask", "writeback"])
def test_cancel_pending_value_await_unwinds_without_an_error(
    tmp_path: Path, await_kind: Literal["analyze", "ask", "writeback"]
) -> None:
    gui = SyntheticGui()
    analysis_started = Event()
    unwinding: list[BaseException | None] = []
    continued: list[bool] = []

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "operation.await" and params["operation_id"] == 100:
            analysis_started.set()
            return (
                {"reason": "completed", "status": "cancelled"}
                if gui.cancelled.is_set()
                else {"reason": "timeout"}
            )
        return gui.respond(method, params)

    @recipe("synthetic", "Cancel a pending value-bearing await")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        try:
            if await_kind == "analyze":
                await tab.analyze("primary", result=Fit)
            elif await_kind == "ask":
                await session.ask(ConfirmQuestion(evidence="pending"))
            else:
                await tab.analyze("primary", result=Fit)
                await tab.ask_writeback()
            continued.append(True)
            return Report(count=inputs.count)
        finally:
            unwinding.append(sys.exception())

    # Only the analysis case holds the operation; writeback reaches a real question.
    client = make_client(
        tmp_path,
        respond if await_kind == "analyze" else gui.respond,
        recipes=(synthetic,),
    )
    try:
        execution = start(client, "synthetic")
        if await_kind == "analyze":
            assert analysis_started.wait(3)
        else:
            execution.wait(3)
            assert execution.snapshot().pending_question is not None
        execution.cancel()
        execution.join()
        snapshot = execution.snapshot()
        assert continued == []
        assert len(unwinding) == 1
        assert isinstance(unwinding[0], recipe_api.RecipeCancelled)
        assert snapshot.status == "cancelled"
        assert snapshot.error is None
        assert snapshot.outcome is not None
        assert snapshot.outcome.report is None
        assert len(snapshot.outcome.runs) == 1
        assert not snapshot.outcome.runs[0].partial_result
    finally:
        client.context.session.close()


def test_script_cancel_preserves_real_operation_failure(tmp_path: Path) -> None:
    gui = SyntheticGui(hold=True)
    cleaned: list[bool] = []

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "operation.await" and gui.cancelled.is_set():
            return {
                "reason": "completed",
                "status": "failed",
                "error": "acquisition failed",
            }
        return gui.respond(method, params)

    @recipe("synthetic", "Keep operation errors during cancellation")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        try:
            await session.open_tab("synthetic").run()
            return Report(count=inputs.count)
        finally:
            cleaned.append(True)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        assert gui.started.wait(3)
        execution.cancel()
        execution.join()
        snapshot = execution.snapshot()
        assert cleaned == [True]
        assert snapshot.status == "cancelled"
        assert snapshot.error is not None
        assert snapshot.error.type == "GuiRpcError"
        assert snapshot.error.message == "acquisition failed"
        assert snapshot.outcome is not None
        assert snapshot.outcome.report is None
        assert snapshot.outcome.runs[0].status == "failed"
    finally:
        client.context.session.close()
