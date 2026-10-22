"""Async recipe contracts through declarations and a recording GUI transport."""

from collections.abc import Callable
from pathlib import Path
from threading import Event, get_ident

import pytest
from pydantic import BaseModel, ValidationError
from zcu_tools.experiment.script_host import CancelScope
from zcu_tools.mcp.measure.recipe_api import RecipeSession, recipe
from zcu_tools.mcp.measure.recipe_contracts import (
    MissingParameter,
    RecipeNeedsParameters,
)

from ._script_recipe_support import (
    ConfirmQuestion,
    Fit,
    Inputs,
    Report,
    SyntheticGui,
    start,
)
from ._support import make_client


def test_async_declaration_runs_edits_analysis_and_single_synchronous_save(
    tmp_path: Path,
) -> None:
    gui = SyntheticGui()

    @recipe("synthetic", "Run and analyze a synthetic tab")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic", aliases={"qubit": "Q1"})
        tab.set(("count",), inputs.count)
        tab.set(("kept",), None)
        tab.select(("module",), ref="captured-key")
        tab.select(("module",), inline=True)
        run = await tab.run()
        assert run.run_id == "native-1"
        fit = await tab.analyze("primary", result=Fit)
        assert fit.result is not None
        assert fit.result.estimate == 3.5
        saved = tab.save()
        assert saved.run_id == run.run_id
        return Report(count=inputs.count)

    client = make_client(tmp_path, gui.respond, recipes=(synthetic,))
    try:
        assert synthetic.inputs.model_json_schema()["required"] == ["count"]
        execution = start(client, "synthetic")
        execution.wait(3)
        outcome = execution.snapshot().outcome
        assert outcome is not None
        assert outcome.status == "completed"
        assert outcome.report == Report(count=9)
        assert len(outcome.runs) == 1
        run = outcome.runs[0]
        assert (run.tab, run.adapter, run.run_id, run.status) == (
            "tab-1",
            "synthetic",
            "native-1",
            "completed",
        )
        assert len(run.saves) == 1
        assert {output.format for output in run.saves[0].outputs} == {"data_h5", "json"}
        methods = [name for name, _ in client.transport.sent]
        assert (
            methods.index("tab.get_analyze_result")
            < methods.index("tab.save_data")
            < methods.index("tab.save_artifacts")
        )
        assert (
            next(
                params
                for method, params in client.transport.sent
                if method == "tab.save_artifacts"
            )["artifacts"]
            == "all"
        )
    finally:
        client.context.session.close()


def test_run_cancel_delivers_partial_and_runs_cleanup_on_author_worker(
    tmp_path: Path,
) -> None:
    gui = SyntheticGui(hold=True)
    threads: list[int] = []
    partial: list[bool] = []

    @recipe("synthetic", "Cancel a synthetic acquisition")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        threads.append(get_ident())
        tab = session.open_tab("synthetic")
        try:
            partial.append((await tab.run()).partial_result)
            fit = await tab.analyze("primary", result=Fit)
            assert fit is not None
            return Report(count=inputs.count)
        finally:
            threads.append(get_ident())

    client = make_client(tmp_path, gui.respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        assert gui.started.wait(3)
        execution.cancel(scope="run")
        execution.wait(3)
        snapshot = execution.snapshot()
        outcome = snapshot.outcome
        assert outcome is not None
        assert outcome.status == "completed"
        assert outcome.report == Report(count=9)
        assert partial == [True]
        assert len(outcome.runs) == 1
        assert all(run.partial_result for run in outcome.runs)
        assert len(threads) == 2 and threads[0] == threads[1] != get_ident()
        assert snapshot.error is None
    finally:
        client.context.session.close()


def test_decorator_rejects_non_coroutine_and_invalid_report_annotations() -> None:
    def sync(session: RecipeSession, inputs: Inputs) -> Report:
        return Report(count=inputs.count)

    async def invalid(session: RecipeSession, inputs: Inputs) -> int:
        return inputs.count

    # This test deliberately sends invalid Python callables at the runtime boundary.
    def apply(decorate: Callable[..., object], function: object) -> object:
        return decorate(function)

    with pytest.raises(TypeError, match="coroutine"):
        apply(recipe("invalid", "Reject invalid authors"), sync)
    with pytest.raises(TypeError, match="Report"):
        apply(recipe("invalid", "Reject invalid authors"), invalid)


def test_invalid_inputs_fail_before_binding_gui(tmp_path: Path) -> None:
    @recipe("synthetic", "Validate input before connecting")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        return Report(count=inputs.count)

    client = make_client(tmp_path, recipes=(synthetic,))
    try:
        execution = client.context.session.recipes.start(
            client.context, "synthetic", {"count": "not an integer"}
        )
        execution.wait(3)
        assert execution.snapshot().status == "failed"
        assert client.transport.sent == []
    finally:
        client.context.session.close()


def test_writeback_question_validates_before_accept_and_save(tmp_path: Path) -> None:
    gui = SyntheticGui()

    @recipe("synthetic", "Ask before writing")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        await tab.analyze("primary", result=Fit)
        decision = await tab.ask_writeback()
        assert decision is not None
        if decision.accepted:
            receipt = tab.accept_writeback(decision.items)
            assert receipt.status == "completed"
            assert receipt.items[0].destination.path == "Q1.freq"
            assert receipt.items[0].destination.label == "preview-context"
        tab.save()
        return Report(count=inputs.count)

    client = make_client(tmp_path, gui.respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.wait(3)
        pending = execution.snapshot().pending_question
        assert pending is not None
        question = pending.question
        items = question.model_dump(mode="json")["proposal"]["items"]
        with pytest.raises((ValueError, ValidationError)):
            execution.answer(
                {
                    "request_id": pending.request_id,
                    "answer": {
                        "accepted": True,
                        "items": [{**items[0], "name": "unknown"}],
                    },
                }
            )
        assert execution.snapshot().pending_question is not None
        reply = execution.answer(
            {
                "request_id": pending.request_id,
                "answer": {"accepted": True, "items": items},
            }
        )
        assert reply.data["outcome"]["status"] == "completed"
        methods = [method for method, _ in client.transport.sent]
        assert methods.index("tab.writeback_accept") < methods.index("tab.save_data")
    finally:
        client.context.session.close()


@pytest.mark.parametrize("summary", [{"estimate": 3.5}, None])
def test_analysis_summary_missing_authored_field_fails_without_repair(
    tmp_path: Path,
    summary: dict[str, float] | None,
) -> None:
    class RequiredFit(BaseModel):
        absent: float

    @recipe("synthetic", "Decode the exact owner summary")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        await tab.analyze("primary", result=RequiredFit)
        return Report(count=inputs.count)

    gui = SyntheticGui()

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.get_analyze_result":
            return {"summary": summary, "invalid": []}
        return gui.respond(method, params)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.wait(3)
        snapshot = execution.snapshot()
        assert snapshot.status == "failed"
        assert snapshot.error is not None
        assert snapshot.error.type == "ValidationError"
        assert (
            "absent" in snapshot.error.message
            if summary is not None
            else "RequiredFit" in snapshot.error.message
        )
    finally:
        client.context.session.close()


def test_missing_parameters_is_terminal_and_releases_recipe_admission(
    tmp_path: Path,
) -> None:
    from zcu_tools.mcp.measure.recipe_contracts import (
        MissingParameter,
        RecipeNeedsParameters,
    )

    @recipe("missing", "Require an authored calibration parameter")
    async def missing(session: RecipeSession, inputs: Inputs) -> Report:
        raise RecipeNeedsParameters(
            (MissingParameter("frequency_mhz", "Provide the center frequency"),)
        )

    @recipe("next", "Prove the next recipe can be admitted")
    async def next_recipe(session: RecipeSession, inputs: Inputs) -> Report:
        return Report(count=inputs.count)

    client = make_client(tmp_path, recipes=(missing, next_recipe))
    try:
        execution = start(client, "missing")
        execution.wait(3)
        snapshot = execution.snapshot()
        assert snapshot.status == "needs_parameters"
        assert snapshot.missing == ("frequency_mhz",)
        assert snapshot.outcome is None
        following = start(client, "next")
        following.wait(3)
        assert following.snapshot().status == "completed"
    finally:
        client.context.session.close()


def test_run_cancel_during_ask_is_acknowledged_noop(tmp_path: Path) -> None:
    from zcu_tools.core.script_protocol import Answer, Question

    class Choice(Answer):
        accepted: bool

    class Confirm(Question[Choice]):
        evidence: str

    @recipe("synthetic", "Keep the question pending when no Run is active")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        answer = await session.ask(Confirm(evidence="no acquisition"))
        assert answer is not None and answer.accepted
        return Report(count=inputs.count)

    client = make_client(tmp_path, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.wait(3)
        pending = execution.snapshot().pending_question
        assert pending is not None
        reply = execution.cancel(scope="run")
        assert reply.data["cancelled"] is False
        assert reply.data["reason"] == "no run in progress"
        assert execution.snapshot().pending_question == pending
        answered = execution.answer(
            {"request_id": pending.request_id, "answer": {"accepted": True}}
        )
        assert answered.data["outcome"]["status"] == "completed"
    finally:
        client.context.session.close()


def test_run_cancel_during_analysis_does_not_stop_analysis(tmp_path: Path) -> None:
    gui = SyntheticGui()
    analysis_started, release_analysis = Event(), Event()

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if (
            method == "operation.await"
            and params["operation_id"] == 100
            and not analysis_started.is_set()
        ):
            analysis_started.set()
            assert release_analysis.wait(3)
            return {"reason": "timeout"}
        return gui.respond(method, params)

    @recipe("synthetic", "Keep analysis running when cancelling only Run")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        fit = await tab.analyze("primary", result=Fit)
        assert fit is not None and fit.result.estimate == 3.5
        return Report(count=inputs.count)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        assert analysis_started.wait(3)
        reply = execution.cancel(scope="run")
        release_analysis.set()
        assert reply.data["cancelled"] is False
        assert reply.data["reason"] == "no run in progress"
        execution.wait(3)
        assert execution.snapshot().status == "completed"
        assert not gui.cancelled.is_set()
    finally:
        release_analysis.set()
        client.context.session.close()


def test_context_reads_session_source_before_opening_any_tab(tmp_path: Path) -> None:
    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        assert method == "value.read"
        assert params == {"key": "device.synthetic.value"}
        return {
            "key": "device.synthetic.value",
            "type": "int",
            "owner": "synthetic",
            "description": "Synthetic source",
            "value": 31,
        }

    @recipe("synthetic", "Read a registered source without opening a tab")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        value = session.context.read("device.synthetic.value")
        assert value == 31
        return Report(count=31)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.wait(3)
        outcome = execution.snapshot().outcome
        assert outcome is not None and outcome.report == Report(count=31)
    finally:
        client.context.session.close()


def test_interactive_analysis_hands_off_then_continues_after_owner_completion(
    tmp_path: Path,
) -> None:
    gui = SyntheticGui()
    finished = Event()

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.analyze":
            return {"operation_id": 100, "interactive": True}
        if (
            method == "operation.await"
            and params["operation_id"] == 100
            and not finished.is_set()
        ):
            return {"reason": "timeout"}
        return gui.respond(method, params)

    @recipe("synthetic", "Wait for the GUI-owned interactive operation")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        fit = await tab.analyze("primary", result=Fit, interactive=True)
        assert fit is not None and fit.result.estimate == 3.5
        return Report(count=inputs.count)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.wait(1)
        snapshot = execution.snapshot()
        assert snapshot.status == "interactive"
        handoff = snapshot.pending_analysis
        assert handoff is not None
        assert (handoff.tab, handoff.stage) == ("tab-1", "primary")
        assert handoff.op > 0
        finished.set()
        execution.join()
        assert execution.snapshot().status == "completed"
    finally:
        finished.set()
        client.context.session.close()


def test_save_preserves_raw_receipt_when_artifact_dispatch_fails(
    tmp_path: Path,
) -> None:
    gui = SyntheticGui()

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.save_artifacts":
            raise OSError("Artifact transport failed")
        return gui.respond(method, params)

    @recipe("synthetic", "Retain confirmed raw Save before a later failure")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        tab.save()
        return Report(count=inputs.count)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.wait(3)
        snapshot = execution.snapshot()
        assert snapshot.status == "failed"
        outcome = snapshot.outcome
        assert outcome is not None
        saved = outcome.runs[0].saves
        assert len(saved) == 1
        assert saved[0].status == "failed"
        assert saved[0].outputs[0].status == "saved"
        assert saved[0].outputs[0].format == "data_h5"
    finally:
        client.context.session.close()


def test_form_reads_keep_structured_owner_values_for_selector_and_sweep_helpers(
    tmp_path: Path,
) -> None:
    from ._recipe_support import LookbackGui

    @recipe(
        "synthetic", "Read complete Form observations without guessing native models"
    )
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("lookback")
        sweep = tab.read(("sweep",))
        assert {"start", "stop", "n"}.issubset(sweep.fields)
        assert sweep.value == {"kind": "linrange", "start": 1.0, "stop": 2.0, "n": 51}
        module = tab.read(("modules", "readout"))
        assert module.fields
        assert tab.read(("rounds",)).value == 3
        tab.set(("rounds",), inputs.count)
        assert tab.read(("rounds",)).value == inputs.count
        return Report(count=inputs.count)

    client = make_client(tmp_path, LookbackGui(), recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.wait(3)
        outcome = execution.snapshot().outcome
        assert outcome is not None and outcome.status == "completed"
    finally:
        client.context.session.close()


@pytest.mark.parametrize("terminal", ["completed", "failed", "cancelled"])
def test_recipe_tool_default_returns_only_authored_outcome(
    tmp_path: Path, terminal: str
) -> None:
    @recipe("synthetic", "Return a typed report")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        if terminal == "failed":
            raise ValueError("synthetic failure")
        if terminal == "cancelled":
            await session.ask(ConfirmQuestion(evidence="cancel here"))
        return Report(count=inputs.count)

    client = make_client(tmp_path, recipes=(synthetic,))
    try:
        if terminal == "cancelled":
            execution = start(client, "synthetic")
            execution.wait(3)
            client.call("cancel", {"execution": execution.snapshot().execution})
            execution.join()
            reply = client.call("wait", {"execution": execution.snapshot().execution})
        else:
            reply = client.call("synthetic", {"count": "12"})
        assert reply.data == {
            "status": terminal,
            "report": {"count": 12} if terminal == "completed" else None,
            "runs": [],
        }
    finally:
        client.context.session.close()


def test_full_recipe_status_retains_run_conditions_analysis_and_writeback(
    tmp_path: Path,
) -> None:
    gui = SyntheticGui()

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.get_analyze_result":
            return {
                "summary": {"estimate": 3.5},
                "invalid": [{"path": "uncertainty", "reason": "unavailable"}],
            }
        return gui.respond(method, params)

    @recipe("synthetic", "Capture detail without implicit saving")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        await tab.analyze("primary", result=Fit)
        tab.writeback_proposal()
        return Report(count=inputs.count)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.join()
        before = list(client.transport.sent)
        full = client.call(
            "status", {"execution": execution.snapshot().execution, "detail": "full"}
        )
        detail = full.get("details")
        assert detail, (
            "Full status must retain owner conditions, analysis and candidates"
        )
        run = detail[0]
        assert (run["tab"], run["adapter"], run["run_id"]) == (
            "tab-1",
            "synthetic",
            "native-1",
        )
        assert run["conditions"]["value"] == {"count": 9}
        assert run["analyses"][0]["summary"] == {"estimate": 3.5}
        assert run["analyses"][0]["invalid"] == [
            {"path": "uncertainty", "reason": "unavailable"}
        ]
        assert run["writeback"]["items"][0]["destination"]["path"] == "Q1.freq"
        assert client.transport.sent == before
        assert not any(method.startswith("tab.save") for method, _ in before)
    finally:
        client.context.session.close()


def test_recipe_full_preserves_native_estimate_uncertainty_and_operation_source(
    tmp_path: Path,
) -> None:
    gui = SyntheticGui()
    summary = {
        "estimate": 3.5,
        "stderr": 0.125,
        "unit": "native",
        "element": {"index": 2, "name": "Q3"},
    }

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.get_analyze_result":
            return {
                "summary": summary,
                "invalid": [{"path": "secondary", "reason": "non_finite"}],
            }
        return gui.respond(method, params)

    @recipe("synthetic", "Retain the analysis owner's native estimate facts")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        await tab.analyze("primary", result=Fit)
        return Report(count=inputs.count)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.join()
        full = client.call(
            "status", {"execution": execution.snapshot().execution, "detail": "full"}
        )
        detail = full["details"][0]
        assert (detail["tab"], detail["run_id"], detail["op"]) == (
            "tab-1",
            "native-1",
            1,
        )
        analysis = detail["analyses"][0]
        assert (analysis["stage"], analysis["op"]) == ("primary", 2)
        assert analysis["summary"] == {
            "estimate": 3.5,
            "stderr": 0.125,
            "unit": "native",
            "element": {"index": 2, "name": "Q3"},
        }
        assert analysis["invalid"] == [{"path": "secondary", "reason": "non_finite"}]
        summary["estimate"] = 99
        assert (
            client.call("status", {"execution": full["execution"], "detail": "full"})[
                "details"
            ][0]["analyses"][0]["summary"]["estimate"]
            == 3.5
        )
    finally:
        client.context.session.close()


@pytest.mark.parametrize("answer_state", ["stale", "none"])
def test_mcp_rejected_answer_preserves_question_and_reports_pending_id(
    tmp_path: Path, answer_state: str
) -> None:
    @recipe("synthetic", "Confirm one request only")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        answer = await session.ask(ConfirmQuestion(evidence="first"))
        assert answer is not None and answer.accepted
        return Report(count=inputs.count)

    client = make_client(tmp_path, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.wait(3)
        pending = execution.snapshot().pending_question
        assert pending is not None
        key = execution.snapshot().execution
        if answer_state == "none":
            execution.answer(
                {"request_id": pending.request_id, "answer": {"accepted": True}}
            )
        reply = client.call(
            "answer",
            {"execution": key, "request_id": "old-id", "answer": {"accepted": True}},
        )
        assert reply.data == {
            "accepted": False,
            "reason": "No pending question for this request id",
            "pending_request_id": pending.request_id
            if answer_state == "stale"
            else None,
        }
        if answer_state == "stale":
            assert execution.snapshot().pending_question == pending
            answered = client.call(
                "answer",
                {
                    "execution": key,
                    "request_id": pending.request_id,
                    "answer": {"accepted": True},
                },
            )
            assert answered.data == {
                "status": "completed",
                "report": {"count": 9},
                "runs": [],
            }
    finally:
        client.context.session.close()


def test_mcp_answer_requires_request_id_before_delivery(tmp_path: Path) -> None:
    @recipe("synthetic", "Keep malformed answers pending")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        await session.ask(ConfirmQuestion(evidence="first"))
        return Report(count=inputs.count)

    client = make_client(tmp_path, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.wait(3)
        with pytest.raises(ValueError, match="request_id"):
            client.call(
                "answer",
                {
                    "execution": execution.snapshot().execution,
                    "answer": {"accepted": True},
                },
            )
        assert execution.snapshot().pending_question is not None
    finally:
        client.context.session.close()


def test_mcp_run_cancel_without_run_returns_noop_ack(tmp_path: Path) -> None:
    @recipe("synthetic", "Keep explicit question pending")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        await session.ask(ConfirmQuestion(evidence="no run"))
        return Report(count=inputs.count)

    client = make_client(tmp_path, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.wait(3)
        pending = execution.snapshot().pending_question
        reply = client.call(
            "cancel", {"execution": execution.snapshot().execution, "scope": "run"}
        )
        assert reply.data.get("cancelled") is False
        assert reply.data["reason"] == "no run in progress"
        assert execution.snapshot().pending_question == pending
    finally:
        client.context.session.close()


def test_mcp_missing_parameters_does_not_create_failed_outcome(tmp_path: Path) -> None:
    @recipe("synthetic", "Report missing authored input")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        raise RecipeNeedsParameters((MissingParameter("frequency", "Provide MHz"),))

    client = make_client(tmp_path, recipes=(synthetic,))
    try:
        reply = client.call("synthetic", {"count": 9})
        assert reply.data == {
            "status": "needs_parameters",
            "missing": ["frequency"],
            "outcome": None,
        }
    finally:
        client.context.session.close()


def test_mcp_interactive_wait_and_status_project_same_pending_owner(
    tmp_path: Path,
) -> None:
    gui = SyntheticGui()
    finished = Event()

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.analyze":
            return {"operation_id": 100, "interactive": True}
        if (
            method == "operation.await"
            and params["operation_id"] == 100
            and not finished.is_set()
        ):
            return {"reason": "timeout"}
        return gui.respond(method, params)

    @recipe("synthetic", "Hand off the same GUI analysis operation")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        await tab.analyze("primary", result=Fit, interactive=True)
        return Report(count=inputs.count)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        reply = client.call("synthetic", {"count": 9})
        key = reply.data["execution"]
        assert reply.data.get("pending") == {
            "tab": "tab-1",
            "stage": "primary",
            "op": 2,
        }
        assert reply.data["status"] == "interactive"
        assert client.call("status", {"execution": key}) == {
            **reply.data,
            "active_path": "recipe:synthetic/operation#2",
            "live_paths": ["recipe:synthetic/operation#2"],
        }
        assert client.call("wait", {"execution": key, "timeout": 1}).data == reply.data
        finished.set()
        client.context.session.recipes.get(key).join()
        assert client.call("status", {"execution": key}) == {
            "status": "completed",
            "active_path": "recipe:synthetic",
            "live_paths": ["recipe:synthetic"],
            "report": {"count": 9},
            "runs": [
                {
                    "tab": "tab-1",
                    "adapter": "synthetic",
                    "run_id": "native-1",
                    "status": "completed",
                    "partial_result": False,
                    "saves": [],
                }
            ],
        }
    finally:
        finished.set()
        client.context.session.close()


def test_mapping_writeback_round_trips_without_guessing_or_mutating_owner_values(
    tmp_path: Path,
) -> None:
    gui = SyntheticGui()
    proposed = {"pulse": {"gain": [0.2, None], "shape": "synthetic"}}

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.writeback_preview":
            return {
                "has_draft": True,
                "revision": 1,
                "items": [
                    {
                        "id": "param-0",
                        "kind": "module",
                        "target_name": "Q1.freq",
                        "actual_path": "library:opaque.key",
                        "proposed": proposed,
                    }
                ],
                "destination_context": {"active_label": "current"},
            }
        return gui.respond(method, params)

    @recipe("synthetic", "Confirm an opaque persistent module value")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        await tab.analyze("primary", result=Fit)
        answer = await tab.ask_writeback()
        assert answer is not None
        receipt = tab.accept_writeback(answer.items)
        assert receipt.items[0].value == proposed
        return Report(count=inputs.count)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.wait(3)
        pending = execution.snapshot().pending_question
        assert pending is not None
        items = pending.question.model_dump(mode="json")["proposal"]["items"]
        assert items[0]["value"] == proposed
        reply = execution.answer(
            {
                "request_id": pending.request_id,
                "answer": {"accepted": True, "items": items},
            }
        )
        assert reply.data["outcome"]["status"] == "completed"
        assert proposed == {"pulse": {"gain": [0.2, None], "shape": "synthetic"}}
        writes = [
            params["stages"]
            for method, params in client.transport.sent
            if method == "tab.writeback_accept"
        ]
        assert writes == [
            [{"subtab_id": "analysis", "expected_revision": 1, "ids": ["param-0"]}]
        ]
    finally:
        client.context.session.close()
