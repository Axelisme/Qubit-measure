"""Async recipe contracts through declarations and a recording GUI transport."""

from collections.abc import Callable
from pathlib import Path
from threading import Event, get_ident

import pytest
from pydantic import BaseModel, Field, ValidationError
from zcu_tools.experiment.script_host import CancelScope
from zcu_tools.gui.form import FormModel
from zcu_tools.mcp.measure.recipe_api import RecipeSession, recipe
from zcu_tools.mcp.measure.script_recipe_execution import ScriptRecipeExecution

from ._support import MeasureClient, make_client


class Inputs(FormModel):
    count: int = Field(description="Synthetic repetition count")


class Report(BaseModel):
    count: int


class Fit(BaseModel):
    estimate: float


class SyntheticGui:
    """A wire collaborator; no production recipe or instrument dependencies."""

    def __init__(self, *, hold: bool = False) -> None:
        self.hold = hold
        self.started = Event()
        self.cancelled = Event()
        self.runs = 0

    def respond(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.new":
            return {"tab_id": "tab-1"}
        if method in {
            "tab.reset_cfg",
            "tab.edit_recipe_fields",
            "tab.select_recipe_module",
        }:
            return {"errors": [], "changed": True}
        return self._operation(method, params)

    def _operation(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.run_start":
            self.runs += 1
            self.started.set()
            return {"operation_id": self.runs}
        if method in {"tab.analyze", "tab.post_analyze"}:
            return {"operation_id": 100}
        if method == "operation.await":
            if (
                self.hold
                and params["operation_id"] != 100
                and not self.cancelled.is_set()
            ):
                return {"reason": "timeout"}
            return {
                "reason": "completed",
                "status": "cancelled"
                if self.cancelled.is_set() and params["operation_id"] != 100
                else "finished",
            }
        if method in {"tab.cancel", "tab.analyze_cancel", "operation.cancel"}:
            self.cancelled.set()
            return {"status": "requested"}
        return self._snapshots(method, params)

    def _snapshots(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.snapshot":
            return {
                "tabs": [
                    {
                        "tab_id": "tab-1",
                        "adapter_name": "synthetic",
                        "interaction": {"is_running": False, "is_analyzing": False},
                        "raw_data": {"run_id": f"native-{self.runs}", "saved": False},
                        "result_state": {
                            "available": True,
                            "source_operation_id": self.runs,
                        },
                    }
                ]
            }
        return self._artifacts(method, params)

    def _artifacts(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.writeback_preview":
            return {
                "has_draft": True,
                "items": [
                    {
                        "id": "param-0",
                        "target_name": "Q1.freq",
                        "actual_path": "Q1.freq",
                        "proposed": 3.5,
                        "current": 2.0,
                        "stderr": None,
                        "note": None,
                        "kind": "parameter",
                    }
                ],
                "destination_context": {"active_label": "preview-context"},
            }
        if method == "tab.writeback_write":
            assert set(params) == {"tab_id", "subtab_id", "write"}
            return {"written": [{"id": "param-0", "target": "Q1.freq"}]}
        if method == "tab.get_analyze_result":
            return {"summary": {"estimate": 3.5}, "invalid": []}
        if method in {"tab.save_data", "tab.save_artifacts"}:
            section = "measurement_data" if method == "tab.save_data" else "analysis"
            return {
                "run_id": f"native-{self.runs}",
                "artifacts": {
                    section: {
                        "results": {
                            "status": "Saved",
                            "members": {
                                "data": [
                                    {
                                        "format": "data_h5"
                                        if method == "tab.save_data"
                                        else "json",
                                        "path": "/synthetic/output",
                                        "status": "saved",
                                        "error": None,
                                    }
                                ]
                            },
                        }
                    }
                },
            }
        raise AssertionError(f"Unexpected GUI request: {method}")


def start(client: MeasureClient, name: str, count: int = 9) -> ScriptRecipeExecution:
    """Start through the shared registry, asserting the declaration route."""
    execution = client.context.session.recipes.start(
        client.context, name, {"count": count}
    )
    assert isinstance(execution, ScriptRecipeExecution)
    return execution


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
        assert fit is not None
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


@pytest.mark.parametrize("scope", ["script", "run"])
def test_cancel_delivers_partial_and_runs_cleanup_on_author_worker(
    tmp_path: Path, scope: CancelScope
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
            if scope == "script":
                partial.append((await tab.run()).partial_result)
                assert await tab.analyze("primary", result=Fit) is None
            else:
                fit = await tab.analyze("primary", result=Fit)
                assert fit is not None
            return Report(count=inputs.count)
        finally:
            threads.append(get_ident())

    client = make_client(tmp_path, gui.respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        assert gui.started.wait(3)
        execution.cancel(scope=scope)
        execution.wait(3)
        snapshot = execution.snapshot()
        outcome = snapshot.outcome
        assert outcome is not None
        assert outcome.status == ("cancelled" if scope == "script" else "completed")
        assert outcome.report == (None if scope == "script" else Report(count=9))
        assert partial == ([True, True] if scope == "script" else [True])
        assert len(outcome.runs) == (2 if scope == "script" else 1)
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
        assert methods.index("tab.writeback_write") < methods.index("tab.save_data")
    finally:
        client.context.session.close()


def test_analysis_summary_missing_authored_field_fails_without_repair(
    tmp_path: Path,
) -> None:
    class RequiredFit(BaseModel):
        absent: float

    @recipe("synthetic", "Decode the exact owner summary")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        await tab.analyze("primary", result=RequiredFit)
        return Report(count=inputs.count)

    client = make_client(tmp_path, SyntheticGui().respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.wait(3)
        snapshot = execution.snapshot()
        assert snapshot.status == "failed"
        assert snapshot.error is not None
        assert snapshot.error.type == "ValidationError"
        assert "absent" in snapshot.error.message
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
        assert sweep.value == {"start": 1.0, "stop": 2.0, "n": 51}
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


def test_mapping_writeback_round_trips_without_guessing_or_mutating_owner_values(
    tmp_path: Path,
) -> None:
    gui = SyntheticGui()
    proposed = {"pulse": {"gain": [0.2, None], "shape": "synthetic"}}

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.writeback_preview":
            return {
                "has_draft": True,
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
            params["write"]
            for method, params in client.transport.sent
            if method == "tab.writeback_write"
        ]
        assert writes == [[{"id": "param-0"}]]
    finally:
        client.context.session.close()
