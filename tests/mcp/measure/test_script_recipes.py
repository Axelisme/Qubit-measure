"""Synthetic declarations through the public measure tool registration seam."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from pydantic import BaseModel, Field, ValidationError
from zcu_tools.core.script_protocol import (
    Answer,
    Ask,
    OperationResult,
    Question,
    ScriptGenerator,
    Wait,
)
from zcu_tools.gui.form import FormModel
from zcu_tools.mcp.core.reply import ToolReply

from ._pipeline_recipe import PIPELINE
from ._support import MeasureClient, make_client


class Inputs(FormModel):
    count: int


class Report(BaseModel):
    status: str
    count: int


class Choice(Answer):
    proceed: bool = Field(description="Continue with the captured evidence when true.")


class Decision(Question[Choice]):
    evidence: str


class SyntheticOperation:
    name = "synthetic acquisition"

    def __init__(self) -> None:
        self.delivery: OperationResult[int] | None = None


def test_declaration_schema_session_and_authored_report(
    tmp_path: Path, clients: list[MeasureClient]
) -> None:
    from zcu_tools.mcp.measure.recipe_api import (
        RecipeGenerator,
        RecipeRun,
        RecipeSession,
        recipe,
    )
    from zcu_tools.mcp.measure.recipe_host import RecipeHost

    session = RecipeSession()
    report = Report(status="author-owned", count=9)

    @recipe("synthetic", description="Return an authored report")
    def synthetic(bound: RecipeSession, inputs: Inputs) -> RecipeGenerator[Report]:
        assert bound is session
        assert inputs.count == 9
        if False:
            yield
        return report

    host = RecipeHost(
        session=session, runs=lambda: (RecipeRun("tab-1", "fake", "run-1"),)
    )
    client = make_client(
        tmp_path, recipes=(synthetic,), recipe_host_factory=lambda _: host
    )
    clients.append(client)
    assert client.tools["synthetic"]["inputSchema"]["required"] == ["count"]
    reply = client.call("synthetic", {"count": 9})
    assert isinstance(reply, ToolReply)
    assert reply.data["outcome"] == {
        "status": "completed",
        "report": {"status": "author-owned", "count": 9},
        "runs": [{"tab": "tab-1", "adapter": "fake", "run_id": "run-1"}],
    }
    from zcu_tools.mcp.measure.script_recipe_execution import ScriptRecipeExecution

    execution = client.context.session.recipes.get(reply.data["execution"])
    assert isinstance(execution, ScriptRecipeExecution)
    outcome = execution.snapshot().outcome
    assert outcome is not None
    assert outcome.report is report
    assert client.transport.sent == []


def test_pending_question_validates_before_resume_and_blocks_other_recipes(
    tmp_path: Path, clients: list[MeasureClient]
) -> None:
    from zcu_tools.mcp.measure.recipe_api import RecipeGenerator, RecipeSession, recipe

    resumed: list[Choice] = []

    @recipe("decision", description="Ask explicitly")
    def synthetic(session: RecipeSession, inputs: Inputs) -> RecipeGenerator[Report]:
        request = Ask(Decision(evidence="captured only"))
        yield request
        assert request.delivery is not None
        resumed.append(request.delivery)
        return Report(status="answered", count=inputs.count)

    client = make_client(tmp_path, recipes=(synthetic, PIPELINE))
    clients.append(client)
    first = client.call("decision", {"count": 3})
    key = first.data["execution"]
    status = client.call("status", {"execution": key})
    assert status["status"] == "paused"
    assert status["pending_question"]["question"] == {"evidence": "captured only"}
    assert (
        client.call("wait", {"execution": key, "timeout": 0}).data["status"] == "paused"
    )
    assert resumed == []
    from zcu_tools.mcp.measure.session import GuiRpcError

    with pytest.raises(GuiRpcError) as busy:
        client.call(PIPELINE.name, {})
    assert busy.value.reason == "recipe_busy"
    for invalid in ({"proceed": "not-a-bool"}, {"proceed": True, "extra": 1}):
        with pytest.raises(ValidationError):
            client.call("answer", {"execution": key, "answer": invalid})
        assert client.call("status", {"execution": key})["status"] == "paused"
        assert resumed == []
    answered = client.call("answer", {"execution": key, "answer": {"proceed": False}})
    assert answered.data["outcome"]["report"] == {"status": "answered", "count": 3}
    assert resumed == [Choice(proceed=False)]
    with pytest.raises(RuntimeError, match="pending question"):
        client.call("answer", {"execution": key, "answer": {"proceed": True}})
    assert client.transport.sent == []


@pytest.mark.parametrize("operation_status", ["completed", "cancelled", "failed"])
def test_wait_delivery_does_not_override_generator_completion(
    tmp_path: Path, clients: list[MeasureClient], operation_status: str
) -> None:
    from zcu_tools.core.script_protocol import OperationStatus
    from zcu_tools.mcp.measure.recipe_api import RecipeGenerator, RecipeSession, recipe
    from zcu_tools.mcp.measure.recipe_host import RecipeHost

    operation = SyntheticOperation()
    statuses: dict[str, OperationStatus] = {
        "completed": "completed",
        "cancelled": "cancelled",
        "failed": "failed",
    }

    def wait(op: object, cancellation: Event) -> None:
        assert op is operation
        operation.delivery = OperationResult(
            statuses[operation_status], 17, "diagnostic"
        )

    @recipe("work", description="Wait for host-owned work")
    def synthetic(session: RecipeSession, inputs: Inputs) -> RecipeGenerator[Report]:
        yield Wait(operation)
        assert operation.delivery is not None
        return Report(
            status=operation.delivery.status, count=operation.delivery.result or 0
        )

    host = RecipeHost(session=RecipeSession(), wait=wait)
    client = make_client(
        tmp_path, recipes=(synthetic,), recipe_host_factory=lambda _: host
    )
    clients.append(client)
    reply = client.call("work", {"count": 1})
    assert reply.data["outcome"]["status"] == "completed"
    assert reply.data["outcome"]["report"] == {"status": operation_status, "count": 17}


def test_cancel_wait_drains_operation_and_closes_generator(
    tmp_path: Path, clients: list[MeasureClient]
) -> None:
    from zcu_tools.mcp.measure.recipe_api import RecipeGenerator, RecipeSession, recipe
    from zcu_tools.mcp.measure.recipe_host import RecipeHost

    operation = SyntheticOperation()
    entered, drained, closed = Event(), Event(), Event()

    def wait(op: object, cancellation: Event) -> None:
        entered.set()
        assert cancellation.wait(3)
        operation.delivery = OperationResult("cancelled", None)
        drained.set()

    def cancel(op: object) -> None:
        assert op is operation

    @recipe("work", description="Cancel suspended work")
    def synthetic(session: RecipeSession, inputs: Inputs) -> RecipeGenerator[Report]:
        try:
            yield Ask(Decision(evidence="start synthetic operation"))
            yield Wait(operation)
            return Report(status="unexpected resume", count=0)
        finally:
            assert drained.is_set()
            closed.set()

    host = RecipeHost(session=RecipeSession(), wait=wait, cancel=cancel)
    client = make_client(
        tmp_path, recipes=(synthetic,), recipe_host_factory=lambda _: host
    )
    clients.append(client)
    key = client.call("work", {"count": 1}).data["execution"]
    with ThreadPoolExecutor(max_workers=1) as worker:
        continuation = worker.submit(
            client.call, "answer", {"execution": key, "answer": {"proceed": True}}
        )
        try:
            assert entered.wait(3)
            client.call("cancel", {"execution": key})
            result = client.call("wait", {"execution": key, "timeout": 3})
            assert result.data["outcome"] == {
                "status": "cancelled",
                "report": None,
                "runs": [],
            }
            assert closed.is_set()
            assert drained.is_set()
        finally:
            client.context.session.close()
        assert continuation.result(timeout=3).data["status"] == "cancelled"


def test_failure_and_default_convenience_stub_are_failed_outcomes(
    tmp_path: Path, clients: list[MeasureClient]
) -> None:
    from zcu_tools.mcp.measure.recipe_api import RecipeGenerator, RecipeSession, recipe

    @recipe("stub", description="No convenience implementation")
    def synthetic(session: RecipeSession, inputs: Inputs) -> RecipeGenerator[Report]:
        session.open_tab("synthetic")
        if False:
            yield
        return None

    client = make_client(tmp_path, recipes=(synthetic,))
    clients.append(client)
    reply = client.call("stub", {"count": 1})
    assert reply.data["outcome"] == {"status": "failed", "report": None, "runs": []}
    assert reply.is_error
    assert reply.data["error"]["type"] == "NotImplementedError"
    assert client.transport.sent == []


def test_invalid_inputs_fail_before_host_binding(
    tmp_path: Path, clients: list[MeasureClient]
) -> None:
    from zcu_tools.mcp.measure.recipe_api import RecipeGenerator, RecipeSession, recipe
    from zcu_tools.mcp.measure.recipe_host import RecipeHost

    bound: list[object] = []

    def factory(context: object) -> RecipeHost:
        bound.append(context)
        return RecipeHost(session=RecipeSession())

    @recipe("validated", description="Validate inputs first")
    def synthetic(session: RecipeSession, inputs: Inputs) -> RecipeGenerator[Report]:
        if False:
            yield
        return Report(status="ok", count=inputs.count)

    client = make_client(tmp_path, recipes=(synthetic,), recipe_host_factory=factory)
    clients.append(client)
    reply = client.call("validated", {"count": "wrong"})
    assert reply.data["status"] == "failed"
    assert reply.data["error"]["type"] == "ValidationError"
    assert bound == []


def test_decorator_rejects_invalid_signatures_without_execution() -> None:
    from zcu_tools.mcp.measure.recipe_api import recipe

    with pytest.raises(ValueError, match="nonblank"):
        recipe("", description="invalid")

    def not_generator(session: object, inputs: Inputs) -> ScriptGenerator[Report]:
        raise AssertionError("declaration must not execute")

    with pytest.raises(TypeError, match="generator"):
        recipe("invalid", description="invalid")(not_generator)

    def missing_inputs(*args: object) -> ScriptGenerator[Report]:
        if False:
            yield
        return Report(status="invalid", count=0)

    with pytest.raises(TypeError, match="two"):
        recipe("invalid", description="invalid")(missing_inputs)


def test_failed_host_work_closes_suspended_generator_before_terminal_reply(
    tmp_path: Path, clients: list[MeasureClient]
) -> None:
    from zcu_tools.mcp.measure.recipe_api import RecipeGenerator, RecipeSession, recipe
    from zcu_tools.mcp.measure.recipe_host import RecipeHost

    closed = Event()
    operation = SyntheticOperation()

    def wait(op: object, cancellation: Event) -> None:
        raise RuntimeError("host work failed after drain")

    @recipe("failure", description="Host failure closes the generator")
    def synthetic(session: RecipeSession, inputs: Inputs) -> RecipeGenerator[Report]:
        try:
            yield Wait(operation)
            return Report(status="unexpected continuation", count=0)
        finally:
            closed.set()

    host = RecipeHost(session=RecipeSession(), wait=wait)
    client = make_client(
        tmp_path, recipes=(synthetic,), recipe_host_factory=lambda _: host
    )
    clients.append(client)
    result = client.call("failure", {"count": 1})
    assert result.data["outcome"]["status"] == "failed"
    assert result.data["error"]["message"] == "host work failed after drain"
    assert closed.is_set()


def test_pending_question_exposes_answer_schema_without_overwriting_evidence(
    tmp_path: Path, clients: list[MeasureClient]
) -> None:
    from zcu_tools.mcp.measure.recipe_api import RecipeGenerator, RecipeSession, recipe

    class Evidence(Question[Choice]):
        question_type: str
        answer_schema: str

    @recipe("schema", description="Expose the answer contract")
    def synthetic(session: RecipeSession, inputs: Inputs) -> RecipeGenerator[Report]:
        request = Ask(
            Evidence(question_type="authored type", answer_schema="authored evidence")
        )
        yield request
        assert request.delivery is not None
        return Report(status="answered", count=int(request.delivery.proceed))

    client = make_client(tmp_path, recipes=(synthetic,))
    clients.append(client)
    key = client.call("schema", {"count": 1}).data["execution"]
    pending = client.call("status", {"execution": key})["pending_question"]
    assert "answer_schema" in pending
    assert pending["question_type"] == "Evidence"
    assert pending["question"] == {
        "question_type": "authored type",
        "answer_schema": "authored evidence",
    }
    schema = pending["answer_schema"]
    assert schema["required"] == ["proceed"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["proceed"]["type"] == "boolean"
    assert schema["properties"]["proceed"]["description"] == (
        "Continue with the captured evidence when true."
    )
    reply = client.call(
        "answer",
        {"execution": key, "answer": {name: True for name in schema["required"]}},
    )
    assert reply.data["outcome"]["report"] == {"status": "answered", "count": 1}
