"""Synthetic scripts exercise the shared host-facing driver contract."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from zcu_tools.core.script_protocol import (
        OperationResult,
        OperationStatus,
        ScriptRequest,
    )


@dataclass
class SyntheticOperation:
    name: str = "synthetic"
    delivery: OperationResult[int] | None = None


def deliver_cancelled(operation: object) -> None:
    """Deliver without starting work, as the synthetic operation's owner."""
    from zcu_tools.core.script_protocol import OperationResult

    if not isinstance(operation, SyntheticOperation):
        raise TypeError("Unknown synthetic operation")
    operation.delivery = OperationResult("cancelled", None)


@pytest.mark.parametrize("status", ["completed", "cancelled", "failed"])
def test_wait_delivers_terminal_outcome_and_resumes_only_with_none(
    status: OperationStatus,
) -> None:
    from zcu_tools.core.script_protocol import OperationResult, Wait
    from zcu_tools.experiment.script_driver import ScriptDriver

    op = SyntheticOperation()
    delivery = OperationResult(
        status, 17, "operation failed" if status == "failed" else None
    )
    resumes: list[None] = []

    def script() -> Generator[ScriptRequest, None, OperationResult[int] | None]:
        resumes.append((yield Wait(op)))
        return op.delivery

    driver = ScriptDriver(script(), deliver_cancelled=deliver_cancelled)
    assert driver.status == "running"
    driver.advance()
    assert driver.status == "waiting"
    assert driver.pending_request == Wait(op)
    driver.advance()
    assert resumes == []
    op.delivery = delivery
    driver.advance()
    assert resumes == [None]
    assert driver.status == "completed"
    assert driver.result is delivery
    assert driver.error is None
    assert driver.pending_request is None


def test_ask_waits_without_default_and_wrong_answer_preserves_question() -> None:
    from zcu_tools.core.script_protocol import Answer, Ask, Question
    from zcu_tools.experiment.script_driver import ScriptDriver

    class Approval(Answer):
        accepted: bool

    class Other(Answer):
        note: str

    class Confirm(Question[Approval]):
        evidence: str

    question = Confirm(evidence="synthetic evidence")
    request = Ask(question)
    resumes: list[None] = []

    def script() -> Generator[ScriptRequest, None, Approval | None]:
        resumes.append((yield request))
        return request.delivery

    driver = ScriptDriver(script(), deliver_cancelled=deliver_cancelled)
    driver.advance()
    driver.advance()
    assert driver.status == "paused"
    assert driver.pending_question is question
    assert request.delivery is None
    with pytest.raises(TypeError, match="Approval"):
        driver.answer(Other(note="wrong model"))
    assert driver.status == "paused"
    assert driver.pending_question is question
    assert resumes == []
    approval = Approval(accepted=True)
    driver.answer(approval)
    assert request.delivery is approval
    assert resumes == [None]
    assert driver.result is approval
    assert driver.status == "completed"
    assert driver.pending_question is None
    with pytest.raises(RuntimeError, match="question"):
        driver.answer(approval)


def test_unparameterized_question_cannot_be_answered() -> None:
    from zcu_tools.core.script_protocol import Answer, Ask, Question
    from zcu_tools.experiment.script_driver import ScriptDriver

    # Deliberately model an authored Question without its required Answer parameter.
    Unspecified = type(
        "Unspecified", (Question,), {"__annotations__": {"evidence": str}}
    )

    def script() -> Generator[ScriptRequest, None, None]:
        yield Ask(Unspecified.model_validate({"evidence": "missing answer contract"}))

    driver = ScriptDriver(script(), deliver_cancelled=deliver_cancelled)
    driver.advance()
    with pytest.raises(TypeError, match=r"Question\[.*Answer"):
        driver.answer(Answer())
    assert driver.status == "paused"
    assert driver.pending_question is not None


def test_inherited_question_accepts_subclass_of_declared_answer() -> None:
    from zcu_tools.core.script_protocol import Answer, Ask, Question
    from zcu_tools.experiment.script_driver import ScriptDriver

    class Approval(Answer):
        accepted: bool

    class DetailedApproval(Approval):
        reason: str

    class Confirm(Question[Approval]):
        evidence: str

    class NamedConfirm(Confirm):
        name: str

    request = Ask(NamedConfirm(evidence="ready", name="confirm"))

    def script() -> Generator[ScriptRequest, None, Approval | None]:
        yield request
        return request.delivery

    driver = ScriptDriver(script(), deliver_cancelled=deliver_cancelled)
    driver.advance()
    answer = DetailedApproval(accepted=True, reason="reviewed")
    driver.answer(answer)
    assert driver.result is answer


def test_signal_delivery_occurs_at_yield_boundaries_in_order() -> None:
    from zcu_tools.core.script_protocol import Signal, Wait
    from zcu_tools.experiment.script_driver import ScriptDriver

    op = SyntheticOperation()
    events: list[str] = []

    def script() -> Generator[ScriptRequest, None, None]:
        events.append("before yield")
        driver.signal(Signal("stop"))
        assert events == ["before yield"]
        yield Wait(op)
        events.append("resumed")

    driver = ScriptDriver(
        script(),
        deliver_cancelled=deliver_cancelled,
        on_signal=lambda signal: events.append(signal.kind),
    )
    driver.signal(Signal("pause"))
    assert events == []
    driver.advance()
    assert events == ["before yield", "pause", "stop"]
    assert driver.status == "waiting"
    driver.signal(Signal("finish_early"))
    assert events == ["before yield", "pause", "stop"]
    driver.advance()
    assert events == ["before yield", "pause", "stop", "finish_early"]


@pytest.mark.parametrize("boundary", ["wait", "ask"])
@pytest.mark.parametrize("operation_status", ["completed", "cancelled"])
def test_cancel_resumes_author_and_cancels_later_requests_without_starting_work(
    boundary: str, operation_status: OperationStatus
) -> None:
    from zcu_tools.core.script_protocol import (
        Answer,
        Ask,
        OperationResult,
        Question,
        Signal,
        Wait,
    )
    from zcu_tools.experiment.script_driver import ScriptDriver

    op, later = SyntheticOperation("active"), SyntheticOperation("later")
    question = Ask(Question[Answer]())
    later_question = Ask(Question[Answer]())
    observed: list[object] = []
    deliveries: list[object] = []
    report = {"saved": 7}

    def cancelled(operation: object) -> None:
        deliveries.append(operation)
        deliver_cancelled(operation)

    def script() -> Generator[ScriptRequest, None, dict[str, int]]:
        yield Wait(op) if boundary == "wait" else question
        observed.append(op.delivery if boundary == "wait" else question.delivery)
        yield Wait(later)
        observed.append(later.delivery)
        yield later_question
        observed.append(later_question.delivery)
        return report

    driver = ScriptDriver(script(), deliver_cancelled=cancelled)
    driver.advance()
    # Host has cancelled and drained existing work, including a completion race.
    terminal = OperationResult(operation_status, 7)
    op.delivery = terminal
    driver.signal(Signal("cancel"))
    assert observed == []
    driver.advance()
    assert observed == [
        terminal if boundary == "wait" else None,
        OperationResult("cancelled", None),
        None,
    ]
    assert deliveries == [later]
    assert driver.status == "cancelled"
    assert driver.result is report
    assert driver.pending_request is None
    driver.advance()
    assert deliveries == [later]


def test_cancel_requested_during_execution_waits_until_yield() -> None:
    from zcu_tools.core.script_protocol import Signal, Wait
    from zcu_tools.experiment.script_driver import ScriptDriver

    events: list[str] = []

    def script() -> Generator[ScriptRequest, None, None]:
        try:
            driver.signal(Signal("cancel"))
            events.append("still executing")
            yield Wait(SyntheticOperation())
            events.append("cancelled resume")
        finally:
            events.append("closed")

    driver = ScriptDriver(script(), deliver_cancelled=deliver_cancelled)
    driver.advance()
    assert events == ["still executing", "cancelled resume", "closed"]
    assert driver.status == "cancelled"


def test_cancel_does_not_resume_active_wait_until_host_terminal_delivery() -> None:
    from zcu_tools.core.script_protocol import OperationResult, Signal, Wait
    from zcu_tools.experiment.script_driver import ScriptDriver

    operation = SyntheticOperation()
    received: list[object] = []

    def script() -> Generator[ScriptRequest, None, None]:
        yield Wait(operation)
        received.append(operation.delivery)

    driver = ScriptDriver(script(), deliver_cancelled=deliver_cancelled)
    driver.advance()
    driver.signal(Signal("cancel"))
    driver.advance()
    assert driver.status == "waiting"
    assert received == []
    operation.delivery = OperationResult("cancelled", 4)
    driver.advance()
    assert received == [OperationResult("cancelled", 4)]
    assert driver.status == "cancelled"


@pytest.mark.parametrize("invalid_delivery", [None, "completed"])
def test_cancelled_delivery_owner_must_supply_cancelled_terminal_result(
    invalid_delivery: str | None,
) -> None:
    from zcu_tools.core.script_protocol import OperationResult, Signal, Wait
    from zcu_tools.experiment.script_driver import ScriptDriver

    operation = SyntheticOperation()

    def broken_owner(op: object) -> None:
        assert op is operation
        if invalid_delivery is not None:
            operation.delivery = OperationResult("completed", 3)

    def script() -> Generator[ScriptRequest, None, None]:
        driver.signal(Signal("cancel"))
        yield Wait(operation)

    driver = ScriptDriver(script(), deliver_cancelled=broken_owner)
    with pytest.raises(RuntimeError, match="cancelled.*delivery"):
        driver.advance()
    assert driver.status == "failed"


def test_script_failure_preserves_original_error_and_terminal_state() -> None:
    from zcu_tools.core.script_protocol import OperationResult, Wait
    from zcu_tools.experiment.script_driver import ScriptDriver

    error = ValueError("authored failure")

    def script() -> Generator[ScriptRequest, None, None]:
        yield Wait(SyntheticOperation(delivery=OperationResult("completed", 1)))
        raise error

    driver = ScriptDriver(script(), deliver_cancelled=deliver_cancelled)
    with pytest.raises(ValueError, match="authored failure") as caught:
        driver.advance()
    assert caught.value is error
    assert driver.error is error
    assert driver.status == "failed"
    assert driver.pending_request is None


def test_generator_exit_is_cancelled_not_failed() -> None:
    from zcu_tools.experiment.script_driver import ScriptDriver

    def script() -> Generator[ScriptRequest, None, None]:
        yield from ()
        raise GeneratorExit

    driver = ScriptDriver(script(), deliver_cancelled=deliver_cancelled)
    driver.advance()
    assert driver.status == "cancelled"
    assert driver.error is None


def test_close_failure_is_failed_and_preserves_cleanup_error() -> None:
    from zcu_tools.core.script_protocol import Signal, Wait
    from zcu_tools.experiment.script_driver import ScriptDriver

    def script() -> Generator[ScriptRequest, None, None]:
        try:
            yield Wait(SyntheticOperation())
        finally:
            raise ValueError("cleanup failed")

    driver = ScriptDriver(script(), deliver_cancelled=deliver_cancelled)
    driver.advance()
    driver.signal(Signal("cancel"))
    with pytest.raises(ValueError, match="cleanup failed") as caught:
        driver.advance()
    assert driver.status == "failed"
    assert driver.error is caught.value


def test_unsupported_request_fails_instead_of_suspending() -> None:
    from zcu_tools.core.script_protocol import ScriptRequest
    from zcu_tools.experiment.script_driver import ScriptDriver

    def script() -> Generator[ScriptRequest, None, None]:
        yield ScriptRequest()

    driver = ScriptDriver(script(), deliver_cancelled=deliver_cancelled)
    with pytest.raises(TypeError, match="Wait.*Ask"):
        driver.advance()
    assert driver.status == "failed"


def test_return_payload_is_not_projected_or_copied() -> None:
    from zcu_tools.experiment.script_driver import ScriptDriver

    report = {"named_report": [3, 5]}

    def script() -> Generator[ScriptRequest, None, dict[str, list[int]]]:
        yield from ()
        return report

    driver = ScriptDriver(script(), deliver_cancelled=deliver_cancelled)
    driver.advance()
    assert driver.status == "completed"
    assert driver.result is report
    driver.advance()
    assert driver.result is report


def test_parse_answer_only_validates_and_preserves_pending_delivery() -> None:
    from pydantic import ValidationError
    from zcu_tools.core.script_protocol import Answer, Ask, Question
    from zcu_tools.experiment.script_driver import ScriptDriver

    class Approval(Answer):
        accepted: bool

    class Confirm(Question[Approval]):
        evidence: str

    request = Ask(Confirm(evidence="captured"))

    def script() -> Generator[ScriptRequest, None, Approval | None]:
        yield request
        return request.delivery

    driver = ScriptDriver(script(), deliver_cancelled=deliver_cancelled)
    with pytest.raises(RuntimeError, match="pending question"):
        driver.parse_answer({"accepted": True})
    driver.advance()
    for invalid in ({"accepted": "wrong"}, {"accepted": True, "unknown": 1}):
        with pytest.raises(ValidationError, match="Confirm"):
            driver.parse_answer(invalid)
        assert driver.status == "paused"
        assert request.delivery is None
    answer = driver.parse_answer({"accepted": False})
    assert answer == Approval(accepted=False)
    assert driver.status == "paused"
    assert request.delivery is None
    driver.answer(answer)
    assert driver.result is answer


def test_validate_answer_does_not_deliver_or_resume() -> None:
    from zcu_tools.core.script_protocol import Answer, Ask, Question
    from zcu_tools.experiment.script_driver import ScriptDriver

    class Approval(Answer):
        accepted: bool

    class Other(Answer):
        message: str

    class Confirm(Question[Approval]):
        evidence: str

    request = Ask(Confirm(evidence="captured"))

    def script() -> Generator[ScriptRequest, None, Approval | None]:
        yield request
        return request.delivery

    driver = ScriptDriver(script(), deliver_cancelled=deliver_cancelled)
    approval = Approval(accepted=True)
    with pytest.raises(RuntimeError, match="pending question"):
        driver.validate_answer(approval)
    driver.advance()
    with pytest.raises(TypeError, match="Confirm.*Approval"):
        driver.validate_answer(Other(message="wrong model"))
    driver.validate_answer(approval)
    assert request.delivery is None
    assert driver.status == "paused"
    driver.answer(approval)
    assert driver.result is approval


def test_pending_answer_type_is_read_only_and_clears_at_terminal() -> None:
    from zcu_tools.core.script_protocol import Answer, Ask, Question
    from zcu_tools.experiment.script_driver import ScriptDriver

    class Approval(Answer):
        accepted: bool

    class Confirm(Question[Approval]):
        evidence: str

    request = Ask(Confirm(evidence="captured"))

    def script() -> Generator[ScriptRequest, None, Approval | None]:
        yield request
        return request.delivery

    driver = ScriptDriver(script(), deliver_cancelled=deliver_cancelled)
    assert driver.pending_answer_type() is None
    driver.advance()
    assert driver.pending_answer_type() is Approval
    assert driver.status == "paused"
    assert request.delivery is None
    driver.answer(Approval(accepted=True))
    assert driver.pending_answer_type() is None
