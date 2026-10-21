"""Native Ask through the registered headless facet and retained operation owner."""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import get_ident

import pytest
from pydantic import ValidationError
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.experiment.workflows import Actor, InvalidRunState, RunMismatch
from zcu_tools.gui.app.measure.services.workflow_control import WorkflowControlFacet
from zcu_tools.gui.form import FormErrors
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.ports import OperationConflictError
from zcu_tools.resources.entry import ResultEntry

from tests._filesystem import DIRECTORY_OPEN_ERROR
from tests.gui.app.measure._workflow_ask import AskAuthor, Decision, OtherAnswer

from ._workflow_harness import StartHarness


@pytest.fixture
def asking(tmp_path):
    harness = StartHarness()
    entry = ResultEntry.create(
        "synthetic",
        result_root=tmp_path / "result",
        database_root=tmp_path / "database",
    )
    entry.new_point("source", description="synthetic")
    harness.services.context.install_entry(entry)
    harness.services.context.use_point(PointUseRequest("source"))
    board = white_noise()
    harness.services.soc_connection.install_soc(board.soc, board.soccfg)
    author = AskAuthor()
    control = WorkflowControlFacet(harness.state, harness.services, harness.owner)
    control.configure((author.attachment(),))
    tab = control.open("asking")
    token = control.start(tab, "question-run")
    assert not isinstance(token, FormErrors)
    harness.complete_segment()
    try:
        yield control, harness, author, tab, token
    finally:
        control.stop("question-run")
        if harness.background.work is not None:
            harness.complete_segment()
        control.close(tab)


def test_answer_keeps_invocation_operation_and_lease_until_owner_delivery(asking):
    control, harness, author, tab, token = asking
    question = control.pending_question("question-run")
    assert question is not None, "Facet must publish the native Ask envelope"
    assert question["question_type"] == "Choose"
    assert question["question"] == {"question_type": "authored evidence", "point": 4}
    assert question["answer_schema"]["properties"]["value"]["type"] == "integer"
    control.apply_tunables("question-run", {("value",): 9}, actor=Actor("user", "test"))
    assert control.answer("question-run", {"value": 23}) == token
    assert author.deliveries == [], "Submission must not execute author code inline"
    with pytest.raises(InvalidRunState):
        control.answer("question-run", {"value": 24})
    other = control.open("asking")
    try:
        with ThreadPoolExecutor(max_workers=1) as worker:
            worker.submit(harness.background.execute).result(timeout=30)
        assert author.captures == [3]
        assert author.deliveries == [(Decision(value=23), author.deliveries[0][1])]
        assert author.deliveries[0][1] != get_ident()
        assert control.tabs()[0].operation_id == token
        assert not control.close(tab)
        with pytest.raises(OperationConflictError):
            control.start(other, "competing")
        harness.owner.pump_all()
        harness.background.deliver()
        assert control.tabs()[0].operation_id is None
        assert control.records("question-run")[0].revision == 0
        assert control.records("question-run")[0].encoded_record.value == 23
        assert harness.handles.known_outcome(token).status == "finished"
        replacement = control.start(other, "after-delivery")
        assert isinstance(replacement, int)
        harness.complete_segment()
        control.stop("after-delivery")
        harness.complete_segment()
    finally:
        control.close(other)


def test_wrong_typed_answer_delivery_stays_paused_and_can_retry(asking, tmp_path):
    control, harness, author, tab, token = asking
    assert control.answer("question-run", OtherAnswer(value=7)) == token
    harness.background.execute()
    assert isinstance(harness.background.error, TypeError)
    harness.owner.pump_all()
    harness.background.deliver()
    assert control.status("question-run").lifecycle == "paused"
    assert control.tabs()[0].operation_id == token
    assert control.pending_question("question-run") is not None
    assert not control.close(tab)
    assert harness.handles.known_outcome(token) is None
    journal = next(tmp_path.rglob("journal.jsonl"))
    assert "resumed" not in [
        json.loads(line)["kind"] for line in journal.read_text().splitlines()
    ]
    assert author.deliveries == []
    assert control.answer("question-run", Decision(value=8)) == token
    harness.complete_segment()
    assert control.status("question-run").lifecycle == "done"
    assert harness.handles.known_outcome(token).status == "finished"


def test_invalid_json_answer_is_rejected_without_consuming_question(asking):
    control, harness, author, tab, token = asking
    with pytest.raises(ValidationError, match="Choose"):
        control.answer("question-run", {"value": "not an integer"})
    assert harness.background.work is None
    assert control.status("question-run").lifecycle == "paused"
    assert control.tabs()[0].operation_id == token
    with pytest.raises(RunMismatch):
        control.answer("stale", {"value": 8})
    control.answer("question-run", {"value": 8})
    harness.complete_segment()
    with pytest.raises(InvalidRunState):
        control.answer("question-run", {"value": 9})
    assert len(author.deliveries) == 1


@pytest.mark.parametrize("io_failure", [False, True])
def test_paused_ask_stop_drains_on_worker_before_terminal_owner_delivery(
    asking, tmp_path, io_failure
):
    control, harness, author, tab, token = asking
    assert control.pending_question("question-run") is not None
    if io_failure:
        journal = next(tmp_path.rglob("journal.jsonl"))
        contents = journal.read_bytes()
        journal.unlink()
        journal.mkdir()
        try:
            with pytest.raises(DIRECTORY_OPEN_ERROR):
                control.stop("question-run")
        finally:
            journal.rmdir()
            journal.write_bytes(contents)
    else:
        assert control.stop("question-run").lifecycle == "stopping"
    assert author.deliveries == [], "Stop is pure control, not author cleanup"
    assert control.tabs()[0].operation_id == token
    assert not control.close(tab)
    with ThreadPoolExecutor(max_workers=1) as worker:
        worker.submit(harness.background.execute).result(timeout=30)
    assert author.deliveries == [(None, author.deliveries[0][1])]
    assert author.deliveries[0][1] != get_ident()
    assert author.cancelled_followups == [None, None]
    assert control.tabs()[0].operation_id == token
    assert not control.close(tab)
    harness.owner.pump_all()
    harness.background.deliver()
    assert control.tabs()[0].operation_id is None
    assert control.status("question-run").lifecycle == (
        "failed" if io_failure else "stopped"
    )
    assert harness.handles.known_outcome(token).status == (
        "failed" if io_failure else "cancelled"
    )
