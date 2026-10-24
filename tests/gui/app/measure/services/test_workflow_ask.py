"""Native Ask through the registered headless facet and retained operation owner."""

import json
from contextlib import contextmanager
from threading import Event, get_ident

import pytest
from pydantic import ValidationError
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.experiment.workflows import (
    Actor,
    QuestionMismatch,
    RunMismatch,
)
from zcu_tools.gui.app.measure.services.workflow_control import WorkflowControlFacet
from zcu_tools.gui.form import FormErrors
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.ports import OperationConflictError
from zcu_tools.resources.entry import ResultEntry

from tests._filesystem import DIRECTORY_OPEN_ERROR
from tests.gui.app.measure._workflow_ask import AskAuthor, Decision, OtherAnswer

from ._workflow_harness import StartHarness

RUN_ID = "21ed5156dc9b441a8b5cda3cf26633f7"
OTHER_RUN_ID = "ffffffffffffffffffffffffffffffff"


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
    control = WorkflowControlFacet(harness.state, harness.services)
    control.configure((author.attachment(),))
    tab = control.open("asking")
    token = control.start(tab, RUN_ID)
    assert not isinstance(token, FormErrors)
    harness.background.start()
    try:
        harness.wait_for(
            lambda: control.pending_question(RUN_ID) is not None,
            label="native Ask publication",
        )
        yield control, harness, author, tab, token
    finally:
        control.stop(RUN_ID)
        if harness.background.worker is not None:
            harness.finish_worker()
        elif harness.background.work is not None:
            harness.complete_segment()
        control.close(tab)


def run_lease_presence(harness: StartHarness) -> tuple[tuple[str, str, str], ...]:
    # Elapsed seconds are presentation, not lease identity. Pair these public
    # presence facts with the retained operation token and admission assertions.
    return tuple(
        (presence.kind, presence.origin_kind, presence.note)
        for presence in harness.gate.snapshot()
        if presence.kind == "run"
    )


def test_answer_keeps_invocation_operation_and_lease_until_owner_delivery(asking):
    control, harness, author, tab, token = asking
    lease = run_lease_presence(harness)
    assert lease
    question = control.pending_question(RUN_ID)
    assert question is not None, "Facet must publish the native Ask envelope"
    assert question["question_type"] == "Choose"
    assert question["question"] == {"question_type": "authored evidence", "point": 4}
    assert question["answer_schema"]["properties"]["value"]["type"] == "integer"
    request_id = question["request_id"]
    control.apply_tunables(RUN_ID, {("value",): 9}, actor=Actor("user", "test"))
    assert control.answer(RUN_ID, {"value": 23}, request_id=request_id) == token
    with pytest.raises(QuestionMismatch):
        control.answer(RUN_ID, {"value": 24}, request_id=request_id)
    other = control.open("asking")
    try:
        harness.wait_for(
            harness.background.completed.is_set, label="answered worker return"
        )
        assert author.captures == [3]
        assert author.deliveries == [(Decision(value=23), author.deliveries[0][1])]
        assert author.deliveries[0][1] != get_ident(), (
            "Answer must not execute author code inline"
        )
        assert control.tabs()[0].operation_id == token
        assert run_lease_presence(harness) == lease
        assert not control.close(tab)
        with pytest.raises(OperationConflictError):
            control.start(other, OTHER_RUN_ID)
        harness.finish_worker()
        assert control.tabs()[0].operation_id is None
        assert run_lease_presence(harness) == ()
        assert control.records(RUN_ID)[0].revision == 0
        assert control.records(RUN_ID)[0].encoded_record.value == {"value": 23}
        assert harness.handles.known_outcome(token).status == "finished"
        replacement = control.start(other, OTHER_RUN_ID)
        assert isinstance(replacement, int)
        harness.background.start()
        harness.wait_for(
            lambda: control.pending_question(OTHER_RUN_ID) is not None,
            label="replacement Ask publication",
        )
        control.stop(OTHER_RUN_ID)
        harness.finish_worker()
    finally:
        control.close(other)


def test_wrong_typed_answer_delivery_stays_paused_and_can_retry(asking, tmp_path):
    control, harness, author, tab, token = asking
    lease = run_lease_presence(harness)
    assert lease
    question = control.pending_question(RUN_ID)
    assert question is not None
    request_id = question["request_id"]
    with pytest.raises(
        TypeError, match="Answer must match the question's declared model"
    ):
        control.answer(RUN_ID, OtherAnswer(value=7), request_id=request_id)
    assert control.status(RUN_ID).lifecycle == "paused"
    assert control.tabs()[0].operation_id == token
    assert control.pending_question(RUN_ID) == question
    assert run_lease_presence(harness) == lease
    assert not control.close(tab)
    assert harness.handles.known_outcome(token) is None
    journal = next(tmp_path.rglob("journal.jsonl"))
    assert "resumed" not in [
        json.loads(line)["kind"] for line in journal.read_text().splitlines()
    ]
    assert author.deliveries == []
    assert control.answer(RUN_ID, Decision(value=8), request_id=request_id) == token
    harness.finish_worker()
    assert control.status(RUN_ID).lifecycle == "done"
    assert harness.handles.known_outcome(token).status == "finished"
    assert run_lease_presence(harness) == ()


def test_invalid_json_answer_is_rejected_without_consuming_question(asking):
    control, harness, author, _tab, token = asking
    question = control.pending_question(RUN_ID)
    assert question is not None
    request_id = question["request_id"]
    with pytest.raises(ValidationError, match="Decision"):
        control.answer(RUN_ID, {"value": "not an integer"}, request_id=request_id)
    assert not harness.background.completed.is_set()
    assert control.pending_question(RUN_ID) == question
    assert control.status(RUN_ID).lifecycle == "paused"
    assert control.tabs()[0].operation_id == token
    with pytest.raises(RunMismatch):
        control.answer("stale", {"value": 8}, request_id=request_id)
    control.answer(RUN_ID, {"value": 8}, request_id=request_id)
    harness.finish_worker()
    with pytest.raises(QuestionMismatch):
        control.answer(RUN_ID, {"value": 9}, request_id=request_id)
    assert len(author.deliveries) == 1


@contextmanager
def failed_journal_write(tmp_path):
    """Make the next journal append fail at the filesystem boundary, then restore it."""
    journal = next(tmp_path.rglob("journal.jsonl"))
    contents = journal.read_bytes()
    journal.unlink()
    journal.mkdir()
    try:
        yield
    finally:
        journal.rmdir()
        journal.write_bytes(contents)


@pytest.mark.parametrize("io_failure", [False, True])
def test_paused_ask_stop_drains_on_worker_before_terminal_owner_delivery(
    asking, tmp_path, io_failure
):
    control, harness, author, tab, token = asking
    lease = run_lease_presence(harness)
    assert lease
    assert control.pending_question(RUN_ID) is not None
    cleanup_entered, release_cleanup = Event(), Event()

    def hold_cleanup():
        cleanup_entered.set()
        # The caller releases this finite author cleanup in finally below.
        assert release_cleanup.wait(30)

    author.on_finish = hold_cleanup
    try:
        if io_failure:
            with failed_journal_write(tmp_path), pytest.raises(DIRECTORY_OPEN_ERROR):
                control.stop(RUN_ID)
        else:
            assert control.stop(RUN_ID).lifecycle == "stopping"
        # Stop must reach the retained coroutine's finally without owner delivery.
        assert cleanup_entered.wait(5), (
            "Stop must drain author cleanup even after journal failure"
        )
        assert author.finally_threads[0] != get_ident()
        assert control.tabs()[0].operation_id == token
        assert run_lease_presence(harness) == lease
        assert not control.close(tab)
        assert harness.handles.known_outcome(token) is None
    finally:
        release_cleanup.set()
        # A broken baseline may reject Stop before releasing the Ask. Restore the
        # filesystem first, then retry only for bounded fixture cleanup.
        if io_failure and not cleanup_entered.is_set():
            control.stop(RUN_ID)
    harness.wait_for(harness.background.completed.is_set, label="Stop worker return")
    assert control.tabs()[0].operation_id == token
    assert run_lease_presence(harness) == lease
    assert not control.close(tab)
    harness.finish_worker()
    assert control.tabs()[0].operation_id is None
    assert run_lease_presence(harness) == ()
    assert control.status(RUN_ID).lifecycle == ("failed" if io_failure else "cancelled")
    assert harness.handles.known_outcome(token).status == (
        "failed" if io_failure else "cancelled"
    )
