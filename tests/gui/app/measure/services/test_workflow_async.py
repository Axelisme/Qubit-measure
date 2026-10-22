"""Retained async workflow execution through the registered headless owner."""

import json
from threading import Thread

import pytest
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.experiment.workflows import QuestionMismatch
from zcu_tools.gui.app.measure.services.workflow_control import WorkflowControlFacet
from zcu_tools.gui.form import FormErrors
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.resources.entry import ResultEntry

from tests.gui._qt_waits import qt_until
from tests.gui.app.measure._workflow_ask import AskAuthor

from ._workflow_harness import StartHarness


@pytest.fixture
def retained_ask(qapp, tmp_path):
    harness = StartHarness()
    entry = ResultEntry.create(
        "synthetic", result_root=tmp_path / "result", database_root=tmp_path / "database"
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
    token = control.start(tab, "question-run")
    assert not isinstance(token, FormErrors)
    worker = Thread(target=harness.background.execute)
    worker.start()
    try:
        # The submitted author reaches Ask without any external release.
        qt_until(
            lambda: (status := control.status("question-run")) is not None
            and status.pending is not None,
            label="author pending Ask",
        )
        yield control, harness, author, tab, token, worker
    finally:
        control.stop("question-run")
        worker.join(30)  # Stop releases the retained Ask and its author cleanup.
        assert not worker.is_alive()
        harness.owner.pump_all()
        if harness.background.work is not None:
            harness.background.deliver()
        assert control.close(tab)


def test_pending_request_is_answerable_before_worker_return(retained_ask, tmp_path):
    control, harness, author, tab, token, worker = retained_ask
    question = control.pending_question("question-run")
    assert question is not None, "Ask must be answerable while execute retains its worker"
    assert question["request_id"]
    assert question["question_type"] == "Choose"
    assert question["question"] == {"question_type": "authored evidence", "point": 4}
    assert question["answer_schema"]["properties"]["value"]["type"] == "integer"
    assert control.tabs()[0].last_exchange.question == question
    with pytest.raises(QuestionMismatch) as stale:
        control.answer("question-run", {"value": 23}, request_id="old-question")
    assert stale.value.pending_request_id == question["request_id"]
    assert control.pending_question("question-run") == question
    assert author.deliveries == []
    assert worker.is_alive()
    assert control.tabs()[0].operation_id == token
    assert not control.close(tab)
    assert control.answer(
        "question-run", {"value": 23}, request_id=question["request_id"]
    ) == token
    worker.join(30)  # The accepted answer permits the submitted author to return.
    assert not worker.is_alive()
    assert control.tabs()[0].operation_id == token, "Only owner delivery releases RUN"
    assert control.tabs()[0].last_exchange.state == "answered"
    assert harness.handles.known_outcome(token) is None
    harness.owner.pump_all()
    harness.background.deliver()
    assert control.tabs()[0].operation_id is None
    assert harness.handles.known_outcome(token).status == "finished"
    assert control.records("question-run")[0].encoded_record.value == {"value": 23}
    with pytest.raises(QuestionMismatch) as terminal:
        control.answer("question-run", {"value": 24}, request_id=question["request_id"])
    assert terminal.value.pending_request_id is None
    journal = next(tmp_path.rglob("journal.jsonl"))
    assert [
        event["payload"]["request_id"]
        for event in map(json.loads, journal.read_text().splitlines())
        if event["kind"] == "question_answered"
    ] == [question["request_id"]]
