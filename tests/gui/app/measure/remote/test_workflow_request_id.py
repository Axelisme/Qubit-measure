"""MCP answers retain the exact question identity through the live RPC owner."""

import json

import pytest
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.gui.app.measure.adapter import SessionEnv
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.remote.errors import RemoteError
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.resources.entry import ResultEntry

from tests.gui._qt_waits import qt_until
from tests.gui.app.measure._workflow_ask import AskAuthor
from tests.mcp.measure._support import make_client

from ._form_rpc import FormRpc, ImmediateScheduler

RUN_ID = "21ed5156dc9b441a8b5cda3cf26633f7"


@pytest.fixture
def answering_client(qapp, tmp_path):
    controller = Controller(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        Registry(),
        None,
        BaseEventBus(),
        project_root=str(tmp_path),
    )
    entry = ResultEntry.create(
        "synthetic",
        result_root=tmp_path / "result",
        database_root=tmp_path / "database",
    )
    entry.new_point("source", description="synthetic")
    controller.session_services.context.install_entry(entry)
    controller.session_services.context.use_point(PointUseRequest("source"))
    board = white_noise()
    controller.session_services.soc_connection.install_soc(board.soc, board.soccfg)
    control = controller.workflow_control()
    control.configure((AskAuthor().attachment(),))
    remote = RemoteControlAdapter(
        controller, ControlOptions(port=0), ImmediateScheduler()
    )
    rpc = FormRpc(remote)

    def respond(method, params):
        reply = rpc.call(method, params)
        if not reply["ok"]:
            error = reply["error"]
            raise RemoteError(
                error["code"],
                error["message"],
                reason=error.get("reason"),
                data=error.get("data"),
            )
        return reply["result"]

    client = make_client(tmp_path, respond)
    try:
        tab = client.call("workflow_open", {"name": "asking"})["tab_id"]
        receipt = client.call("workflow_start", {"tab_id": tab, "run_id": RUN_ID})
        # Start submits the author, which reaches its Ask without another command.
        qt_until(
            lambda: control.pending_question(RUN_ID) is not None,
            label="MCP pending request",
        )
        yield client, receipt, control
    finally:
        controller.close()
        client.context.session.close()
        for tab in controller.workflow_control().tabs():
            controller.workflow_control().close(tab.tab_id)
        remote.on_client_close(rpc.link, on_owner_thread=True)
        qapp.processEvents()


def test_missing_stale_and_replayed_ids_preserve_current_question(
    answering_client, tmp_path
):
    client, receipt, control = answering_client
    before = client.call("workflow_status", {"run_id": RUN_ID})
    pending = before["pending_question"]
    request = pending["request_id"]
    assert before["status"]["pending_request_id"] == request
    with pytest.raises(GuiRpcError) as missing:
        client.call("workflow_answer", {"run_id": RUN_ID, "answer": {"value": 4}})
    assert missing.value.code == "invalid_params"
    with pytest.raises(GuiRpcError) as stale:
        client.call(
            "workflow_answer",
            {"run_id": RUN_ID, "request_id": "old", "answer": {"value": "bad"}},
        )
    assert stale.value.reason == "question_mismatch", (
        "ID matching precedes answer schema validation"
    )
    assert stale.value.data == {"pending_request_id": request}
    assert client.call("workflow_status", {"run_id": RUN_ID}) == before
    with pytest.raises(GuiRpcError) as invalid:
        client.call(
            "workflow_answer",
            {"run_id": RUN_ID, "request_id": request, "answer": {"value": "bad"}},
        )
    assert invalid.value.reason == "answer_validation"
    accepted = client.call(
        "workflow_answer",
        {"run_id": RUN_ID, "request_id": request, "answer": {"value": 4}},
    )
    assert accepted["handle"] == receipt["handle"]
    # The accepted answer lets the author return; owner delivery releases busy.
    qt_until(
        lambda: not control.tabs()[0].busy,
        label="MCP answer terminal delivery",
    )
    records = client.call("workflow_records", {"run_id": RUN_ID})["records"]
    assert records[0].get("operations") == [], (
        "Wire records retain the native operation facts"
    )
    assert records[0]["directory"] == "."
    assert records[0]["revisions"] == [0]
    with pytest.raises(GuiRpcError) as replay:
        client.call(
            "workflow_answer",
            {"run_id": RUN_ID, "request_id": request, "answer": {"value": 5}},
        )
    assert replay.value.reason == "question_mismatch"
    assert replay.value.data == {"pending_request_id": None}
    journal = next(tmp_path.rglob("journal.jsonl"))
    assert [
        event["request_id"]
        for event in map(json.loads, journal.read_text().splitlines())
        if event["kind"] == "question_answered"
    ] == [request]
