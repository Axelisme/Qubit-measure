"""Workflow RPC uses the shared headless facet, not the selected GUI pane."""

import base64

import pytest
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.experiment.workflows import (
    Live,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.app.measure.adapter import SessionEnv
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.services.workflow_attachment import WorkflowAttachment
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.form import FormModel
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.resources.entry import ResultEntry

from tests.gui._qt_waits import qt_until
from tests.gui.app.measure._workflow_ask import AskAuthor, Decision
from tests.mcp.measure._support import make_client

from ._form_rpc import FormRpc, ImmediateScheduler


class Values(FormModel):
    value: int = 3


class Native(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    value: int = 3


@workflow("synthetic", "Synthetic RPC forms, plots and records")
async def script(env: WorkflowEnv[None], plan: Native, tun: Live[Native]) -> None:
    env.plots.subplots("trend")

    def draw(figure: Figure) -> None:
        figure.axes[0].plot([0, 1], [plan.value, tun.current.value])

    env.plots.draw("trend", draw)
    await env.wait_until(env.started_at)
    await env.commit(Native(value=tun.current.value))


@pytest.fixture
def workflow_rpc(qapp, tmp_path):
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
    facet = controller.workflow_control()
    facet.configure(
        (
            WorkflowAttachment(
                script=script,
                plan_model=Values,
                knobs_model=Values,
                plan_to_cfg=lambda form, env, snapshot: Native(value=form.value),
                knobs_to_cfg=lambda form: Native(value=form.value),
                seed=lambda plan, form: None,
            ),
        )
    )
    remote = RemoteControlAdapter(
        controller, ControlOptions(port=0), ImmediateScheduler()
    )
    rpc = FormRpc(remote)
    try:
        yield controller, rpc
    finally:
        for tab in facet.tabs():
            if tab.busy and tab.run_id is not None:
                facet.stop(tab.run_id)
        controller.close()
        for tab in facet.tabs():
            facet.close(tab.tab_id)
        remote.on_client_close(rpc.link, on_owner_thread=True)
        qapp.processEvents()


def result(rpc, method, **params):
    reply = rpc.call(method, params)
    assert reply["ok"], reply
    return reply["result"]


def test_mcp_status_question_envelope_and_answer_share_native_operation(
    workflow_rpc, tmp_path
):
    controller, rpc = workflow_rpc
    author = AskAuthor()
    controller.workflow_control().configure((author.attachment(),))

    def respond(method, params):
        reply = rpc.call(method, params)
        if not reply["ok"]:
            from zcu_tools.gui.remote.errors import RemoteError

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
        started = client.call("workflow_start", {"tab_id": tab, "run_id": "ask-mcp"})
        # Pending Ask is read-only status, not a terminal owner notification.
        qt_until(
            lambda: (
                controller.workflow_control().pending_question("ask-mcp") is not None
            ),
            label="MCP pending Ask",
        )
        status = client.call("workflow_status", {"run_id": "ask-mcp"})
        question = status.get("pending_question")
        assert question is not None, "Status must expose the native question envelope"
        assert question["question_type"] == "Choose"
        assert question["question"] == {
            "question_type": "authored evidence",
            "point": 4,
        }
        assert question["answer_schema"]["properties"]["value"]["type"] == "integer"
        from zcu_tools.mcp.measure.session import GuiRpcError

        with pytest.raises(GuiRpcError) as invalid:
            client.call(
                "workflow_answer",
                {
                    "run_id": "ask-mcp",
                    "request_id": question["request_id"],
                    "answer": {"value": "bad"},
                },
            )
        assert invalid.value.code == "invalid_params"
        assert (
            client.call("workflow_status", {"run_id": "ask-mcp"})["pending_question"]
            == question
        )
        answered = client.call(
            "workflow_answer",
            {
                "run_id": "ask-mcp",
                "request_id": question["request_id"],
                "answer": {"value": 29},
            },
        )
        assert answered["handle"] == started["handle"]
        # The answered coroutine returns; terminal wait observes owner delivery.
        qt_until(
            lambda: not controller.workflow_control().tabs()[0].busy,
            label="MCP answer delivery",
        )
        assert (
            client.call("wait", {"op": started["handle"], "timeout": 0})["status"]
            == "finished"
        )
        assert author.deliveries[0][0] == Decision(value=29)
        assert (
            client.call("workflow_status", {"run_id": "ask-mcp"})["pending_question"]
            is None
        )
    finally:
        client.context.session.close()


def test_mcp_status_retains_accepted_exchange_after_answer(workflow_rpc, tmp_path):
    controller, rpc = workflow_rpc
    control = controller.workflow_control()
    control.configure((AskAuthor().attachment(),))
    client = make_client(tmp_path, lambda method, params: result(rpc, method, **params))
    try:
        tab = client.call("workflow_open", {"name": "asking"})["tab_id"]
        client.call("workflow_start", {"tab_id": tab, "run_id": "exchange"})
        # The retained invocation must publish its answerable Ask on-owner.
        qt_until(
            lambda: control.pending_question("exchange") is not None,
            label="exchange Ask",
        )
        question = client.call("workflow_status", {"run_id": "exchange"})[
            "pending_question"
        ]
        client.call(
            "workflow_answer",
            {
                "run_id": "exchange",
                "request_id": question["request_id"],
                "answer": {"value": 31},
            },
        )
        # The answered coroutine must reach terminal owner delivery.
        qt_until(lambda: not control.tabs()[0].busy, label="exchange answer delivery")
        status = client.call("workflow_status", {"run_id": "exchange"})
        assert status.get("last_exchange") == {
            "question": question,
            "answer": {"value": 31},
            "state": "answered",
        }
        assert status["pending_question"] is None
    finally:
        client.context.session.close()


def test_workflow_open_edit_preview_and_close_share_headless_drafts(workflow_rpc):
    controller, rpc = workflow_rpc
    first = result(rpc, "workflow.open", name="synthetic")["tab_id"]
    second = result(rpc, "workflow.open", name="synthetic")["tab_id"]
    edited = result(
        rpc,
        "workflow.edit_form",
        tab_id=first,
        form="plan",
        batch=[{"path": ["value"], "value": 7}],
    )
    assert edited["input"] == {"value": 7}
    preview = result(
        rpc,
        "workflow.describe_form",
        tab_id=first,
        form="plan",
        path=["value"],
        batch=[{"path": ["value"], "value": 11}],
    )
    assert preview["input"] == {"value": 11}
    assert preview["values"] == [{"path": ["value"], "value": 11}]
    assert preview["description"]["target"]["kind"] == "integer"
    assert result(rpc, "workflow.form", tab_id=first, form="plan")["input"] == {
        "value": 7
    }
    assert result(rpc, "workflow.form", tab_id=second, form="plan")["input"] == {
        "value": 3
    }
    assert controller.workflow_control().form_state(first, "plan").input.data == {
        "value": 7
    }
    assert len(result(rpc, "workflow.status")["tabs"]) == 2
    assert result(rpc, "workflow.close", tab_id=first) == {
        "tab_id": first,
        "closed": True,
    }
    assert result(rpc, "workflow.status")["tabs"][0]["tab_id"] == second


@pytest.mark.parametrize("explicit", [False, True])
def test_workflow_start_observes_draft_or_explicit_fresh_defaults_and_keeps_terminal_data(
    workflow_rpc, explicit
):
    controller, rpc = workflow_rpc
    tab = result(rpc, "workflow.open", name="synthetic")["tab_id"]
    result(
        rpc,
        "workflow.edit_form",
        tab_id=tab,
        form="tunables",
        batch=[{"path": ["value"], "value": 9}],
    )
    args = {"plan": [], "tunables": []} if explicit else {}
    receipt = result(rpc, "workflow.start", tab_id=tab, run_id="native", **args)
    assert receipt["run_id"] == "native"
    assert receipt["operation_id"] > 0
    assert not result(rpc, "workflow.close", tab_id=tab)["closed"]
    # Submitted worker and queued delivery must settle; the timer is only a watchdog.
    qt_until(
        lambda: not controller.workflow_control().tabs()[0].busy,
        label="workflow terminal delivery",
    )
    status = result(rpc, "workflow.status", run_id="native")
    assert status["tab_id"] == tab
    assert status["status"]["lifecycle"] == "done"
    records = result(rpc, "workflow.records", run_id="native")["records"]
    assert records[0]["encoded_record"]["value"] == {"value": 3 if explicit else 9}
    assert result(rpc, "workflow.records", run_id="native", since=1)["records"] == []
    assert result(rpc, "workflow.figures", run_id="native")["names"] == ["trend"]
    png = result(rpc, "workflow.figure", run_id="native", name="trend")
    assert base64.b64decode(png["png_b64"]).startswith(b"\x89PNG")


@pytest.mark.parametrize(
    "batch",
    [
        [{"path": ["value"], "value": 1}, {"path": ["value"], "value": 2}],
        [{"path": [], "value": 1}],
    ],
)
def test_workflow_rejects_malformed_batch_without_editing_draft(workflow_rpc, batch):
    _, rpc = workflow_rpc
    tab = result(rpc, "workflow.open", name="synthetic")["tab_id"]
    rejected = rpc.call(
        "workflow.edit_form", {"tab_id": tab, "form": "plan", "batch": batch}
    )
    assert not rejected["ok"]
    assert rejected["error"]["code"] == "invalid_params"
    assert result(rpc, "workflow.form", tab_id=tab, form="plan")["input"] == {
        "value": 3
    }


def test_workflow_invalid_path_and_stale_run_keep_native_errors(workflow_rpc):
    _, rpc = workflow_rpc
    tab = result(rpc, "workflow.open", name="synthetic")["tab_id"]
    bad_path = rpc.call(
        "workflow.describe_form", {"tab_id": tab, "form": "plan", "path": ["unknown"]}
    )
    assert bad_path["error"]["reason"] == "form_path"
    assert bad_path["error"]["data"]["form_path"] == ["unknown"]
    stale = rpc.call("workflow.stop", {"run_id": "stale"})
    assert stale["error"]["code"] == "precondition_failed"
    assert stale["error"]["reason"] == "run_mismatch"


def test_workflow_readonly_connection_can_preview_but_cannot_edit(workflow_rpc):
    controller, rpc = workflow_rpc
    tab = result(rpc, "workflow.open", name="synthetic")["tab_id"]
    result(rpc, "control.release")
    rpc.remote.on_client_close(rpc.link, on_owner_thread=True)
    rpc.link = ClientLink("observer", token_required=False)
    rpc.remote.on_client_open(rpc.link)
    result(rpc, "control.connect", label="observer", mode="readonly")
    assert result(
        rpc,
        "workflow.form",
        tab_id=tab,
        form="plan",
        batch=[{"path": ["value"], "value": 5}],
    )["input"] == {"value": 5}
    rejected = rpc.call(
        "workflow.edit_form", {"tab_id": tab, "form": "plan", "batch": []}
    )
    assert rejected["error"]["reason"] == "not_holder"
    assert controller.workflow_control().form_state(tab, "plan").input.data == {
        "value": 3
    }
