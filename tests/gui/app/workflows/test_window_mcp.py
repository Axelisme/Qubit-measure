"""Window mcp contracts through public native/GUI seams."""

import json
from dataclasses import asdict

import pytest
from zcu_tools.experiment.workflows import (
    Done,
    InitEnv,
    Next,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.resources.entry import (
    ResultEntry,
)

from tests.gui.app.workflows._start_fakes import StartHarness

from ._recording_transport import WindowTransport
from ._reload_source import InjectedCatalog
from ._sample_input import sample_input
from ._synthetic import Context, Plan, Tunables
from ._window_support import start_plan_batch


@pytest.fixture
def mcp_paused_controls(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    mcp_tools: ToolTable,
    catalog_source: InjectedCatalog,
) -> tuple[list[tuple[int, float, float, str | None]], list[dict[str, object]]]:
    """Pause via MCP after one commit; return attempts and later Stop receipts.

    Each attempt holds state, value, captured scale and source point.
    Declared steps expose control timing; they are not acquisition evidence.
    """
    window, harness, _entry = attachment
    attempts: list[tuple[int, float, float, str | None]] = []
    pause_receipts: list[dict[str, object]] = []
    stop_receipts: list[dict[str, object]] = []

    def init(_env: InitEnv[Context], _plan: Plan) -> int:
        return 0

    @workflow("mcp-controls", plan=Plan, tunables=Tunables, init=init)
    def step(
        env: WorkflowEnv[Context],
        plan: Plan,
        tun: Tunables,
        state: int,
    ) -> Step[int, float]:
        attempts.append(
            (state, tun.sample.value, env.context.scale, plan.snapshot.point)
        )
        if state == 1 and len(attempts) == 2:
            pause_receipts.append(
                mcp_tools["workflows_workflow_pause"]["handler"](
                    {"run_id": "mcp-controls"}
                )
            )
            yield from env.wait_until(env.started_at)
        if state == 2:
            stop_receipts.append(
                mcp_tools["workflows_workflow_stop"]["handler"](
                    {"run_id": "mcp-controls"}
                )
            )
            yield from env.wait_until(env.started_at)
            return Done()
        return Next(tun.sample.value, state + 1)

    catalog_source.install(step, window)
    pending = mcp_tools["workflows_workflow_start"]["handler"](
        {
            "run_id": "mcp-controls",
            "plan": start_plan_batch(),
            "tunables": [],
        }
    )
    token = pending["operation_id"]
    assert isinstance(token, int)
    harness.complete_segment()
    assert harness.background.error is None
    controller = window.controller
    assert controller is not None and controller.engine is not None
    assert pause_receipts[0]["operation_id"] == token
    paused = mcp_tools["workflows_workflow_status"]["handler"]({})
    assert paused["status"]["lifecycle"] == "paused" and paused["operation_id"] == token
    return attempts, stop_receipts


def test_mcp_native_start_reads_committed_records_and_png(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    mcp_tools: ToolTable,
    mcp_transport: WindowTransport,
) -> None:
    """Read a synthetic native commit and retained Plots through real MCP tools."""
    window, harness, entry = attachment
    start = mcp_tools["workflows_workflow_start"]["handler"]
    plan = start_plan_batch()
    for edit in plan:
        if edit["path"] == ["samples", "values"]:
            edit["value"] = [1.0]
    before = [
        (panel.state.input.data, panel.state.base.data)
        for panel in (window.plan_panel, window.tunables_panel)
    ]
    with pytest.raises(RuntimeError, match="invalid_params.*form_errors"):
        start(
            {
                "run_id": "mcp-native",
                "plan": plan,
                "tunables": [
                    {"path": ["sample", "count"], "value": 0},
                ],
            }
        )
    assert window.controller is not None and window.controller.binding is None
    assert harness.background.work is None and not harness.gate.snapshot()
    assert before == [
        (panel.state.input.data, panel.state.base.data)
        for panel in (window.plan_panel, window.tunables_panel)
    ]
    assert not (entry.result_path / "workflows").exists()
    assert len(mcp_transport.requests) == 1

    pending = start(
        {
            "run_id": "mcp-native",
            "plan": plan,
            "tunables": [
                {"path": ["sample", "count"], "value": 2},
            ],
        }
    )
    token = window.controller.operation_id
    assert isinstance(token, int)
    assert pending == {"run_id": "mcp-native", "operation_id": token, "status": None}
    assert window.plan_panel.state.input.data["scale"] == {"expr": "0.06 * 2"}
    harness.complete_segment()
    assert harness.background.error is None
    status = window.controller.status()
    engine = window.controller.engine
    assert status is not None and engine is not None
    assert status.lifecycle == "done" and status.committed_seq == 1
    assert mcp_tools["workflows_workflow_status"]["handler"]({}) == {
        "run_id": "mcp-native",
        "operation_id": None,
        "status": json.loads(json.dumps(asdict(status))),
    }
    record = engine.records("mcp-native")[0]
    expected = {
        "commit_seq": record.commit_seq,
        "call_seq": record.call_seq,
        "revision": record.revision,
        "encoded_record": asdict(record.encoded_record),
        "iteration_dir": record.iteration_dir,
    }
    assert mcp_tools["workflows_workflow_records"]["handler"](
        {"run_id": "mcp-native"}
    ) == {
        "run_id": "mcp-native",
        "records": [expected],
    }
    figures = mcp_tools["workflows_workflow_figures"]["handler"](
        {"run_id": "mcp-native"}
    )
    names = window.figure_names("mcp-native")
    assert names and figures == {"run_id": "mcp-native", "names": list(names)}
    image = mcp_tools["workflows_workflow_figure"]["handler"](
        {"run_id": "mcp-native", "name": names[0]}
    )
    assert isinstance(image, ToolReply) and not image.is_error
    assert image.data == {"run_id": "mcp-native", "name": names[0]}
    assert len(image.images) == 1
    assert image.images[0].data.startswith(b"\x89PNG\r\n\x1a\n")
    assert image.images[0].data == window.figure_png("mcp-native", names[0])
    assert [request.method for request in mcp_transport.requests] == [
        "workflow.start",
        "workflow.start",
        "workflow.status",
        "workflow.records",
        "workflow.figures",
        "workflow.figure",
    ]
    assert len({request.id for request in mcp_transport.requests}) == 6
    assert not harness.gate.snapshot()


def test_mcp_native_controls_keep_capture_lease_and_record_revisions(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    mcp_tools: ToolTable,
    mcp_transport: WindowTransport,
    mcp_paused_controls: tuple[
        list[tuple[int, float, float, str | None]], list[dict[str, object]]
    ],
) -> None:
    """Read and edit the same paused owner, then resume and stop via MCP."""
    window, harness, entry = attachment
    attempts, stop_receipts = mcp_paused_controls
    controller = window.controller
    assert controller is not None and controller.engine is not None
    engine, token = controller.engine, controller.operation_id
    assert isinstance(token, int)
    lease = tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot())
    assert lease and harness.handles.live_count() == 1
    first = engine.records("mcp-controls")[0]
    assert first.revision == 0 and first.encoded_record.value == 20.0
    context = harness.services.context
    context.set_parameter(context.read_context().target, "general.ext.value", 99.0)
    entry.new_point("later-mcp")
    context.use_point(PointUseRequest("later-mcp"))
    context.set_alias_rows((AliasRow("qubit", ""),))
    inspected = mcp_tools["workflows_workflow_form"]["handler"](
        {"form": "tunables", "run_id": "mcp-controls"}
    )
    assert inspected["revision"] == 0
    assert sample_input(inspected["input"])["value"] == {"expr": "general.ext.value"}
    assert {"path": ["sample", "value"], "value": 20.0} in inspected["values"]
    applied = mcp_tools["workflows_workflow_tunables"]["handler"](
        {
            "run_id": "mcp-controls",
            "batch": [
                {
                    "path": ["sample", "value"],
                    "value": {"expr": "general.ext.value * 2"},
                },
            ],
        }
    )
    assert applied["status"]["revision"] == 1
    assert engine.records("mcp-controls") == (first,)
    panel = window.tuning_panel
    assert panel is not None
    assert sample_input(panel.host.accepted_input.data)["value"] == {
        "expr": "general.ext.value * 2"
    }
    observed = mcp_tools["workflows_workflow_form"]["handler"](
        {"form": "tunables", "run_id": "mcp-controls"}
    )
    assert observed["revision"] == 1
    assert {"path": ["sample", "value"], "value": 40.0} in observed["values"]
    resumed = mcp_tools["workflows_workflow_resume"]["handler"](
        {"run_id": "mcp-controls"}
    )
    assert resumed["operation_id"] == token and controller.operation_id == token
    assert (
        tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot()) == lease
    )
    assert window.controller is controller and controller.engine is engine
    harness.complete_segment()
    assert harness.background.error is None
    terminal = mcp_tools["workflows_workflow_status"]["handler"]({})
    assert (
        terminal["status"]["lifecycle"] == "stopped"
        and terminal["operation_id"] is None
    )
    assert stop_receipts[0]["operation_id"] == token
    assert attempts == [
        (0, 20.0, 0.12, "source"),
        (1, 20.0, 0.12, "source"),
        (1, 40.0, 0.12, "source"),
        (2, 40.0, 0.12, "source"),
    ]
    records = engine.records("mcp-controls")
    assert records[0] == first and len(records) == 2
    assert records[1].revision == 1 and records[1].encoded_record.value == 40.0
    delta = mcp_tools["workflows_workflow_records"]["handler"](
        {"run_id": "mcp-controls", "since": 1}
    )
    assert len(delta["records"]) == 1 and delta["records"][0]["revision"] == 1
    assert delta["records"][0]["call_seq"] == records[1].call_seq
    assert [request.method for request in mcp_transport.requests] == [
        "workflow.start",
        "workflow.pause",
        "workflow.status",
        "workflow.form",
        "workflow.tunables",
        "workflow.form",
        "workflow.resume",
        "workflow.stop",
        "workflow.status",
        "workflow.records",
    ]
    assert (
        not harness.gate.snapshot() and harness.handles.known_outcome(token) is not None
    )


@pytest.mark.usefixtures("mcp_paused_controls")
def test_mcp_native_stale_edit_preserves_current_owner_without_retry(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    mcp_tools: ToolTable,
    mcp_transport: WindowTransport,
) -> None:
    window, harness, _entry = attachment
    controller, panel = window.controller, window.tuning_panel
    assert (
        controller is not None and controller.engine is not None and panel is not None
    )
    status, records = controller.status(), controller.engine.records("mcp-controls")
    accepted = panel.host.accepted_input.data
    draft = panel.panel.state.input.data
    # Presence age advances on reads; compare the run lease's stable fields.
    lease = tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot())
    count = len(mcp_transport.requests)
    with pytest.raises(RuntimeError, match="precondition_failed.*run_mismatch"):
        mcp_tools["workflows_workflow_tunables"]["handler"](
            {
                "run_id": "old-mcp",
                "batch": [{"path": ["sample", "value"], "value": 500.0}],
            }
        )
    assert len(mcp_transport.requests) == count + 1
    assert window.controller is controller and controller.status() == status
    assert controller.engine.records("mcp-controls") == records
    assert (
        panel.host.accepted_input.data == accepted
        and panel.panel.state.input.data == draft
    )
    assert (
        tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot()) == lease
    )
    mcp_tools["workflows_workflow_stop"]["handler"]({"run_id": "mcp-controls"})
    assert not harness.gate.snapshot()


@pytest.mark.parametrize("method", ["pause", "resume", "records", "form", "tunables"])
def test_mcp_native_pending_error_is_not_retried_or_claimed_as_acceptance(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    mcp_tools: ToolTable,
    mcp_transport: WindowTransport,
    method: str,
) -> None:
    window, harness, entry = attachment
    pending = mcp_tools["workflows_workflow_start"]["handler"](
        {
            "run_id": "mcp-pending",
            "plan": start_plan_batch(),
            "tunables": [],
        }
    )
    assert pending["status"] is None and isinstance(pending["operation_id"], int)
    params: dict[str, object] = {"run_id": "mcp-pending"}
    if method == "form":
        params["form"] = "tunables"
    if method == "tunables":
        params["batch"] = []
    with pytest.raises(RuntimeError, match=r"GUI Error \(controller_error\)"):
        mcp_tools["workflows_workflow_" + method]["handler"](params)
    assert [request.method for request in mcp_transport.requests] == [
        "workflow.start",
        "workflow." + method,
    ]
    assert window.controller is not None and window.controller.status() is None
    assert window.controller.operation_id == pending["operation_id"]
    assert harness.gate.snapshot() and not (entry.result_path / "workflows").exists()
    assert (
        mcp_tools["workflows_workflow_stop"]["handler"]({"run_id": "mcp-pending"})
        == pending
    )
    harness.complete_segment()
    assert (
        not harness.gate.snapshot() and not (entry.result_path / "workflows").exists()
    )
