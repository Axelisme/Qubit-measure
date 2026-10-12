"""Window contracts through public native/GUI seams."""

import pytest
from pydantic import JsonValue
from qtpy.QtWidgets import QApplication, QLabel, QLineEdit, QPushButton
from zcu_tools.datafile import RunSnapshot
from zcu_tools.experiment.workflows import (
    Done,
    InitEnv,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.app.workflows.remote import FormWrite
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.form import FormEnv, FormErrors
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.operation_handles import OperationOutcome
from zcu_tools.resources.entry import (
    ResultEntry,
)

from tests.gui.app.workflows._start_fakes import StartHarness
from tests.gui.app.workflows._synthetic import PlanForm

from ._reload_source import InjectedCatalog
from ._sample_input import sample_input
from ._synthetic import Context, Plan, Tunables, inline_input
from ._window_support import RpcRequest, start_plan_batch


def test_window_services_run_and_progress_share_session(
    run_attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    qapp: QApplication,
    catalog_source: InjectedCatalog,
) -> None:
    _existing_window, harness, _entry = run_attachment
    closed: list[str] = []
    window = WorkflowWindow(
        harness.state,
        harness.services,
        harness.owner,
        registrations=catalog_source.catalog.attachments,
        loader=catalog_source.loader,
        parent=None,
        on_close=lambda: closed.append("closed"),
    )
    window.plan_panel.state.reset(inline_input())

    def init(_env: InitEnv[Context], _plan: Plan) -> None:
        return None

    @workflow("session-progress", plan=Plan, tunables=Tunables, init=init)
    def step(
        env: WorkflowEnv[Context],
        plan: Plan,
        tun: Tunables,
        state: None,
    ) -> Step[None, int]:
        del plan, tun, state
        bar = env.pbar("session work", 4)
        bar.set_progress(2)
        bar.refresh()
        window.pause("session-services")
        yield from env.wait_until(env.started_at)
        return Done()

    catalog_source.install(step, window)
    try:
        token = window.start("session-services")
        assert isinstance(token, int)
        assert harness.handles.live_count() == 1
        harness.complete_segment()
        status = window.controller.status()
        assert status is not None and status.lifecycle == "paused"
        assert window.controller.operation_id == token
        assert harness.handles.known_outcome(token) is None
        bars = harness.services.progress.bars_for_operation(token)
        assert [(bar.label, bar.total, bar.n) for _, bar in bars] == [
            ("session work", 4, 2)
        ]
        assert harness.services.progress.bars_for_owner("workflow") == bars
        stop_button = window.findChild(QPushButton, "workflow.run.stop")
        assert stop_button is not None
        stop_button.click()
        status = window.controller.status()
        assert status is not None and status.lifecycle == "stopped"
        assert harness.handles.known_outcome(token) == OperationOutcome("cancelled")
        assert harness.services.progress.bars_for_operation(token) == ()
        assert not harness.gate.snapshot()
        assert window.close()
        assert closed == ["closed"]
    finally:
        if harness.background.work is not None:
            run_id = window.controller.run_id
            assert run_id is not None
            window.stop(run_id)
            harness.complete_segment()
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_invalid_plan_does_not_submit_or_create_workflow_directories(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
) -> None:
    window, harness, entry = attachment
    window.plan_panel.state.write({("scale",): 0.0}, FormEnv())
    result = window.start("invalid")
    assert isinstance(result, FormErrors) and ("scale",) in result.messages
    assert window.controller.binding is None and harness.background.work is None
    assert not (entry.result_path / "workflows").exists()
    assert not (entry.database_path / "workflows").exists()


def test_startup_source_failure_is_not_replaced_by_old_constructor_capture(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
) -> None:
    window, harness, entry = attachment
    harness.services.context.clear_context()
    with pytest.raises(FailedPreconditionError):
        window.start("empty")
    assert window.controller.binding is None and harness.background.work is None
    assert not (entry.result_path / "workflows").exists()


def test_start_captures_current_values_then_keeps_them_fixed_until_execute(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    catalog_source: InjectedCatalog,
) -> None:
    window, harness, entry = attachment
    context = harness.services.context
    context.set_parameter(context.read_context().target, "general.ext.value", 31.0)
    window.plan_panel.state.write({("scale",): {"expr": "0.06 * 2"}}, FormEnv())
    observed: list[tuple[Context, RunSnapshot, float]] = []

    def init(_env: InitEnv[Context], _plan: Plan) -> None:
        return None

    @workflow("capture-observer", plan=Plan, tunables=Tunables, init=init)
    def observe(
        env: WorkflowEnv[Context],
        plan: Plan,
        tun: Tunables,
        state: None,
    ) -> Step[None, int]:
        del state
        observed.append((env.context, plan.snapshot, tun.sample.value))
        yield from ()
        return Done()

    catalog_source.install(observe, window)

    def edit_on_progress_notification() -> None:
        window.tunables_panel.state.write({("sample", "value"): 77.0}, FormEnv())

    detach = harness.progress.attach_by_owner("workflow", edit_on_progress_notification)
    try:
        token = window.start("captured")
    finally:
        detach()
    assert sample_input(window.tunables_panel.state.input.data)["value"] == {
        "expr": "general.ext.value"
    }
    assert isinstance(token, int)
    controller = window.controller
    assert controller is not None and controller.status() is None
    assert not (entry.result_path / "workflows/captured").exists()
    context.set_parameter(context.read_context().target, "general.ext.value", 99.0)
    entry.new_point("later", description="later description")
    context.use_point(PointUseRequest("later"))
    harness.background.execute()
    assert len(observed) == 1
    seed, snapshot, guess = observed[0]
    assert seed.scale == 0.12 and guess == 31.0
    assert str(snapshot.entry_id) == entry.entry_id
    assert snapshot.entry_name == "entry" and snapshot.point == "source"
    assert snapshot.description == "captured description" and snapshot.roles == {
        "qubit": "Q1"
    }
    status = controller.status()
    assert status is not None and status.lifecycle == "done"
    assert harness.gate.snapshot()
    harness.background.deliver()
    assert not harness.gate.snapshot()
    assert (entry.result_path / "workflows/captured").is_dir()
    assert (entry.result_path / "workflows/captured/manifest.json").is_file()
    assert window.tuning_panel is not None
    assert sample_input(window.tuning_panel.host.accepted_input.data)["value"] == {
        "expr": "general.ext.value"
    }


def test_submit_failure_leaves_no_accepted_run_or_paths(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
) -> None:
    window, harness, entry = attachment
    harness.background.fail_submit = True
    with pytest.raises(OSError, match="submit failed"):
        window.start("submit-error")
    assert window.controller.binding is None and harness.handles.live_count() == 0
    assert not harness.gate.snapshot()
    assert not (entry.result_path / "workflows").exists()
    assert not (entry.database_path / "workflows").exists()


def test_rpc_start_partial_input_uses_current_defaults_and_one_fixed_capture(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    rpc: RpcRequest,
    catalog_source: InjectedCatalog,
) -> None:
    window, harness, entry = attachment
    context = harness.services.context
    context.set_parameter(context.read_context().target, "general.ext.value", 31.0)
    window.tunables_panel.state.write({("sample", "value"): 999.0}, FormEnv())
    observed: list[tuple[Context, Plan, Tunables]] = []

    def init(_env: InitEnv[Context], _plan: Plan) -> None:
        return None

    @workflow("rpc-start-observer", plan=Plan, tunables=Tunables, init=init)
    def observe(
        env: WorkflowEnv[Context],
        plan: Plan,
        tun: Tunables,
        state: None,
    ) -> Step[None, int]:
        del state
        observed.append((env.context, plan, tun))
        yield from ()
        return Done()

    catalog_source.install(observe, window)
    reply = rpc(
        "workflow.start",
        {
            "run_id": "partial-rpc",
            "plan": start_plan_batch(),
            "tunables": [{"path": ["sample", "count"], "value": 9}],
        },
    )
    assert reply["ok"] is True
    controller = window.controller
    assert controller is not None and isinstance(controller.operation_id, int)
    assert reply["result"] == {
        "run_id": "partial-rpc",
        "operation_id": controller.operation_id,
        "status": None,
    }
    assert controller.status() is None and not observed
    assert not (entry.result_path / "workflows/partial-rpc").exists()
    for panel in (window.plan_panel, window.tunables_panel):
        assert panel.state.base.data == panel.state.input.data
        assert not panel.isEnabled()
    assert window.plan_panel.state.input.data["scale"] == {"expr": "0.06 * 2"}
    assert sample_input(window.tunables_panel.state.input.data)["value"] == {
        "expr": "general.ext.value"
    }
    shown = window.tunables_panel.editor.findChild(QLabel, "form.value.sample.value")
    assert isinstance(shown, QLabel) and float(shown.text()) == 31.0
    context.set_parameter(context.read_context().target, "general.ext.value", 88.0)
    assert float(shown.text()) == 31.0
    harness.background.execute()
    assert len(observed) == 1
    seed, plan, tun = observed[0]
    assert seed.scale == 0.12 and tun.sample.value == 31.0 and tun.sample.count == 9
    assert plan.points == PlanForm.default_input(FormEnv()).data["points"]
    assert plan.samples == (1.0, 1.1) and plan.snapshot.point == "source"
    assert harness.gate.snapshot()
    harness.background.deliver()
    assert not harness.gate.snapshot()
    assert window.tuning_panel is not None
    assert (
        window.tuning_panel.host.accepted_input.data
        == window.tunables_panel.state.base.data
    )


@pytest.mark.parametrize(
    "form,bad_value",
    [("plan", 0.0), ("tunables", 0), ("plan", {"expr": "missing_name"})],
)
def test_rpc_start_model_errors_leave_both_panel_drafts_and_bases_untouched(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    rpc: RpcRequest,
    form: str,
    bad_value: JsonValue,
) -> None:
    window, harness, entry = attachment
    before = [
        (p.state.input.data, p.state.base.data)
        for p in (window.plan_panel, window.tunables_panel)
    ]
    plan = start_plan_batch()
    tunables: list[FormWrite] = []
    path = ["scale"] if form == "plan" else ["sample", "count"]
    if form == "plan":
        for edit in plan:
            if edit["path"] == path:
                edit["value"] = bad_value
    else:
        tunables = [{"path": path, "value": bad_value}]
    reply = rpc(
        "workflow.start", {"run_id": "invalid-rpc", "plan": plan, "tunables": tunables}
    )
    assert reply["ok"] is False
    error = reply["error"]
    assert isinstance(error, dict) and error["code"] == "invalid_params"
    assert error["reason"] == "form_errors"
    envelope = error["data"]
    assert isinstance(envelope, dict)
    data = envelope["messages"]
    assert isinstance(data, list) and any(
        item["path"] == path and isinstance(item["message"], str) for item in data
    )
    assert before == [
        (p.state.input.data, p.state.base.data)
        for p in (window.plan_panel, window.tunables_panel)
    ]
    assert window.controller.binding is None and harness.background.work is None
    assert not (entry.result_path / "workflows").exists()
    assert not (entry.database_path / "workflows").exists()


@pytest.mark.parametrize(
    "batch",
    [
        {},
        [42],
        [{"path": ["points"]}],
        [{"path": [], "value": 17}],
        [{"path": ["samples", 0], "value": 1.0}],
        [{"path": ["points"], "value": 17}, {"path": ["points"], "value": 19}],
        [{"path": ["unknown"], "value": 17}],
        [{"path": ["samples"], "value": {"kind": "list", "values": [0.0]}}],
        [
            {"path": ["samples", "kind"], "value": "list"},
            {"path": ["samples", "values", "0"], "value": 1.0},
        ],
        [
            {"path": ["samples"], "value": {}},
            {"path": ["samples", "kind"], "value": "list"},
        ],
    ],
)
def test_rpc_start_rejects_bad_batches_without_mutating_drafts(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    rpc: RpcRequest,
    batch: object,
) -> None:
    window, harness, entry = attachment
    before = [
        (p.state.input.data, p.state.base.data)
        for p in (window.plan_panel, window.tunables_panel)
    ]
    reply = rpc(
        "workflow.start", {"run_id": "bad-batch", "plan": batch, "tunables": []}
    )
    assert reply["ok"] is False
    error = reply["error"]
    assert isinstance(error, dict) and error["code"] == "invalid_params"
    assert before == [
        (p.state.input.data, p.state.base.data)
        for p in (window.plan_panel, window.tunables_panel)
    ]
    assert window.controller.binding is None and harness.background.work is None
    assert not (entry.result_path / "workflows").exists()


def test_rpc_start_ignores_unparseable_gui_text_then_resets_accepted_base(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    rpc: RpcRequest,
) -> None:
    window, _harness, _entry = attachment
    widget = window.plan_panel.editor.findChild(QLineEdit, "form.input.scale")
    assert isinstance(widget, QLineEdit)
    widget.setText("-")
    before = window.plan_panel.state.input.data
    local = window.plan_panel.editor.local_parse_errors()
    assert local is not None and ("scale",) in local.messages
    gui = window.start("bad-gui")
    assert isinstance(gui, FormErrors) and ("scale",) in gui.messages
    assert window.plan_panel.state.input.data == before
    reply = rpc(
        "workflow.start",
        {"run_id": "clean-rpc", "plan": start_plan_batch(), "tunables": []},
    )
    assert reply["ok"] is True
    assert window.plan_panel.editor.local_parse_errors() is None
    assert window.plan_panel.state.base.data == window.plan_panel.state.input.data
    assert window.plan_panel.state.input.data["scale"] == {"expr": "0.06 * 2"}


def test_window_start_explicit_empty_batch_uses_defaults_not_the_gui_draft(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
) -> None:
    window, _harness, _entry = attachment
    window.tunables_panel.state.write({("sample", "value"): -1.0}, FormEnv())
    gui = window.start("invalid-tunables")
    assert isinstance(gui, FormErrors) and ("sample", "value") in gui.messages
    original_plan = window.plan_panel.state.input.data
    token = window.start("default-tunables", tunables_batch={})
    assert isinstance(token, int)
    assert window.plan_panel.state.base.data == original_plan
    assert sample_input(window.tunables_panel.state.base.data)["value"] == {
        "expr": "general.ext.value"
    }


@pytest.mark.parametrize("method", ["gui", "rpc"])
def test_start_submit_failure_preserves_error_and_allows_window_close(
    run_attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    rpc: RpcRequest,
    method: str,
) -> None:
    window, harness, entry = run_attachment
    controller = window.controller
    harness.background.fail_submit = True
    if method == "gui":
        with pytest.raises(OSError, match="submit failed"):
            window.start("submit-error")
    else:
        reply = rpc(
            "workflow.start",
            {
                "run_id": "submit-error",
                "plan": start_plan_batch(),
                "tunables": [],
            },
        )
        assert reply["ok"] is False
        error = reply["error"]
        assert isinstance(error, dict) and error["code"] == "controller_error"
        assert "submit failed" in error["message"]
    assert window.controller is controller
    assert harness.handles.live_count() == 0
    assert not harness.gate.snapshot()
    assert not (entry.result_path / "workflows").exists()
    assert not (entry.database_path / "workflows").exists()
    assert window.close()


def test_rpc_start_native_submit_failure_does_not_install_new_panel_bases(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    rpc: RpcRequest,
) -> None:
    window, harness, entry = attachment
    before = [
        (p.state.input.data, p.state.base.data)
        for p in (window.plan_panel, window.tunables_panel)
    ]
    harness.background.fail_submit = True
    reply = rpc(
        "workflow.start",
        {"run_id": "failed-rpc", "plan": start_plan_batch(), "tunables": []},
    )
    assert reply["ok"] is False
    error = reply["error"]
    assert isinstance(error, dict) and error["code"] == "controller_error"
    assert "submit failed" in error["message"]
    assert window.controller.binding is None and harness.handles.live_count() == 0
    assert not harness.gate.snapshot()
    assert before == [
        (p.state.input.data, p.state.base.data)
        for p in (window.plan_panel, window.tunables_panel)
    ]
    assert not (entry.result_path / "workflows").exists()


def test_rpc_start_source_failure_does_not_fall_back_to_old_gui_capture(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    rpc: RpcRequest,
) -> None:
    window, harness, entry = attachment
    before = [
        (p.state.input.data, p.state.base.data)
        for p in (window.plan_panel, window.tunables_panel)
    ]
    harness.services.context.clear_context()
    reply = rpc(
        "workflow.start",
        {"run_id": "empty-source", "plan": start_plan_batch(), "tunables": []},
    )
    assert reply["ok"] is False
    error = reply["error"]
    assert isinstance(error, dict) and error["code"] == "precondition_failed"
    assert before == [
        (p.state.input.data, p.state.base.data)
        for p in (window.plan_panel, window.tunables_panel)
    ]
    assert window.controller.binding is None and harness.background.work is None
    assert not (entry.result_path / "workflows").exists()


def test_rpc_start_cannot_replace_pending_native_run(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    rpc: RpcRequest,
) -> None:
    window, harness, _entry = attachment
    token = window.start("attached")
    assert isinstance(token, int)
    controller = window.controller
    before = [
        (p.state.input.data, p.state.base.data)
        for p in (window.plan_panel, window.tunables_panel)
    ]
    reply = rpc(
        "workflow.start",
        {"run_id": "replacement", "plan": start_plan_batch(), "tunables": []},
    )
    assert reply["ok"] is False
    error = reply["error"]
    assert isinstance(error, dict) and error["code"] == "precondition_failed"
    assert error["reason"] == "invalid_run_state"
    assert error["data"] == {"action": "start", "lifecycle": "running"}
    assert window.controller is controller
    assert controller is not None and controller.run_id == "attached"
    assert controller.operation_id == token and controller.status() is None
    assert harness.gate.snapshot()
    assert before == [
        (p.state.input.data, p.state.base.data)
        for p in (window.plan_panel, window.tunables_panel)
    ]
