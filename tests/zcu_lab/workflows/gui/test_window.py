"""Window contracts through public native/GUI seams."""

from pathlib import Path

import pytest
from pydantic import JsonValue
from qtpy.QtWidgets import QApplication, QLabel, QLineEdit, QPushButton
from zcu_tools.datafile import RunSnapshot
from zcu_tools.experiment.native_persistence import load_run_with_metadata, save_run
from zcu_tools.experiment.workflows import (
    Done,
    InitEnv,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.form import FormEnv, FormErrors, FormInput
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.operation_handles import OperationOutcome
from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.resources.entry import (
    ResultEntry,
)

from tests.gui.app.workflows._start_fakes import StartHarness
from tests.zcu_lab.workflows._sim_inputs import Inputs
from zcu_lab.v2.twotone.time_domain.t1.core import T1Exp
from zcu_lab.workflows.calibration import Calibrated
from zcu_lab.workflows.gui import window as window_module
from zcu_lab.workflows.gui.plan import T1PlanForm
from zcu_lab.workflows.gui.remote import FormWrite
from zcu_lab.workflows.gui.tunables import T1TunablesForm
from zcu_lab.workflows.gui.window import T1WorkflowWindow
from zcu_lab.workflows.t1 import T1Plan, T1Tunables

from ._plan_inputs import inline_input
from ._window_support import RpcRequest, start_plan_batch


def test_window_services_run_and_progress_share_session(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    inputs: Inputs,
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _existing_window, harness, _entry = run_attachment
    closed: list[str] = []
    window = T1WorkflowWindow(
        harness.state,
        harness.services,
        harness.owner,
        parent=None,
        on_close=lambda: closed.append("closed"),
    )
    window.plan_panel.state.reset(inline_input(inputs))

    def init(_env: InitEnv[Calibrated], _plan: T1Plan) -> None:
        return None

    @workflow("session-progress", plan=T1Plan, tunables=T1Tunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated], plan: T1Plan, tun: T1Tunables, state: None
    ) -> Step[None, int]:
        del plan, tun, state
        bar = env.pbar("session work", 4)
        bar.set_progress(2)
        bar.refresh()
        window.pause("session-services")
        yield from env.wait_until(env.started_at)
        return Done()

    monkeypatch.setattr(window_module, "t1_fluxdep", step)
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
        assert harness.services.progress.bars_for_owner("t1-workflow") == bars
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


def test_invalid_pi_does_not_submit_or_create_workflow_directories(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
) -> None:
    window, harness, entry = attachment
    window.plan_panel.state.write({("pi_length_us",): 0.0}, FormEnv())
    result = window.start("invalid")
    assert isinstance(result, FormErrors) and ("pi_length_us",) in result.messages
    assert window.controller.binding is None and harness.background.work is None
    assert not (entry.result_path / "workflows").exists()
    assert not (entry.database_path / "workflows").exists()


def test_startup_source_failure_is_not_replaced_by_old_constructor_capture(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
) -> None:
    window, harness, entry = attachment
    harness.services.context.clear_context()
    with pytest.raises(FailedPreconditionError):
        window.start("empty")
    assert window.controller.binding is None and harness.background.work is None
    assert not (entry.result_path / "workflows").exists()


def test_start_captures_current_values_then_keeps_them_fixed_until_execute(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, harness, entry = attachment
    context = harness.services.context
    context.set_parameter(context.read_context().target, "general.ext.guess_us", 31.0)
    window.plan_panel.state.write({("pi_length_us",): {"expr": "0.06 * 2"}}, FormEnv())
    observed: list[tuple[Calibrated, RunSnapshot, float]] = []

    def init(_env: InitEnv[Calibrated], _plan: T1Plan) -> None:
        return None

    @workflow("capture-observer", plan=T1Plan, tunables=T1Tunables, init=init)
    def observe(
        env: WorkflowEnv[Calibrated], plan: T1Plan, tun: T1Tunables, state: None
    ) -> Step[None, int]:
        del state
        observed.append((env.context, plan.snapshot, tun.guess_us))
        yield from ()
        return Done()

    monkeypatch.setattr(window_module, "t1_fluxdep", observe)

    def edit_on_progress_notification() -> None:
        window.tunables_panel.state.write({("guess_us",): 77.0}, FormEnv())

    detach = harness.progress.attach_by_owner(
        "t1-workflow", edit_on_progress_notification
    )
    try:
        token = window.start("captured")
    finally:
        detach()
    assert window.tunables_panel.state.input.data["guess_us"] == {
        "expr": "general.ext.guess_us"
    }
    assert isinstance(token, int)
    controller = window.controller
    assert controller is not None and controller.status() is None
    assert not (entry.result_path / "workflows/captured").exists()
    context.set_parameter(context.read_context().target, "general.ext.guess_us", 99.0)
    entry.new_point("later", description="later description")
    context.use_point(PointUseRequest("later"))
    harness.background.execute()
    assert len(observed) == 1
    seed, snapshot, guess = observed[0]
    assert seed.pi_length_us == 0.12 and guess == 31.0
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
    assert window.tuning_panel.host.accepted_input.data["guess_us"] == {
        "expr": "general.ext.guess_us"
    }


def test_submit_failure_leaves_no_accepted_run_or_paths(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
) -> None:
    window, harness, entry = attachment
    harness.background.fail_submit = True
    with pytest.raises(OSError, match="submit failed"):
        window.start("submit-error")
    assert window.controller.binding is None and harness.handles.live_count() == 0
    assert not harness.gate.snapshot()
    assert not (entry.result_path / "workflows").exists()
    assert not (entry.database_path / "workflows").exists()


def test_real_t1_gui_caller_runs_native_offline_acquisitions(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
) -> None:
    window, harness, entry = attachment
    window.plan_panel.state.write(
        {("fluxes", "kind"): "list", ("fluxes", "values"): [1.0]}, FormEnv()
    )
    window.tunables_panel.state.reset(
        FormInput(
            T1TunablesForm,
            {
                "cal": {"span_mhz": 2.0, "rabi_start_factor": 0.125},
                "rounds": 2,
            },
        )
    )
    token = window.start("native")
    assert isinstance(token, int)
    harness.background.execute()
    harness.background.deliver()
    controller = window.controller
    assert controller is not None
    status = controller.status()
    assert status is not None and status.lifecycle == "done"
    assert status.committed_seq == 1
    engine = controller.engine
    assert engine is not None
    record = engine.records(status.run_id)[0]
    assert len(record.run_files) >= 4
    for relative in record.run_files:
        assert (
            entry.database_path / "workflows/native" / record.iteration_dir / relative
        ).is_file()
    assert window.tuning_panel is not None and not window.tuning_panel.isEnabled()
    assert not harness.gate.snapshot()


@pytest.mark.parametrize("starter", ["gui", "mcp"])
@pytest.mark.parametrize("device", ["flux", "offline_device"])
def test_default_device_alias_start_and_native_save_use_the_run_capture(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    inputs: Inputs,
    mcp_tools: ToolTable,
    starter: str,
    device: str,
    tmp_path: Path,
) -> None:
    window, harness, entry = run_attachment
    context = harness.services.context
    context.set_parameter(context.read_context().target, "Q1.flux_source", device)
    context.set_alias_rows(
        (AliasRow("qubit", "Q1"), AliasRow("flux_dev", "$qubit.flux_source"))
    )
    window.prepare_forms()
    plan = [e for e in start_plan_batch(inputs) if e["path"] != ["flux_dev"]]
    for edit in plan:
        if edit["path"] == ["fluxes", "values"]:
            edit["value"] = [inputs.flux_value]
    tunables: list[FormWrite] = [
        {"path": ["cal", "span_mhz"], "value": 2.0},
        {"path": ["cal", "rabi_start_factor"], "value": 0.125},
        {"path": ["rounds"], "value": 2},
    ]
    run_id = "alias-" + starter
    if starter == "gui":
        window.plan_panel.state.write(
            {tuple(e["path"]): e["value"] for e in plan}, window.plan_panel.env
        )
        window.tunables_panel.state.write(
            {tuple(e["path"]): e["value"] for e in tunables}, FormEnv()
        )
        token = window.start(run_id)
    else:
        receipt = None
        start_error = None
        try:
            receipt = mcp_tools["workflows_workflow_start"]["handler"](
                {"run_id": run_id, "plan": plan, "tunables": tunables}
            )
        except RuntimeError as exc:
            start_error = exc
        assert start_error is None, start_error
        assert isinstance(receipt, dict)
        token = receipt["operation_id"]
    # No membership check at Start, even for an unknown device name.
    assert isinstance(token, int), token
    assert window.plan_panel.state.input.data["flux_dev"] == {"expr": "$flux_dev"}
    context.set_parameter(
        context.read_context().target, "Q1.flux_source", "later_device"
    )
    context.set_alias_rows(
        (AliasRow("qubit", "Q1"), AliasRow("flux_dev", "later_device"))
    )
    harness.complete_segment()
    status = window.controller.status()
    assert status is not None
    engine = window.controller.engine
    assert engine is not None
    if device == "offline_device":
        assert status.lifecycle == "failed" and device in str(status.reason)
        assert engine.records(run_id) == ()
        return
    assert status.lifecycle == "done" and status.committed_seq == 1, status
    record = engine.records(run_id)[0]
    source = (
        entry.database_path
        / "workflows"
        / run_id
        / record.iteration_dir
        / record.run_files[-1]
    )
    spec = T1Exp().persistence_spec()
    native, metadata = load_run_with_metadata(source, spec=spec)
    assert native.cfg is not None
    assert metadata.snapshot.roles == {"qubit": "Q1", "flux_dev": "flux"}
    # Save again after the live alias edit; neither form input nor history refreshes.
    destination = tmp_path / "saved-again.h5"
    save_run(native, destination, spec=spec, metadata=metadata)
    _, saved = load_run_with_metadata(destination, spec=spec)
    assert saved.snapshot == metadata.snapshot
    assert saved.snapshot.roles["flux_dev"] == "flux"
    assert window.plan_panel.state.input.data["flux_dev"] == {"expr": "$flux_dev"}


def test_rpc_start_partial_input_uses_current_defaults_and_one_fixed_capture(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    inputs: Inputs,
    rpc: RpcRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, harness, entry = attachment
    context = harness.services.context
    context.set_parameter(context.read_context().target, "general.ext.guess_us", 31.0)
    window.tunables_panel.state.write({("guess_us",): 999.0}, FormEnv())
    observed: list[tuple[Calibrated, T1Plan, T1Tunables]] = []

    def init(_env: InitEnv[Calibrated], _plan: T1Plan) -> None:
        return None

    @workflow("rpc-start-observer", plan=T1Plan, tunables=T1Tunables, init=init)
    def observe(
        env: WorkflowEnv[Calibrated], plan: T1Plan, tun: T1Tunables, state: None
    ) -> Step[None, int]:
        del state
        observed.append((env.context, plan, tun))
        yield from ()
        return Done()

    monkeypatch.setattr(window_module, "t1_fluxdep", observe)
    reply = rpc(
        "workflow.start",
        {
            "run_id": "partial-rpc",
            "plan": start_plan_batch(inputs),
            "tunables": [{"path": ["reps"], "value": 9}],
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
    assert window.plan_panel.state.input.data["pi_length_us"] == {"expr": "0.06 * 2"}
    assert window.tunables_panel.state.input.data["guess_us"] == {
        "expr": "general.ext.guess_us"
    }
    shown = window.tunables_panel.editor.findChild(QLabel, "form.value.guess_us")
    assert isinstance(shown, QLabel) and float(shown.text()) == 31.0
    context.set_parameter(context.read_context().target, "general.ext.guess_us", 88.0)
    assert float(shown.text()) == 31.0
    harness.background.execute()
    assert len(observed) == 1
    seed, plan, tun = observed[0]
    assert seed.pi_length_us == 0.12 and tun.guess_us == 31.0 and tun.reps == 9
    assert plan.points == T1PlanForm.default_input(FormEnv()).data["points"]
    assert plan.fluxes == (1.0, 1.1) and plan.snapshot.point == "source"
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
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    inputs: Inputs,
    rpc: RpcRequest,
    form: str,
    bad_value: JsonValue,
) -> None:
    window, harness, entry = attachment
    before = [
        (p.state.input.data, p.state.base.data)
        for p in (window.plan_panel, window.tunables_panel)
    ]
    plan = start_plan_batch(inputs)
    tunables: list[FormWrite] = []
    path = ["pi_length_us"] if form == "plan" else ["reps"]
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
        [{"path": ["fluxes", 0], "value": 1.0}],
        [{"path": ["points"], "value": 17}, {"path": ["points"], "value": 19}],
        [{"path": ["unknown"], "value": 17}],
        [{"path": ["fluxes"], "value": {"kind": "list", "values": [0.0]}}],
        [
            {"path": ["fluxes", "kind"], "value": "list"},
            {"path": ["fluxes", "values", "0"], "value": 1.0},
        ],
        [
            {"path": ["fluxes"], "value": {}},
            {"path": ["fluxes", "kind"], "value": "list"},
        ],
    ],
)
def test_rpc_start_rejects_bad_batches_without_mutating_drafts(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
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
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    inputs: Inputs,
    rpc: RpcRequest,
) -> None:
    window, _harness, _entry = attachment
    widget = window.plan_panel.editor.findChild(QLineEdit, "form.input.pi_length_us")
    assert isinstance(widget, QLineEdit)
    widget.setText("-")
    before = window.plan_panel.state.input.data
    local = window.plan_panel.editor.local_parse_errors()
    assert local is not None and ("pi_length_us",) in local.messages
    gui = window.start("bad-gui")
    assert isinstance(gui, FormErrors) and ("pi_length_us",) in gui.messages
    assert window.plan_panel.state.input.data == before
    reply = rpc(
        "workflow.start",
        {"run_id": "clean-rpc", "plan": start_plan_batch(inputs), "tunables": []},
    )
    assert reply["ok"] is True
    assert window.plan_panel.editor.local_parse_errors() is None
    assert window.plan_panel.state.base.data == window.plan_panel.state.input.data
    assert window.plan_panel.state.input.data["pi_length_us"] == {"expr": "0.06 * 2"}


def test_window_start_explicit_empty_batch_uses_defaults_not_the_gui_draft(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
) -> None:
    window, _harness, _entry = attachment
    window.tunables_panel.state.write({("guess_us",): -1.0}, FormEnv())
    gui = window.start("invalid-tunables")
    assert isinstance(gui, FormErrors) and ("guess_us",) in gui.messages
    original_plan = window.plan_panel.state.input.data
    token = window.start("default-tunables", tunables_batch={})
    assert isinstance(token, int)
    assert window.plan_panel.state.base.data == original_plan
    assert window.tunables_panel.state.base.data["guess_us"] == {
        "expr": "general.ext.guess_us"
    }


@pytest.mark.parametrize("method", ["gui", "rpc"])
def test_start_submit_failure_preserves_error_and_allows_window_close(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    inputs: Inputs,
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
                "plan": start_plan_batch(inputs),
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
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    inputs: Inputs,
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
        {"run_id": "failed-rpc", "plan": start_plan_batch(inputs), "tunables": []},
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
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    inputs: Inputs,
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
        {"run_id": "empty-source", "plan": start_plan_batch(inputs), "tunables": []},
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
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    inputs: Inputs,
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
        {"run_id": "replacement", "plan": start_plan_batch(inputs), "tunables": []},
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
