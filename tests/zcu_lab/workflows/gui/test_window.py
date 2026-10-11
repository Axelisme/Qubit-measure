"""Concrete Qt startup uses one fixed native capture and no hardware."""

import json
import sys
from base64 import b64decode
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from threading import get_ident
from time import monotonic
from types import TracebackType

import pytest
from matplotlib.figure import Figure
from pydantic import JsonValue
from qtpy.QtWidgets import QApplication, QLabel, QLineEdit, QPushButton, QStackedWidget
from zcu_tools.datafile import RunSnapshot
from zcu_tools.device import DeviceInfo
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.experiment.native_persistence import load_run_with_metadata, save_run
from zcu_tools.experiment.workflows import (
    Actor,
    Done,
    InitEnv,
    InvalidRunState,
    Next,
    RunMismatch,
    Step,
    TunablesSnapshot,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.form import FormEnv, FormErrors, FormInput, FormWritePath
from zcu_tools.gui.remote.control_service import ControlOptions
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.gui.remote.wire import Request, parse_request
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.events import GateChangedPayload
from zcu_tools.gui.session.operation_handles import OperationOutcome
from zcu_tools.gui.session.services.device import ConnectDeviceRequest
from zcu_tools.mcp.core.bridge import DeliverFn, McpBridge, MCPBridgeConfig, OnClosedFn
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.resources.entry import (
    ResultEntry,
    RoleSpec,
    component_registry,
    role_registry,
)

from tests.gui.app.workflows._start_fakes import StartHarness
from tests.resources.entry.fakes import registry_state, restore_registry
from tests.zcu_lab.workflows._sim_inputs import Inputs
from zcu_lab.components import QubitSchema
from zcu_lab.v2.twotone.time_domain.t1.core import T1Exp
from zcu_lab.workflows.calibration import Calibrated
from zcu_lab.workflows.gui import window as window_module
from zcu_lab.workflows.gui.plan import T1PlanForm
from zcu_lab.workflows.gui.remote import FormWrite
from zcu_lab.workflows.gui.tunables import T1TunablesForm
from zcu_lab.workflows.gui.window import T1WorkflowWindow
from zcu_lab.workflows.mcp import build_workflow_tools
from zcu_lab.workflows.t1 import T1Plan, T1Tunables

from ._plan_inputs import inline_input, library_env


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """Keep the Qt application alive with its process-global plotting host."""
    instance = QApplication.instance()
    return instance if isinstance(instance, QApplication) else QApplication([])


@pytest.fixture
def attachment(
    qapp: QApplication,
    tmp_path: Path,
    inputs: Inputs,
) -> Iterator[tuple[T1WorkflowWindow, StartHarness, ResultEntry]]:
    before = registry_state()
    kind = "qubit/workflow-window-test"
    component_registry.register(kind, QubitSchema)
    try:
        role_registry.get("qubit")
    except ValueError:
        role_registry.register("qubit", RoleSpec("qubit/*"))
    installed = registry_state()
    harness = StartHarness()
    window = None
    try:
        entry = ResultEntry.create(
            "entry",
            result_root=tmp_path / "result",
            database_root=tmp_path / "Database",
        )
        entry.setup.add_component("Q1", kind=kind)
        entry.new_point("source", description="captured description")
        context = harness.services.context
        context.install_entry(entry)
        context.use_point(PointUseRequest("source"))
        context.set_alias_rows((AliasRow("qubit", "Q1"),))
        context.set_parameter(
            context.read_context().target, "general.ext.guess_us", 20.0
        )
        harness.state.set_context(
            replace(
                harness.state.session_env,
                soc=inputs.soc,
                soccfg=inputs.soccfg,
            )
        )
        harness.services.device.start_connect_device(
            ConnectDeviceRequest("FakeDevice", "flux", "none")
        )
        harness.background.execute()
        harness.background.deliver()
        window = T1WorkflowWindow(
            harness.state,
            harness.services,
            harness.owner,
        )
        window.plan_panel.state.reset(inline_input(inputs))
        yield window, harness, entry
    finally:
        if window is not None:
            if harness.background.work is not None:
                controller = window.controller
                assert controller is not None and controller.operation_id is not None
                harness.handles.cancel(controller.operation_id)
                harness.background.execute()
                harness.background.deliver()
            window.close()
            window.deleteLater()
            qapp.processEvents()
        try:
            assert registry_state() == installed, (
                "workflow window test polluted registries"
            )
        finally:
            restore_registry(before)


@pytest.fixture
def run_attachment(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[T1WorkflowWindow, StartHarness, ResultEntry]:
    """Supply typed cached native form info for Window/Controller ownership.

    The substitute does not replace Runner, Controller or native Engine.
    """
    _window, harness, _entry = attachment
    native: dict[str, DeviceInfo] = {"flux": FakeDeviceInfo(address="none", value=0.0)}
    monkeypatch.setattr(
        harness.services.device, "capture_native_device_info", lambda: native
    )
    return attachment


@pytest.fixture
def escaped_qt_errors(monkeypatch: pytest.MonkeyPatch) -> list[BaseException]:
    """Observe Qt's public exception hook without aborting the test process."""
    errors: list[BaseException] = []

    def record(
        _kind: type[BaseException],
        error: BaseException,
        _traceback: TracebackType | None,
    ) -> None:
        errors.append(error)

    monkeypatch.setattr(sys, "excepthook", record)
    return errors


@pytest.fixture
def settlements(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
) -> list[int]:
    """Record real public settlement calls without replacing their behavior."""
    _window, harness, _entry = run_attachment
    tokens: list[int] = []
    settle = harness.handles.settle

    def record(token: int, outcome: OperationOutcome) -> None:
        tokens.append(token)
        settle(token, outcome)

    monkeypatch.setattr(harness.handles, "settle", record)
    return tokens


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


def test_close_pending_start_requests_stop_without_destroying_producer_host(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
) -> None:
    window, harness, entry = run_attachment
    token = window.start("close-pending")
    assert isinstance(token, int)
    assert not window.close()
    assert harness.gate.snapshot()
    harness.background.execute()
    harness.background.deliver()
    assert not harness.gate.snapshot()
    assert window.close()
    assert not (entry.result_path / "workflows").exists()


@pytest.mark.parametrize("action", ["stop", "close"])
@pytest.mark.parametrize("previous_start", [False, True])
@pytest.mark.parametrize("notification", ["progress", "gate"])
def test_admission_notification_can_cancel_current_start(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    previous_start: bool,
    notification: str,
) -> None:
    window, harness, entry = run_attachment
    fixed = window.controller
    calls: list[str] = []

    def init(_env: InitEnv[Calibrated], _plan: T1Plan) -> None:
        calls.append("init")

    @workflow("admission-observer", plan=T1Plan, tunables=T1Tunables, init=init)
    def observe(
        env: WorkflowEnv[Calibrated], plan: T1Plan, tun: T1Tunables, state: None
    ) -> Step[None, int]:
        del env, plan, tun, state
        calls.append("step")
        yield from ()
        return Done()

    monkeypatch.setattr(window_module, "t1_fluxdep", observe)
    if previous_start:
        assert isinstance(window.start("previous"), int)
        harness.complete_segment()
        assert not harness.gate.snapshot()
        calls.clear()
    old = fixed.binding
    if old is not None:
        old.state.write({("guess_us",): 87.0}, old.env)
    old_input = None if old is None else old.state.input.data
    close_results: list[bool] = []
    observed: list[tuple[str | None, int | None, bool, bool]] = []
    errors: list[str] = []

    def cancel_on_notification() -> None:
        binding = window.controller.binding
        observed.append(
            (
                window.controller.run_id,
                window.controller.operation_id,
                window.controller is fixed,
                binding is not None and binding is not old and binding.engine is None,
            )
        )
        try:
            window.apply_tunables(
                "notification", {("guess_us",): 55.0}, actor=Actor("agent", "rpc")
            )
        except RuntimeError as error:
            errors.append(str(error))
        if action == "close":
            close_results.append(window.close())
        else:
            window.stop("notification")

    def cancel_on_register(payload: GateChangedPayload) -> None:
        if payload.active:
            cancel_on_notification()

    detach: Callable[[], None]
    if notification == "progress":
        detach = harness.progress.attach_by_owner("t1-workflow", cancel_on_notification)
    else:
        subscription = harness.bus.subscribe(GateChangedPayload, cancel_on_register)
        detach = subscription.unsubscribe
    try:
        token = window.start("notification")
    finally:
        detach()
    assert isinstance(token, int)
    assert observed == [("notification", token, True, True)]
    assert len(errors) == 1 and "accepted run" in errors[0]
    if old is not None:
        assert old.state.input.data == old_input
    pending = fixed.binding
    assert (
        pending is not None and pending.state.input.data == pending.accepted_input.data
    )
    if action == "close":
        assert close_results == [False]
    controller = window.controller
    assert controller is fixed and controller.operation_id == token
    assert controller.status() is None and harness.gate.snapshot()
    harness.background.execute()
    assert calls == []
    assert controller.status() is None and harness.gate.snapshot()
    harness.background.deliver()
    assert controller.operation_id is None and harness.handles.live_count() == 0
    assert not harness.gate.snapshot()
    assert not (entry.result_path / "workflows/notification").exists()
    assert not (entry.database_path / "workflows/notification").exists()
    assert window.close()


def test_replacement_accepted_progress_uses_current_host_and_form_capture(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, harness, _entry = run_attachment
    attempts: list[float] = []

    def init(_env: InitEnv[Calibrated], _plan: T1Plan) -> int:
        return 0

    @workflow("accepted-replacement", plan=T1Plan, tunables=T1Tunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated], plan: T1Plan, tun: T1Tunables, state: int
    ) -> Step[int, float]:
        del plan
        attempts.append(tun.guess_us)
        if state == 0:
            env.pbar("work", 2).refresh()
            harness.owner.pump_all()
            yield from ()
            return Next(tun.guess_us, 1)
        return Done()

    monkeypatch.setattr(window_module, "t1_fluxdep", step)
    assert isinstance(window.start("old"), int)
    harness.complete_segment()
    old = window.controller.binding
    assert old is not None
    old.state.write({("guess_us",): 87.0}, old.env)
    old_input, old_base = old.state.input.data, old.state.base.data
    context = harness.services.context
    context.set_parameter(context.read_context().target, "general.ext.guess_us", 41.0)
    observed: list[bool] = []
    replies: list[TunablesSnapshot | FormErrors] = []

    def apply_on_accepted_progress() -> None:
        if observed:
            return
        status = window.controller.status()
        if status is None or status.run_id != "next" or status.lifecycle != "running":
            return
        replies.append(
            window.apply_tunables(
                "next",
                {("guess_us",): {"expr": "general.ext.guess_us + 2"}},
                actor=Actor("agent", "rpc"),
            )
        )
        panel = window.tuning_panel
        observed.append(
            panel is not None
            and panel.host is window.controller.binding
            and panel.host is not old
            and panel.host.env is not old.env
        )

    detach = harness.progress.attach_by_owner("t1-workflow", apply_on_accepted_progress)
    try:
        assert isinstance(window.start("next"), int)
        harness.complete_segment()
    finally:
        detach()
    assert observed == [True] and len(replies) == 1
    reply = replies[0]
    assert isinstance(reply, TunablesSnapshot) and reply.revision == 1
    assert isinstance(reply.values, dict) and reply.values["guess_us"] == 43.0
    assert attempts == [20.0, 20.0, 41.0, 43.0]
    assert old.state.input.data == old_input and old.state.base.data == old_base
    status = window.controller.status()
    assert status is not None and status.lifecycle == "done"
    assert not harness.gate.snapshot() and harness.handles.live_count() == 0


@pytest.mark.parametrize("previous_start", [False, True])
def test_gate_registration_reentrant_start_preserves_outer_controller(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
    previous_start: bool,
) -> None:
    window, harness, entry = run_attachment
    calls: list[str] = []

    def init(_env: InitEnv[Calibrated], _plan: T1Plan) -> None:
        calls.append("init")

    @workflow("reentry-observer", plan=T1Plan, tunables=T1Tunables, init=init)
    def observe(
        env: WorkflowEnv[Calibrated], plan: T1Plan, tun: T1Tunables, state: None
    ) -> Step[None, int]:
        del env, plan, tun, state
        calls.append("step")
        yield from ()
        return Done()

    monkeypatch.setattr(window_module, "t1_fluxdep", observe)
    if previous_start:
        assert isinstance(window.start("previous"), int)
        window.stop("previous")
        harness.background.execute()
        harness.background.deliver()
        assert not harness.gate.snapshot()

    observed: list[tuple[str | None, int | None, bool]] = []
    rejections: list[InvalidRunState] = []
    preserved: list[bool] = []

    def reenter_on_register(payload: GateChangedPayload) -> None:
        if not payload.active:
            return
        outer = window.controller
        observed.append((outer.run_id, outer.operation_id, outer.has_pending_segment))
        try:
            window.start("nested")
        except InvalidRunState as exc:
            rejections.append(exc)
        finally:
            preserved.append(window.controller is outer)

    subscription = harness.bus.subscribe(GateChangedPayload, reenter_on_register)
    try:
        token = window.start("outer")
    finally:
        subscription.unsubscribe()
    assert observed == [("outer", token, True)]
    assert len(rejections) == 1 and rejections[0].action == "start"
    assert rejections[0].lifecycle == "running" and preserved == [True]
    controller = window.controller
    assert isinstance(token, int) and controller is not None
    assert (
        controller.run_id == "outer"
        and controller.operation_id == token
        and controller.status() is None
        and harness.gate.snapshot()
    )
    window.stop("outer")
    harness.background.execute()
    assert (
        calls == []
        and controller.status() is None
        and harness.gate.snapshot()
        and controller.operation_id == token
    )
    harness.background.deliver()
    assert not harness.gate.snapshot() and harness.handles.live_count() == 0
    assert controller.operation_id is None and window.controller is controller
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


def test_window_pause_keeps_plan_plots_and_capture_then_resumes_with_lease(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, harness, entry = run_attachment
    attempts: list[float] = []
    figures: list[int] = []

    def init(_env: InitEnv[Calibrated], _plan: T1Plan) -> None:
        return None

    @workflow("pause-window", plan=T1Plan, tunables=T1Tunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated], plan: T1Plan, tun: T1Tunables, state: None
    ) -> Step[None, int]:
        del plan, tun, state
        attempts.append(env.context.pi_length_us)

        def draw(figure: Figure) -> None:
            figures.append(id(figure))
            figure.clear()
            figure.subplots().plot([0, 1], [len(attempts), len(attempts)])

        if len(attempts) == 1:
            env.plots.subplots("live")
        env.plots.draw("live", draw)
        bar = env.pbar("work", 2)
        bar.set_progress(len(attempts))
        bar.refresh()
        if len(attempts) == 1:
            window.pause("paused")
            yield from env.wait_until(env.started_at)
        return Done()

    monkeypatch.setattr(window_module, "t1_fluxdep", step)
    token = window.start("paused")
    assert isinstance(token, int)
    harness.complete_segment()
    controller = window.controller
    assert controller is not None
    paused = controller.status()
    assert paused is not None and paused.lifecycle == "paused"

    # Resume must not capture this later point/alias as new run inputs.
    entry.new_point("later")
    harness.services.context.use_point(PointUseRequest("later"))
    harness.services.context.set_alias_rows((AliasRow("qubit", ""),))
    assert controller.operation_id == token and harness.gate.snapshot()
    assert harness.handles.known_outcome(token) is None
    bars = harness.progress.bars_for_operation(token)
    assert bars
    resume_button = window.findChild(QPushButton, "workflow.run.resume")
    assert resume_button is not None and resume_button.isEnabled()
    resume_button.click()
    resumed = controller.operation_id
    assert resumed == token
    assert harness.handles.live_count() == 1
    assert harness.progress.bars_for_operation(token) == bars
    assert not window.plan_panel.isEnabled()
    harness.complete_segment()
    terminal = controller.status()
    assert (
        terminal is not None
        and terminal.lifecycle == "done"
        and harness.background.error is None
    )
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert len(figures) == 2 and figures[0] == figures[1]
    assert window.plan_panel.isEnabled()
    assert harness.handles.known_outcome(token) is not None
    assert not harness.gate.snapshot()


@pytest.mark.parametrize("action", ["stop", "close"])
def test_resume_submission_stop_or_close_keeps_window_locked_until_delivery(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    window, harness, _entry = run_attachment
    attempts: list[str] = []

    def init(_env: InitEnv[Calibrated], _plan: T1Plan) -> None:
        return None

    @workflow("register-window", plan=T1Plan, tunables=T1Tunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated], plan: T1Plan, tun: T1Tunables, state: None
    ) -> Step[None, int]:
        del plan, tun, state
        attempts.append("step")
        env.pbar("work", 2).refresh()
        window.pause("register-window")
        yield from env.wait_until(env.started_at)
        return Done()

    monkeypatch.setattr(window_module, "t1_fluxdep", step)
    assert isinstance(window.start("register-window"), int)
    harness.complete_segment()
    active_controller = window.controller
    assert active_controller is not None
    observations: list[tuple[bool, bool]] = []
    close_results: list[bool] = []

    token = active_controller.operation_id
    assert token is not None

    def during_submit() -> None:
        if action == "close":
            close_results.append(window.close())
        else:
            window.stop("register-window")
        observations.append(
            (window.plan_panel.isEnabled(), active_controller.has_pending_segment)
        )

    harness.background.on_submit = during_submit
    assert window.resume("register-window") == token
    harness.background.on_submit = None
    assert observations == [(False, True)]
    assert active_controller.has_pending_segment
    assert close_results == ([False] if action == "close" else [])
    assert not window.plan_panel.isEnabled() and harness.gate.snapshot()
    harness.background.execute()
    assert not window.plan_panel.isEnabled() and harness.gate.snapshot()
    harness.background.deliver()
    controller = window.controller
    assert controller is not None
    status = controller.status()
    assert status is not None and status.lifecycle == "stopped"
    assert attempts == ["step"]
    assert window.plan_panel.isEnabled() and window.tunables_panel.isEnabled()
    assert harness.handles.live_count() == 0 and not harness.gate.snapshot()
    assert harness.progress.bars_for_owner("t1-workflow") == ()
    assert window.close()


def test_resume_stop_rejection_unlocks_window_without_a_producer(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, harness, _entry = run_attachment
    attempts: list[str] = []

    def init(_env: InitEnv[Calibrated], _plan: T1Plan) -> None:
        return None

    @workflow("reject-window", plan=T1Plan, tunables=T1Tunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated], plan: T1Plan, tun: T1Tunables, state: None
    ) -> Step[None, int]:
        del plan, tun, state
        attempts.append("step")
        env.pbar("work", 2).refresh()
        window.pause("reject-window")
        yield from env.wait_until(env.started_at)
        return Done()

    monkeypatch.setattr(window_module, "t1_fluxdep", step)
    assert isinstance(window.start("reject-window"), int)
    harness.complete_segment()

    def during_submit() -> None:
        window.stop("reject-window")

    harness.background.on_submit = during_submit
    harness.background.fail_submit = True
    with pytest.raises(OSError, match="submit failed"):
        window.resume("reject-window")
    harness.background.on_submit = None
    controller = window.controller
    assert controller is not None
    status = controller.status()
    assert status is not None and status.lifecycle == "stopped"
    start = window.findChild(QPushButton, "workflow.run.start")
    pause = window.findChild(QPushButton, "workflow.run.pause")
    resume = window.findChild(QPushButton, "workflow.run.resume")
    assert window.plan_panel.isEnabled() and window.tunables_panel.isEnabled()
    assert start is not None and start.isEnabled()
    assert pause is not None and not pause.isEnabled()
    assert resume is not None and not resume.isEnabled()
    assert harness.progress.bars_for_owner("t1-workflow") == ()
    assert not harness.gate.snapshot() and harness.handles.live_count() == 0
    assert not controller.has_pending_segment
    assert harness.background.work is None and attempts == ["step"]
    assert window.close()


@pytest.mark.parametrize("action", ["stop", "close"])
def test_paused_window_stop_or_close_ends_run_without_another_worker(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    window, harness, entry = run_attachment

    def init(_env: InitEnv[Calibrated], _plan: T1Plan) -> None:
        return None

    @workflow("paused-close", plan=T1Plan, tunables=T1Tunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated], plan: T1Plan, tun: T1Tunables, state: None
    ) -> Step[None, int]:
        del plan, tun, state
        env.pbar("work", 2).refresh()
        window.pause("paused-stop")
        yield from env.wait_until(env.started_at)
        return Done()

    monkeypatch.setattr(window_module, "t1_fluxdep", step)
    assert isinstance(window.start("paused-stop"), int)
    harness.complete_segment()
    assert harness.background.work is None
    assert harness.progress.bars_for_owner("t1-workflow")
    assert not window.plan_panel.isEnabled() and not window.tunables_panel.isEnabled()
    assert window.tuning_panel is not None and window.tuning_panel.isEnabled()
    start = window.findChild(QPushButton, "workflow.run.start")
    pause = window.findChild(QPushButton, "workflow.run.pause")
    resume = window.findChild(QPushButton, "workflow.run.resume")
    assert start is not None and not start.isEnabled()
    assert pause is not None and not pause.isEnabled()
    assert resume is not None and resume.isEnabled()
    with pytest.raises(InvalidRunState):
        window.start("replacement")
    assert not (entry.result_path / "workflows/replacement").exists()
    if action == "close":
        assert window.close()
    else:
        status = window.stop("paused-stop")
        assert status is not None and status.lifecycle == "stopped"
        assert window.plan_panel.isEnabled() and start.isEnabled()
        assert not pause.isEnabled() and not resume.isEnabled()
    controller = window.controller
    assert controller is not None
    status = controller.status()
    assert status is not None and status.lifecycle == "stopped"
    assert harness.background.work is None
    assert not harness.gate.snapshot()
    assert harness.progress.bars_for_owner("t1-workflow") == ()


def test_resume_failed_tunables_unwind_unlocks_window_and_preserves_error(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, harness, entry = run_attachment
    native_errors: list[IsADirectoryError] = []
    observed: list[tuple[bool, bool]] = []

    def init(_env: InitEnv[Calibrated], _plan: T1Plan) -> None:
        return None

    @workflow("failed-read", plan=T1Plan, tunables=T1Tunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated], plan: T1Plan, tun: T1Tunables, state: None
    ) -> Step[None, int]:
        del plan, tun, state
        env.pbar("retained", 2).refresh()
        window.pause("failed-read")
        yield from env.wait_until(env.started_at)
        return Done()

    monkeypatch.setattr(window_module, "t1_fluxdep", step)
    token = window.start("failed-read")
    assert isinstance(token, int)
    harness.complete_segment()
    controller, panel = window.controller, window.tuning_panel
    assert panel is not None
    accepted = panel.host.accepted_input.data
    journal = entry.result_path / "workflows/failed-read/journal.jsonl"
    original = journal.read_bytes()
    journal.unlink()
    journal.mkdir()

    def during_submit() -> None:
        try:
            window.apply_tunables(
                "failed-read", {("guess_us",): 45.0}, actor=Actor("agent", "rpc")
            )
        except IsADirectoryError as error:
            native_errors.append(error)
        observed.append((window.plan_panel.isEnabled(), controller.has_pending_segment))

    harness.background.on_submit = during_submit
    harness.background.fail_submit = True
    try:
        with pytest.raises(OSError, match="submit failed"):
            window.resume("failed-read")
    finally:
        harness.background.on_submit = None
        journal.rmdir()
        journal.write_bytes(original)
    assert len(native_errors) == 1 and observed == [(False, True)]
    failed = controller.status()
    assert failed is not None and failed.lifecycle == "failed" and failed.revision == 0
    assert panel.host.accepted_input.data == panel.panel.state.base.data == accepted
    assert panel.panel.state.input.data["guess_us"] == 45.0
    assert window.plan_panel.isEnabled() and window.tunables_panel.isEnabled()
    assert controller.operation_id is None and not controller.has_pending_segment
    assert harness.handles.known_outcome(token) == OperationOutcome(
        "failed", failed.reason
    )
    assert harness.background.work is None and not harness.gate.snapshot()
    assert not harness.progress.bars_for_owner("t1-workflow")
    assert window.close()


def test_prepare_forms_rejects_pending_and_paused_run(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, harness, _entry = run_attachment

    def init(_env: InitEnv[Calibrated], _plan: T1Plan) -> None:
        return None

    @workflow("prepare-guard", plan=T1Plan, tunables=T1Tunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated], plan: T1Plan, tun: T1Tunables, state: None
    ) -> Step[None, int]:
        del plan, tun, state
        window.pause("prepare-guard")
        with pytest.raises(InvalidRunState):
            window.prepare_forms()
        yield from env.wait_until(env.started_at)
        return Done()

    monkeypatch.setattr(window_module, "t1_fluxdep", step)
    before = window.plan_panel
    assert isinstance(window.start("prepare-guard"), int)
    with pytest.raises(InvalidRunState):
        window.prepare_forms()
    harness.complete_segment()
    status = window.controller.status()
    assert status is not None and status.lifecycle == "paused"
    with pytest.raises(InvalidRunState):
        window.prepare_forms()
    assert window.plan_panel is before
    window.stop("prepare-guard")


@pytest.mark.parametrize("command", ["pause", "resume", "stop"])
def test_window_without_run_rejects_explicit_target(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    command: str,
) -> None:
    window, harness, _entry = attachment
    commands: dict[str, Callable[[], object]] = {
        "pause": lambda: window.pause("old"),
        "resume": lambda: window.resume("old"),
        "stop": lambda: window.stop("old"),
    }
    with pytest.raises(RunMismatch) as raised:
        commands[command]()
    assert (raised.value.expected, raised.value.actual) == ("old", None)
    stop_button = window.findChild(QPushButton, "workflow.run.stop")
    assert stop_button is not None
    stop_button.click()
    assert window.controller.run_id is None and window.controller.status() is None
    assert window.controller.binding is None and harness.background.work is None
    assert harness.handles.live_count() == 0 and not harness.gate.snapshot()


@pytest.mark.parametrize("phase", ["pending", "paused"])
def test_old_window_commands_cannot_change_replacement_run_and_gui_stops_current(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    window, harness, entry = run_attachment
    attempts: list[str] = []

    def init(_env: InitEnv[Calibrated], _plan: T1Plan) -> None:
        return None

    @workflow("replacement-target", plan=T1Plan, tunables=T1Tunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated],
        plan: T1Plan,
        tun: T1Tunables,
        state: None,
    ) -> Step[None, int]:
        del plan, tun, state
        attempts.append("step")
        env.plots.subplots("live")
        env.pbar("work", 2).refresh()
        window.pause("run-B")
        yield from env.wait_until(env.started_at)
        return Done()

    monkeypatch.setattr(window_module, "t1_fluxdep", step)
    assert isinstance(window.start("run-A"), int)
    window.stop("run-A")
    harness.complete_segment()
    token = window.start("run-B")
    assert isinstance(token, int)
    controller = window.controller
    assert controller is not None and controller.run_id == "run-B"
    if phase == "paused":
        harness.complete_segment()
        assert harness.background.error is None
    status = controller.status()
    # Lease age advances on each read; compare stable presence fields.
    gate = tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot())
    bars = harness.progress.bars_for_owner("t1-workflow")
    tuning = window.tuning_panel
    for invoke in (
        lambda: window.pause("run-A"),
        lambda: window.resume("run-A"),
        lambda: window.stop("run-A"),
    ):
        with pytest.raises(RunMismatch) as raised:
            invoke()
        assert (raised.value.expected, raised.value.actual) == ("run-A", "run-B")
        assert controller.status() == status
        assert (
            tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot()),
            harness.progress.bars_for_owner("t1-workflow"),
        ) == (gate, bars)
        assert (
            window.tuning_panel,
            window.plan_panel.isEnabled(),
            window.tunables_panel.isEnabled(),
        ) == (tuning, False, False)
    assert attempts == ([] if phase == "pending" else ["step"])
    stop_button = window.findChild(QPushButton, "workflow.run.stop")
    assert stop_button is not None
    stop_button.click()
    if phase == "pending":
        assert harness.gate.snapshot()
        harness.complete_segment()
        assert not (entry.result_path / "workflows/run-B").exists()
    else:
        stopped = controller.status()
        assert stopped is not None and stopped.lifecycle == "stopped"
    assert controller.run_id == "run-B" and window.plan_panel.isEnabled()
    assert harness.progress.bars_for_owner("t1-workflow") == ()
    assert not harness.gate.snapshot() and harness.handles.live_count() == 0


RpcRequest = Callable[[str, dict[str, object]], dict[str, object]]


def test_form_inspection_rpc_reads_constructor_drafts(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    rpc: RpcRequest,
) -> None:
    window, _harness, _entry = attachment
    for form, panel in (
        ("plan", window.plan_panel),
        ("tunables", window.tunables_panel),
    ):
        before = panel.state.input.data, panel.state.base.data
        reply = rpc("workflow.form", {"form": form})
        assert reply["ok"] is True
        result = reply["result"]
        assert isinstance(result, dict)
        assert result["input"] == before[0]
        assert result["revision"] is None
        assert isinstance(result["schema"], dict)
        assert isinstance(result["values"], list)
        assert isinstance(result["errors"], list)
        assert (panel.state.input.data, panel.state.base.data) == before
    assert window.controller.binding is None


@pytest.mark.parametrize(
    "value", [{"expr": "2 * 7"}, {"expr": "missing_knob"}, {"expr": "1e308 * 1e308"}]
)
def test_form_inspection_rpc_candidate_does_not_write_start_draft(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    rpc: RpcRequest,
    value: JsonValue,
) -> None:
    window, _harness, _entry = attachment
    state = window.tunables_panel.state
    before = state.input.data, state.base.data
    reply = rpc(
        "workflow.form",
        {"form": "tunables", "batch": [{"path": ["guess_us"], "value": value}]},
    )
    assert reply["ok"] is True
    result = reply["result"]
    assert isinstance(result, dict)
    candidate = result["input"]
    assert isinstance(candidate, dict) and candidate["guess_us"] == value
    assert (state.input.data, state.base.data) == before
    if value == {"expr": "2 * 7"}:
        assert {"path": ["guess_us"], "value": 14.0} in result["values"]
        assert result["errors"] == []
    else:
        assert any(e["path"] == ["guess_us"] for e in result["errors"])
        assert not any(v["path"] == ["guess_us"] for v in result["values"])


def test_form_inspection_rpc_preserves_complex_tag_on_invalid_numeric_candidate(
    rpc: RpcRequest,
) -> None:
    tagged = {"__complex__": [1.0, 2.0]}
    reply = rpc(
        "workflow.form",
        {
            "form": "tunables",
            "batch": [{"path": ["guess_us"], "value": tagged}],
        },
    )
    assert reply["ok"] is True
    result = reply["result"]
    assert isinstance(result, dict)
    assert {"path": ["guess_us"], "value": tagged} in result["values"]
    assert any(e["path"] == ["guess_us"] for e in result["errors"])


def test_form_inspection_rpc_plan_branch_batch_does_not_change_gui(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    rpc: RpcRequest,
) -> None:
    window, _harness, _entry = attachment
    before = window.plan_panel.state.input.data, window.plan_panel.state.base.data
    reply = rpc(
        "workflow.form",
        {
            "form": "plan",
            "batch": [
                {"path": ["fluxes", "kind"], "value": "linrange"},
                {"path": ["fluxes", "start"], "value": {"expr": "1 / 2"}},
                {"path": ["fluxes", "stop"], "value": 1.0},
                {"path": ["fluxes", "n"], "value": 3},
            ],
        },
    )
    assert reply["ok"] is True
    result = reply["result"]
    assert isinstance(result, dict)
    assert result["errors"] == []
    assert {"path": ["fluxes", "start"], "value": 0.5} in result["values"]
    assert (
        window.plan_panel.state.input.data,
        window.plan_panel.state.base.data,
    ) == before


def test_form_inspection_rpc_uses_accepted_host_and_preserves_local_text(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    paused_tuning: list[tuple[int, float, int]],
    rpc: RpcRequest,
) -> None:
    del paused_tuning
    window, harness, _entry = attachment
    panel = window.tuning_panel
    engine = window.controller.engine
    assert panel is not None and engine is not None
    state = panel.host.state
    state.write({("guess_us",): {"expr": "3 * 11"}}, panel.host.env)
    local = panel.panel.editor.findChild(QLineEdit, "form.input.sweep_factor")
    assert isinstance(local, QLineEdit)
    local.setText("-")
    before = (
        state.input.data,
        state.base.data,
        panel.host.accepted_input.data,
        engine.tunables("tuning"),
        window.controller.operation_id,
        tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot()),
    )
    read = rpc("workflow.form", {"form": "tunables", "run_id": "tuning"})
    assert read["ok"] is True
    result = read["result"]
    assert isinstance(result, dict)
    assert result["input"] == before[0]
    assert result["revision"] == before[3].revision
    assert {"path": ["guess_us"], "value": 33.0} in result["values"]
    assert any(e["path"] == ["sweep_factor"] for e in result["errors"])
    for batch in (
        [],
        [{"path": ["guess_us"], "value": {"expr": "2 * 20"}}],
        [{"path": ["guess_us"], "value": -1}],
    ):
        reply = rpc(
            "workflow.form",
            {
                "form": "tunables",
                "run_id": "tuning",
                "batch": batch,
            },
        )
        assert reply["ok"] is True
        preview = reply["result"]
        assert isinstance(preview, dict)
        assert not any(e["path"] == ["sweep_factor"] for e in preview["errors"])
        if batch == []:
            assert preview["input"] == before[0]
        elif batch[0]["value"] == -1:
            assert any(e["path"] == ["guess_us"] for e in preview["errors"])
        else:
            assert {"path": ["guess_us"], "value": 40.0} in preview["values"]
    assert (
        state.input.data,
        state.base.data,
        panel.host.accepted_input.data,
        engine.tunables("tuning"),
        window.controller.operation_id,
        tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot()),
    ) == before
    assert local.text() == "-"
    assert panel.panel.editor.local_parse_errors() is not None


@pytest.mark.parametrize(
    "params",
    [
        {"form": "tunables", "batch": {}},
        {"form": "tunables", "batch": [{"path": ["unknown"], "value": 2}]},
        {"form": "tunables", "batch": [{"path": [], "value": 2}]},
        {
            "form": "tunables",
            "batch": [
                {"path": ["reps"], "value": 2},
                {"path": ["reps"], "value": 3},
            ],
        },
    ],
)
def test_form_inspection_rpc_rejects_bad_form_or_batch_without_writes(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    rpc: RpcRequest,
    params: dict[str, object],
) -> None:
    window, _harness, _entry = attachment
    before = window.tunables_panel.state.input.data
    reply = rpc("workflow.form", params)
    assert reply["ok"] is False
    error = reply["error"]
    assert isinstance(error, dict) and error["code"] == "invalid_params"
    assert window.tunables_panel.state.input.data == before


def test_form_inspection_rpc_rejects_unknown_form_at_parameter_seam(
    rpc: RpcRequest,
) -> None:
    with pytest.raises(RemoteError) as raised:
        rpc("workflow.form", {"form": "other"})
    assert raised.value.code is ErrorCode.INVALID_PARAMS


def test_form_inspection_rpc_requires_accepted_explicit_target(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    rpc: RpcRequest,
) -> None:
    window, harness, _entry = run_attachment
    absent = rpc("workflow.form", {"form": "plan", "run_id": "absent"})
    assert absent["ok"] is False
    token = window.start("pending-form")
    assert isinstance(token, int)
    for run_id, expected in (
        ("pending-form", "controller_error"),
        ("wrong", "precondition_failed"),
        (None, "precondition_failed"),
    ):
        reply = rpc("workflow.form", {"form": "tunables", "run_id": run_id})
        assert reply["ok"] is False
        error = reply["error"]
        assert isinstance(error, dict) and error["code"] == expected
    window.stop("pending-form")
    harness.complete_segment()


WireRoute = Callable[[Request], dict[str, object]]


@pytest.fixture
def wire_route(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    inputs: Inputs,
) -> Iterator[WireRoute]:
    """Route original requests off-owner and pump the real owner for RPC and MCP."""
    window, harness, _entry = attachment
    context = harness.services.context

[Showing lines 1-1347 of 3494 (50.0KB limit). Use offset=1348 to continue.]