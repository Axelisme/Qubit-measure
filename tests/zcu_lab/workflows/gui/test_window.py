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
        context.set_component_focus("Q1")
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
            harness.runner,
            harness.owner,
            harness.progress,
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
    """Supply a controlled device snapshot to test Window/Controller ownership.

    DeviceService's real live-read contract has separate formals. This boundary
    substitute lets Window ownership tests reach admission before that seed body
    is implemented; it does not replace Runner, Controller or native Engine.
    """
    _window, harness, _entry = attachment
    native: dict[str, DeviceInfo] = {"flux": FakeDeviceInfo(address="none", value=0.0)}
    monkeypatch.setattr(
        harness.services.device, "capture_live_native_device_info", lambda: native
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


def test_start_live_read_failure_names_device_before_admission(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, harness, entry = attachment
    driver = harness.services.device.get_connected_devices()["flux"]
    before = window.plan_panel.state.input.data

    def reject_read() -> DeviceInfo:
        raise OSError("instrument read failed")

    monkeypatch.setattr(driver, "get_info", reject_read)
    with pytest.raises(RuntimeError, match="flux"):
        window.start("live-read-failure")

    assert window.plan_panel.state.input.data == before
    assert harness.background.work is None and harness.handles.live_count() == 0
    assert not harness.gate.snapshot()
    assert not (entry.result_path / "workflows/live-read-failure").exists()
    assert not (entry.database_path / "workflows/live-read-failure").exists()


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
    captured: list[float] = []
    live_value = 0.12

    def capture() -> dict[str, DeviceInfo]:
        captured.append(live_value)
        return {"flux": FakeDeviceInfo(address="none", value=live_value)}

    def forbid_cache() -> dict[str, DeviceInfo]:
        raise AssertionError("Start/Resume must call the live capture boundary")

    monkeypatch.setattr(
        harness.services.device, "capture_live_native_device_info", capture
    )
    monkeypatch.setattr(
        harness.services.device, "capture_native_device_info", forbid_cache
    )

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

    # Resume must not capture this later point/focus as new run inputs.
    entry.new_point("later")
    harness.services.context.use_point(PointUseRequest("later"))
    harness.services.context.set_component_focus(None)
    assert captured == [0.12]
    assert controller.operation_id == token and harness.gate.snapshot()
    assert harness.handles.known_outcome(token) is None
    bars = harness.progress.bars_for_operation(token)
    assert bars
    live_value = 0.34
    resume_button = window.findChild(QPushButton, "workflow.run.resume")
    assert resume_button is not None and resume_button.isEnabled()
    resume_button.click()
    resumed = controller.operation_id
    assert resumed == token and captured == [0.12, 0.34]
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
    journal = entry.result_path / "workflows/paused/journal.jsonl"
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    started = next(event for event in events if event["kind"] == "run_started")
    resumed_event = next(event for event in events if event["kind"] == "resumed")
    assert started["identity"]["devices"]["items"]["flux"]["value"] == 0.12
    assert resumed_event["devices"]["items"]["flux"]["value"] == 0.34
    assert window.plan_panel.isEnabled()
    assert harness.handles.known_outcome(token) is not None
    assert not harness.gate.snapshot()


@pytest.mark.parametrize("action", ["stop", "close"])
def test_resume_capture_stop_or_close_keeps_window_locked_until_delivery(
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
    capture_live = harness.services.device.capture_live_native_device_info

    def capture() -> dict[str, DeviceInfo]:
        if action == "close":
            close_results.append(window.close())
        else:
            window.stop("register-window")
        observations.append(
            (window.plan_panel.isEnabled(), active_controller.has_pending_segment)
        )
        return capture_live()

    monkeypatch.setattr(
        harness.services.device, "capture_live_native_device_info", capture
    )
    assert window.resume("register-window") == token
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


@pytest.mark.parametrize("failure", ["capture", "submit"])
def test_resume_stop_rejection_unlocks_window_without_a_producer(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
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
    capture_native = harness.services.device.capture_live_native_device_info
    capture_error = ValueError("flux live snapshot failed after Stop")

    def capture() -> dict[str, DeviceInfo]:
        window.stop("reject-window")
        if failure == "capture":
            raise capture_error
        return capture_native()

    monkeypatch.setattr(
        harness.services.device, "capture_live_native_device_info", capture
    )
    harness.background.fail_submit = failure == "submit"
    with pytest.raises(ValueError if failure == "capture" else OSError) as raised:
        window.resume("reject-window")
    if failure == "capture":
        assert raised.value is capture_error
    else:
        assert str(raised.value) == "submit failed"
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


def test_resume_live_capture_failure_retains_paused_run_and_presentation(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, harness, _entry = run_attachment

    def init(_env: InitEnv[Calibrated], _plan: T1Plan) -> None:
        return None

    @workflow("read-rejection", plan=T1Plan, tunables=T1Tunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated], plan: T1Plan, tun: T1Tunables, state: None
    ) -> Step[None, int]:
        del plan, tun, state
        env.pbar("retained", 2).refresh()
        window.pause("read-rejection")
        yield from env.wait_until(env.started_at)
        return Done()

    monkeypatch.setattr(window_module, "t1_fluxdep", step)
    token = window.start("read-rejection")
    assert isinstance(token, int)
    harness.complete_segment()
    controller = window.controller
    paused = controller.status()
    binding = controller.binding
    panel = window.tuning_panel
    bars = harness.progress.bars_for_operation(token)
    assert paused is not None and paused.lifecycle == "paused" and bars
    failure = RuntimeError("flux: live instrument read failed")

    def reject_live() -> dict[str, DeviceInfo]:
        raise failure

    def forbid_cache() -> dict[str, DeviceInfo]:
        raise AssertionError("Resume must not fall back to cached device info")

    monkeypatch.setattr(
        harness.services.device, "capture_live_native_device_info", reject_live
    )
    monkeypatch.setattr(
        harness.services.device, "capture_native_device_info", forbid_cache
    )
    with pytest.raises(RuntimeError, match="flux") as raised:
        window.resume("read-rejection")
    assert raised.value is failure
    assert window.controller is controller and controller.status() == paused
    assert controller.binding is binding and window.tuning_panel is panel
    assert controller.operation_id == token and not controller.has_pending_segment
    assert harness.background.work is None and harness.handles.live_count() == 1
    assert harness.handles.known_outcome(token) is None and harness.gate.snapshot()
    assert harness.progress.bars_for_operation(token) == bars
    assert not window.plan_panel.isEnabled() and not window.tunables_panel.isEnabled()
    resume = window.findChild(QPushButton, "workflow.run.resume")
    assert resume is not None and resume.isEnabled()
    stopped = window.stop("read-rejection")
    assert stopped is not None and stopped.lifecycle == "stopped"
    assert not harness.gate.snapshot() and harness.handles.live_count() == 0


@pytest.mark.parametrize("failure", ["capture", "submit"])
def test_resume_failed_tunables_unwind_unlocks_window_and_preserves_error(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
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
    capture_error = OSError("flux live read failed during Resume")
    journal = entry.result_path / "workflows/failed-read/journal.jsonl"
    original = journal.read_bytes()
    journal.unlink()
    journal.mkdir()

    def capture() -> dict[str, DeviceInfo]:
        try:
            window.apply_tunables(
                "failed-read", {("guess_us",): 45.0}, actor=Actor("agent", "rpc")
            )
        except IsADirectoryError as error:
            native_errors.append(error)
        observed.append((window.plan_panel.isEnabled(), controller.has_pending_segment))
        if failure == "capture":
            raise capture_error
        return {"flux": FakeDeviceInfo(address="none", value=0.34)}

    monkeypatch.setattr(
        harness.services.device, "capture_live_native_device_info", capture
    )
    harness.background.fail_submit = failure == "submit"
    try:
        with pytest.raises(OSError, match="failed") as raised:
            window.resume("failed-read")
    finally:
        journal.rmdir()
        journal.write_bytes(original)
    assert (
        raised.value is capture_error
        if failure == "capture"
        else str(raised.value) == "submit failed"
    )
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
    captures: list[str] = []
    original_capture = harness.services.device.capture_live_native_device_info

    def capture() -> dict[str, DeviceInfo]:
        captures.append("devices")
        return original_capture()

    monkeypatch.setattr(
        harness.services.device, "capture_live_native_device_info", capture
    )
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
    assert captures == [] and attempts == ([] if phase == "pending" else ["step"])
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
    library = library_env(inputs).library
    assert library is not None
    target = context.read_context().target
    for key in ("Q1.pulses.pi", "Q1.readout.ro"):
        context.set_module(target, key, library.module_spec(key))
    service = window.create_remote_service(ControlOptions(port=0), harness.bus)
    link = ClientLink("workflow-contract", token_required=False)
    service.on_client_open(link)
    try:
        with ThreadPoolExecutor(max_workers=1) as io:

            def request(original: Request) -> dict[str, object]:
                pending = io.submit(service.route, link, original)
                deadline = monotonic() + 3
                while not pending.done():
                    assert monotonic() < deadline, "RPC did not settle on the owner"
                    harness.owner.pump_once(block=True, timeout=0.01)
                pending.result(timeout=1)
                return json.loads(link.outbound.get(timeout=1))

            yield request
    finally:
        service.on_client_close(link, on_owner_thread=True)
        service.stop()


@pytest.fixture
def rpc(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    wire_route: WireRoute,
) -> RpcRequest:
    """Keep existing RPC cases on their controlled live-device seam."""
    return lambda method, params: wire_route(Request("request", method, params))


class WindowTransport:
    """Move wire messages through the Window route, replacing only TCP.

    route sends one Request through public GUI dispatch and returns its wire reply.
    requests retains submitted identities for forwarding assertions. deliver_reply
    is the bridge callback, or None before attach. is_open is true until close.
    attach supplies bridge callbacks before send_line; close only detaches bytes.
    Owner dispatch and native execution remain with the existing GUI harness.
    """

    def __init__(self, route: WireRoute) -> None:
        self.route = route
        self.requests: list[Request] = []
        self.deliver_reply: DeliverFn | None = None
        self.is_open = True

    def attach(
        self, deliver_reply: DeliverFn, deliver_event: DeliverFn, on_closed: OnClosedFn
    ) -> None:
        """Install the bridge reply callback; this route publishes no events."""
        self.deliver_reply = deliver_reply

    def send_line(self, payload: dict[str, object]) -> None:
        """Round-trip the original JSON message without rewriting its identity."""
        if not self.is_open or self.deliver_reply is None:
            raise RuntimeError("Window transport is not attached")
        request = parse_request(json.loads(json.dumps(payload)))
        self.requests.append(request)
        self.deliver_reply(self.route(request))

    def close(self) -> None:
        """Detach the transport without issuing Stop to the Window."""
        self.is_open = False


@pytest.fixture
def mcp_transport(wire_route: WireRoute) -> WindowTransport:
    return WindowTransport(wire_route)


@pytest.fixture
def mcp_tools(mcp_transport: WindowTransport, tmp_path: Path) -> Iterator[ToolTable]:
    """Bind real MCP tools to this test's Window without launching any process."""
    config = MCPBridgeConfig(
        tool_prefix="workflows_",
        server_display_name="workflows",
        server_instructions="Workflow RPC contract",
        app_name="workflows",
        default_port=8768,
        mcp_version=1,
        wire_version=1,
        pid_file=tmp_path / "gui.pid",
        log_file=tmp_path / "gui.log",
        run_script_name="run_workflows_gui.py",
        app_slug="workflows",
    )
    bridge = McpBridge(config, transport=mcp_transport)
    try:
        yield build_workflow_tools(config, bridge, tmp_path)
    finally:
        bridge.disconnect()


def test_mcp_native_start_acquires_and_reads_committed_records_and_png(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    inputs: Inputs,
    mcp_tools: ToolTable,
    mcp_transport: WindowTransport,
) -> None:
    """Use the production T1 workflow and offline SoC, not a declared fake step."""
    window, harness, entry = attachment
    start = mcp_tools["workflows_workflow_start"]["handler"]
    plan = _start_plan_batch(inputs)
    for edit in plan:
        if edit["path"] == ["fluxes", "values"]:
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
                    {"path": ["reps"], "value": 0},
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
                {"path": ["cal", "span_mhz"], "value": 2.0},
                {"path": ["cal", "rabi_start_factor"], "value": 0.125},
                {"path": ["rounds"], "value": 2},
            ],
        }
    )
    token = window.controller.operation_id
    assert isinstance(token, int)
    assert pending == {"run_id": "mcp-native", "operation_id": token, "status": None}
    assert window.plan_panel.state.input.data["pi_length_us"] == {"expr": "0.06 * 2"}
    pulse = window.plan_panel.state.input.data["qub_pulse"]
    assert isinstance(pulse, dict) and pulse["name"] == "Q1.pulses.pi"
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
        "run_files": list(record.run_files),
    }
    assert mcp_tools["workflows_workflow_records"]["handler"](
        {"run_id": "mcp-native"}
    ) == {
        "run_id": "mcp-native",
        "records": [expected],
    }
    assert len(record.run_files) >= 4
    for relative in record.run_files:
        assert (
            entry.database_path
            / "workflows/mcp-native"
            / record.iteration_dir
            / relative
        ).is_file()
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


@pytest.fixture
def mcp_paused_controls(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    inputs: Inputs,
    mcp_tools: ToolTable,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[tuple[int, float, float, str | None]], list[dict[str, object]]]:
    """Pause via MCP after one commit; return attempts and later Stop receipts.

    Each attempt holds state, guess_us, captured pi_length_us and source point.
    Declared steps expose control timing; they are not acquisition evidence.
    """
    window, harness, _entry = attachment
    attempts: list[tuple[int, float, float, str | None]] = []
    pause_receipts: list[dict[str, object]] = []
    stop_receipts: list[dict[str, object]] = []

    def init(_env: InitEnv[Calibrated], _plan: T1Plan) -> int:
        return 0

    @workflow("mcp-controls", plan=T1Plan, tunables=T1Tunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated], plan: T1Plan, tun: T1Tunables, state: int
    ) -> Step[int, float]:
        attempts.append(
            (state, tun.guess_us, env.context.pi_length_us, plan.snapshot.point)
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
        return Next(tun.guess_us, state + 1)

    monkeypatch.setattr(window_module, "t1_fluxdep", step)
    pending = mcp_tools["workflows_workflow_start"]["handler"](
        {
            "run_id": "mcp-controls",
            "plan": _start_plan_batch(inputs),
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


def test_mcp_native_controls_keep_capture_lease_and_record_revisions(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
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
    context.set_parameter(context.read_context().target, "general.ext.guess_us", 99.0)
    entry.new_point("later-mcp")
    context.use_point(PointUseRequest("later-mcp"))
    context.set_component_focus(None)
    inspected = mcp_tools["workflows_workflow_form"]["handler"](
        {"form": "tunables", "run_id": "mcp-controls"}
    )
    assert inspected["revision"] == 0
    assert inspected["input"]["guess_us"] == {"expr": "general.ext.guess_us"}
    assert {"path": ["guess_us"], "value": 20.0} in inspected["values"]
    applied = mcp_tools["workflows_workflow_tunables"]["handler"](
        {
            "run_id": "mcp-controls",
            "batch": [
                {"path": ["guess_us"], "value": {"expr": "general.ext.guess_us * 2"}},
            ],
        }
    )
    assert applied["status"]["revision"] == 1
    assert engine.records("mcp-controls") == (first,)
    panel = window.tuning_panel
    assert panel is not None
    assert panel.host.accepted_input.data["guess_us"] == {
        "expr": "general.ext.guess_us * 2"
    }
    observed = mcp_tools["workflows_workflow_form"]["handler"](
        {"form": "tunables", "run_id": "mcp-controls"}
    )
    assert observed["revision"] == 1
    assert {"path": ["guess_us"], "value": 40.0} in observed["values"]
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
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
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
                "batch": [{"path": ["guess_us"], "value": 500.0}],
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
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    inputs: Inputs,
    mcp_tools: ToolTable,
    mcp_transport: WindowTransport,
    method: str,
) -> None:
    window, harness, entry = attachment
    pending = mcp_tools["workflows_workflow_start"]["handler"](
        {
            "run_id": "mcp-pending",
            "plan": _start_plan_batch(inputs),
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


@pytest.mark.parametrize(
    "method", ["workflow.records", "workflow.figures", "workflow.figure"]
)
def test_rpc_read_targets_preserve_pending_and_absent(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    rpc: RpcRequest,
    method: str,
) -> None:
    window, harness, entry = run_attachment
    params: dict[str, object] = {"run_id": "reader"}
    if method == "workflow.figure":
        params["name"] = "live"
    absent = rpc(method, params)
    assert absent["ok"] is False
    error = absent["error"]
    assert isinstance(error, dict)
    assert (error["code"], error["reason"], error["data"]) == (
        "precondition_failed",
        "run_mismatch",
        {"expected": "reader", "actual": None},
    )
    token = window.start("reader")
    assert isinstance(token, int)
    pending = rpc(method, params)
    assert pending["ok"] is False
    error = pending["error"]
    assert isinstance(error, dict) and error["code"] == "controller_error"
    assert window.controller.operation_id == token
    assert window.controller.status() is None and harness.gate.snapshot()
    assert not (entry.result_path / "workflows/reader").exists()
    window.stop("reader")
    harness.complete_segment()


@pytest.mark.parametrize(
    "method", ["workflow.records", "workflow.figures", "workflow.figure"]
)
@pytest.mark.parametrize("run_id", [None, "", 3])
def test_rpc_read_requires_target(rpc: RpcRequest, method: str, run_id: object) -> None:
    params: dict[str, object] = {"run_id": run_id, "name": "live"}
    with pytest.raises(RemoteError) as error:
        rpc(method, params)
    assert error.value.code is ErrorCode.INVALID_PARAMS


@pytest.mark.parametrize("since", [True, "1", 1.5])
def test_rpc_records_rejects_noninteger_cursor(rpc: RpcRequest, since: object) -> None:
    with pytest.raises(RemoteError) as error:
        rpc("workflow.records", {"run_id": "absent", "since": since})
    assert error.value.code is ErrorCode.INVALID_PARAMS


def test_rpc_records_retains_commits_and_revision_after_resume(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    paused_tuning: list[tuple[int, float, int]],
    rpc: RpcRequest,
) -> None:
    window, harness, _entry = attachment
    engine = window.controller.engine
    assert engine is not None and len(paused_tuning) == 2
    first = engine.records("tuning")[0]
    expected = {
        "commit_seq": first.commit_seq,
        "call_seq": first.call_seq,
        "revision": first.revision,
        "encoded_record": asdict(first.encoded_record),
        "iteration_dir": first.iteration_dir,
        "run_files": list(first.run_files),
    }
    assert rpc("workflow.records", {"run_id": "tuning"})["result"] == {
        "run_id": "tuning",
        "records": [expected],
    }
    assert rpc("workflow.records", {"run_id": "tuning", "since": first.commit_seq})[
        "result"
    ] == {"run_id": "tuning", "records": []}
    negative = rpc("workflow.records", {"run_id": "tuning", "since": -1})
    assert negative["ok"] is False
    error = negative["error"]
    assert isinstance(error, dict) and error["code"] == "invalid_params"
    assert (
        rpc(
            "workflow.tunables",
            {"run_id": "tuning", "batch": [{"path": ["guess_us"], "value": 41.0}]},
        )["ok"]
        is True
    )
    assert rpc("workflow.resume", {"run_id": "tuning"})["ok"] is True
    harness.complete_segment()
    result = rpc("workflow.records", {"run_id": "tuning", "since": 1})["result"]
    records = engine.records("tuning", since=1)
    assert len(records) == 1 and records[0].revision == 1
    expected = {
        "commit_seq": records[0].commit_seq,
        "call_seq": records[0].call_seq,
        "revision": records[0].revision,
        "encoded_record": asdict(records[0].encoded_record),
        "iteration_dir": records[0].iteration_dir,
        "run_files": list(records[0].run_files),
    }
    assert result == {"run_id": "tuning", "records": [expected]}
    assert engine.records("tuning")[0] == first
    assert window.controller.status() is not None and not harness.gate.snapshot()


def test_rpc_records_running_empty_and_committed_native_summary(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    rpc: RpcRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, harness, _entry = attachment
    observations: list[dict[str, object]] = []

    def init(_env: InitEnv[Calibrated], _plan: T1Plan) -> int:
        return 0

    @workflow("read-running", plan=T1Plan, tunables=T1Tunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated], plan: T1Plan, tun: T1Tunables, state: int
    ) -> Step[int, dict[str, float]]:
        del env, plan, tun
        observations.append(rpc("workflow.records", {"run_id": "reader"}))
        yield from ()
        if state == 0:
            return Next({"measurement": 2.5}, 1)
        return Done()

    monkeypatch.setattr(window_module, "t1_fluxdep", step)
    assert isinstance(window.start("reader"), int)
    harness.complete_segment()
    assert harness.background.error is None
    assert observations[0]["result"] == {"run_id": "reader", "records": []}
    engine = window.controller.engine
    assert engine is not None
    native = engine.records("reader")[0]
    expected = {
        "commit_seq": native.commit_seq,
        "call_seq": native.call_seq,
        "revision": native.revision,
        "encoded_record": asdict(native.encoded_record),
        "iteration_dir": native.iteration_dir,
        "run_files": list(native.run_files),
    }
    assert observations[1]["result"] == {"run_id": "reader", "records": [expected]}
    assert native.encoded_record.value == {"measurement": 2.5}


@pytest.mark.parametrize("ending", ["resume", "stop"])
def test_rpc_figures_latest_owner_render_and_lifetime(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    rpc: RpcRequest,
    monkeypatch: pytest.MonkeyPatch,
    ending: str,
) -> None:
    window, harness, _entry = attachment
    renderer = window_module.render_figure_png
    owner_id = get_ident()
    renders: list[int] = []
    figures: list[Figure] = []
    attempts: list[int] = []

    def render(figure: Figure) -> bytes:
        assert figure.get_facecolor() == (0.0, 0.0, 1.0, 1.0)
        renders.append(get_ident())
        figures.append(figure)
        return renderer(figure)

    monkeypatch.setattr(window_module, "render_figure_png", render)

    def init(_env: InitEnv[Calibrated], _plan: T1Plan) -> int:
        return 0

    @workflow("read-plots", plan=T1Plan, tunables=T1Tunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated], plan: T1Plan, tun: T1Tunables, state: int
    ) -> Step[int, int]:
        del plan, tun
        attempts.append(state)
        if state == 0:
            env.plots.subplots("live")
            env.plots.subplots("reference")
            env.plots.draw("live", lambda figure: figure.set_facecolor("red"))
            yield from ()
            return Next(1, 1)
        env.plots.draw("live", lambda figure: figure.set_facecolor("blue"))
        if len(attempts) == 2:
            window.pause("reader")
            yield from env.wait_until(env.started_at)
        return Done()

    monkeypatch.setattr(window_module, "t1_fluxdep", step)
    assert isinstance(window.start("reader"), int)
    harness.complete_segment()
    assert harness.background.error is None
    assert rpc("workflow.figures", {"run_id": "reader"})["result"] == {
        "run_id": "reader",
        "names": ["live", "reference"],
    }
    assert window.figure_names("reader") == ("live", "reference")
    assert rpc("workflow.status", {})["ok"] is True and renders == []
    reply = rpc("workflow.figure", {"run_id": "reader", "name": "live"})
    result = reply["result"]
    assert isinstance(result, dict) and isinstance(result["png_base64"], str)
    png = b64decode(result["png_base64"], validate=True)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert result["run_id"] == "reader" and result["name"] == "live"
    assert renders == [owner_id]
    assert window.figure_png("reader", "live") == png
    assert figures[0] is figures[1]
    missing = rpc("workflow.figure", {"run_id": "reader", "name": "missing"})
    error = missing["error"]
    assert missing["ok"] is False and isinstance(error, dict)
    assert (error["code"], error["reason"]) == ("invalid_params", "unknown_figure")
    assert len(renders) == 2

    def failed_render(_figure: Figure) -> bytes:
        raise RuntimeError("Renderer failed")

    monkeypatch.setattr(window_module, "render_figure_png", failed_render)
    failed = rpc("workflow.figure", {"run_id": "reader", "name": "live"})
    error = failed["error"]
    assert failed["ok"] is False and isinstance(error, dict)
    assert error["code"] == "controller_error" and error["message"] == "Renderer failed"
    with pytest.raises(RuntimeError, match="Renderer failed"):
        window.figure_png("reader", "live")
    assert harness.gate.snapshot()
    monkeypatch.setattr(window_module, "render_figure_png", render)
    assert rpc("workflow." + ending, {"run_id": "reader"})["ok"] is True
    if ending == "resume":
        harness.complete_segment()
    assert window.figure_png("reader", "live") == png
    assert window.close()
    closed = rpc("workflow.figures", {"run_id": "reader"})
    error = closed["error"]
    assert closed["ok"] is False and isinstance(error, dict)
    assert (error["code"], error["reason"]) == ("precondition_failed", "no_plots")


def test_rpc_empty_figures_and_stale_run_do_not_render(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    paused_tuning: list[tuple[int, float, int]],
    rpc: RpcRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, harness, _entry = attachment
    assert paused_tuning
    renders: list[Figure] = []

    def reject_render(figure: Figure) -> bytes:
        renders.append(figure)
        raise AssertionError("Read must not render")

    monkeypatch.setattr(window_module, "render_figure_png", reject_render)
    assert rpc("workflow.figures", {"run_id": "tuning"})["result"] == {
        "run_id": "tuning",
        "names": [],
    }
    for method in ("workflow.records", "workflow.figures", "workflow.figure"):
        stale = rpc(method, {"run_id": "old", "name": "live"})
        error = stale["error"]
        assert stale["ok"] is False and isinstance(error, dict)
        assert (error["code"], error["reason"], error["data"]) == (
            "precondition_failed",
            "run_mismatch",
            {"expected": "old", "actual": "tuning"},
        )
    assert renders == [] and harness.gate.snapshot()
    assert rpc("workflow.stop", {"run_id": "tuning"})["ok"] is True
    # New Start must not retain even a previous terminal run's figure membership.
    assert isinstance(window.start("replacement"), int)
    assert window.controller.status() is None
    stale = rpc("workflow.figures", {"run_id": "tuning"})
    assert stale["ok"] is False
    window.stop("replacement")
    harness.complete_segment()


def test_rpc_pending_status_and_stop(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    rpc: RpcRequest,
) -> None:
    window, harness, entry = run_attachment
    assert rpc("workflow.status", {})["result"] == {
        "run_id": None,
        "operation_id": None,
        "status": None,
    }
    token = window.start("pending-rpc")
    assert isinstance(token, int)
    expected = {"run_id": "pending-rpc", "operation_id": token, "status": None}
    assert rpc("workflow.status", {})["result"] == expected
    assert rpc("workflow.stop", {"run_id": "pending-rpc"})["result"] == expected
    assert harness.gate.snapshot()
    harness.complete_segment()
    assert not harness.gate.snapshot()
    assert not (entry.result_path / "workflows/pending-rpc").exists()
    assert window.plan_panel.isEnabled()


@pytest.mark.parametrize("method", ["workflow.pause", "workflow.resume"])
def test_rpc_pending_commands_do_not_claim_acceptance(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    rpc: RpcRequest,
    method: str,
) -> None:
    window, harness, _entry = attachment
    token = window.start("pending-rpc")
    assert isinstance(token, int)
    native_command = window.pause if method == "workflow.pause" else window.resume
    with pytest.raises(RuntimeError) as native:
        native_command("pending-rpc")
    reply = rpc(method, {"run_id": "pending-rpc"})
    error = reply["error"]
    assert reply["ok"] is False and isinstance(error, dict)
    assert error["code"] == "controller_error"
    assert error["message"] == str(native.value)
    assert window.controller is not None and window.controller.status() is None
    assert window.controller.operation_id == token and harness.gate.snapshot()


@pytest.mark.parametrize(
    "method", ["workflow.pause", "workflow.resume", "workflow.stop"]
)
@pytest.mark.parametrize(
    "params", [{}, {"run_id": None}, {"run_id": 3}, {"run_id": ""}]
)
def test_rpc_target_parameter_admission(
    rpc: RpcRequest,
    method: str,
    params: dict[str, object],
) -> None:
    # Public route exposes ParamSpec errors; the endpoint envelopes these on the wire.
    with pytest.raises(RemoteError) as raised:
        rpc(method, params)
    assert raised.value.code is ErrorCode.INVALID_PARAMS


@pytest.mark.parametrize(
    "method", ["workflow.pause", "workflow.resume", "workflow.stop"]
)
def test_rpc_without_attached_run_reports_native_target(
    rpc: RpcRequest,
    method: str,
) -> None:
    reply = rpc(method, {"run_id": "absent"})
    error = reply["error"]
    assert reply["ok"] is False and isinstance(error, dict)
    assert (error["code"], error["reason"], error["data"]) == (
        "precondition_failed",
        "run_mismatch",
        {"expected": "absent", "actual": None},
    )


def test_rpc_pause_resume_retains_token_lease_and_reads_live_devices(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    rpc: RpcRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, harness, entry = attachment
    attempts: list[float] = []
    receipts: list[dict[str, object]] = []

    def init(_env: InitEnv[Calibrated], _plan: T1Plan) -> None:
        return None

    @workflow("rpc-pause", plan=T1Plan, tunables=T1Tunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated],
        plan: T1Plan,
        tun: T1Tunables,
        state: None,
    ) -> Step[None, int]:
        del plan, tun, state
        attempts.append(env.context.pi_length_us)
        env.pbar("rpc-work", 2).set_progress(len(attempts))
        if len(attempts) == 1:
            receipts.append(rpc("workflow.pause", {"run_id": "rpc-run"}))
            assert harness.gate.snapshot()
            yield from env.wait_until(env.started_at)
        return Done()

    monkeypatch.setattr(window_module, "t1_fluxdep", step)
    token = window.start("rpc-run")
    assert isinstance(token, int)
    harness.complete_segment()
    assert receipts[0]["ok"] is True
    requested = receipts[0]["result"]
    assert isinstance(requested, dict) and isinstance(requested["status"], dict)
    assert (
        requested["operation_id"] == token
        and requested["status"]["lifecycle"] == "pausing"
    )
    paused = rpc("workflow.status", {})["result"]
    assert isinstance(paused, dict) and isinstance(paused["status"], dict)
    assert paused["run_id"] == "rpc-run" and paused["operation_id"] == token
    assert paused["status"]["lifecycle"] == "paused"
    assert window.controller is not None
    native = window.controller.status()
    assert native is not None
    assert paused["status"] == json.loads(json.dumps(asdict(native)))
    assert paused["status"]["progress"][0]["name"] == "rpc-work"
    lease = tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot())
    bars = harness.progress.bars_for_operation(token)
    assert lease and bars and harness.handles.live_count() == 1
    assert harness.handles.known_outcome(token) is None

    entry.new_point("later-rpc")
    harness.services.context.use_point(PointUseRequest("later-rpc"))
    harness.services.context.set_component_focus(None)
    captures: list[bool] = []

    def capture() -> dict[str, DeviceInfo]:
        captures.append(bool(harness.gate.snapshot()))
        return {"flux": FakeDeviceInfo(address="none", value=0.34)}

    monkeypatch.setattr(
        harness.services.device, "capture_live_native_device_info", capture
    )
    resume_reply = rpc("workflow.resume", {"run_id": "rpc-run"})
    assert resume_reply["ok"] is True
    resumed = resume_reply["result"]
    assert isinstance(resumed, dict)
    assert resumed["run_id"] == "rpc-run"
    assert resumed["operation_id"] == token
    assert captures == [True] and not window.plan_panel.isEnabled()
    assert (
        tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot()) == lease
    )
    assert harness.progress.bars_for_operation(token) == bars
    assert harness.handles.live_count() == 1
    harness.complete_segment()
    terminal = rpc("workflow.status", {})["result"]
    assert isinstance(terminal, dict) and isinstance(terminal["status"], dict)
    assert (
        terminal["operation_id"] is None
        and terminal["status"]["lifecycle"] == "done"
        and terminal["status"]["run_id"] == "rpc-run"
    )
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    journal = entry.result_path / "workflows/rpc-run/journal.jsonl"
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    resumed_event = next(event for event in events if event["kind"] == "resumed")
    assert resumed_event["devices"]["items"]["flux"]["value"] == 0.34
    assert harness.background.error is None
    assert harness.handles.known_outcome(token) is not None
    assert not harness.gate.snapshot() and window.plan_panel.isEnabled()


@pytest.mark.parametrize("phase", ["pending", "paused"])
def test_rpc_old_target_does_not_change_current_run(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    rpc: RpcRequest,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    window, harness, _entry = attachment
    attempts: list[str] = []

    def init(_env: InitEnv[Calibrated], _plan: T1Plan) -> None:
        return None

    @workflow("rpc-target", plan=T1Plan, tunables=T1Tunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated],
        plan: T1Plan,
        tun: T1Tunables,
        state: None,
    ) -> Step[None, int]:
        del plan, tun, state
        attempts.append("step")
        window.pause("current")
        yield from env.wait_until(env.started_at)
        return Done()

    monkeypatch.setattr(window_module, "t1_fluxdep", step)
    assert isinstance(window.start("old"), int)
    window.stop("old")
    harness.complete_segment()
    assert isinstance(window.start("current"), int)
    if phase == "paused":
        harness.complete_segment()
    before = rpc("workflow.status", {})["result"]
    gate = tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot())
    captures: list[str] = []
    original = harness.services.device.capture_live_native_device_info

    def capture() -> dict[str, DeviceInfo]:
        captures.append("devices")
        return original()

    monkeypatch.setattr(
        harness.services.device, "capture_live_native_device_info", capture
    )
    for method in ("workflow.pause", "workflow.resume", "workflow.stop"):
        reply = rpc(method, {"run_id": "old"})
        error = reply["error"]
        assert reply["ok"] is False and isinstance(error, dict)
        assert (error["code"], error["reason"], error["data"]) == (
            "precondition_failed",
            "run_mismatch",
            {"expected": "old", "actual": "current"},
        )
        assert rpc("workflow.status", {})["result"] == before
        assert (
            tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot())
            == gate
        )
    assert captures == [] and attempts == ([] if phase == "pending" else ["step"])
    stopped = rpc("workflow.stop", {"run_id": "current"})["result"]
    assert isinstance(stopped, dict)
    if phase == "pending":
        assert stopped["status"] is None
        harness.complete_segment()
    else:
        assert (
            isinstance(stopped["status"], dict)
            and stopped["status"]["lifecycle"] == "stopped"
        )
        assert harness.background.work is None
        rejected = rpc("workflow.resume", {"run_id": "current"})
        error = rejected["error"]
        assert rejected["ok"] is False and isinstance(error, dict)
        assert (error["code"], error["reason"], error["data"]) == (
            "precondition_failed",
            "invalid_run_state",
            {"action": "resume", "lifecycle": "stopped"},
        )
    assert not harness.gate.snapshot() and window.plan_panel.isEnabled()


@pytest.fixture
def paused_tuning(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[int, float, int]]:
    """One committed step, followed by an owner-delivered paused step."""
    window, harness, _entry = run_attachment
    attempts: list[tuple[int, float, int]] = []

    def init(_env: InitEnv[Calibrated], _plan: T1Plan) -> int:
        return 0

    @workflow("tuning-paused", plan=T1Plan, tunables=T1Tunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated], plan: T1Plan, tun: T1Tunables, state: int
    ) -> Step[int, float]:
        del plan
        if state == 2:
            return Done()
        attempts.append((state, tun.guess_us, tun.reps))
        if state == 1 and len(attempts) == 2:
            window.pause("tuning")
            yield from env.wait_until(env.started_at)
        return Next(tun.guess_us, state + 1)

    monkeypatch.setattr(window_module, "t1_fluxdep", step)
    token = window.start("tuning")
    assert isinstance(token, int)
    harness.complete_segment()
    status = window.controller.status()
    assert status is not None and status.lifecycle == "paused"
    assert window.tuning_panel is not None
    assert window.controller.operation_id == token
    assert not window.controller.has_pending_segment
    assert harness.gate.snapshot() and harness.handles.live_count() == 1
    assert harness.handles.known_outcome(token) is None
    return attempts


def test_rpc_tunables_running_submits_shared_draft_but_keeps_current_step_capture(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    rpc: RpcRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, harness, entry = attachment
    observed: list[tuple[float, int]] = []
    replies: list[dict[str, object]] = []

    def init(_env: InitEnv[Calibrated], _plan: T1Plan) -> int:
        return 0

    @workflow("tuning-running", plan=T1Plan, tunables=T1Tunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated], plan: T1Plan, tun: T1Tunables, state: int
    ) -> Step[int, float]:
        yield from ()
        del env, plan
        if state == 2:
            return Done()
        if state == 0:
            # Public edit installs the accepted panel even before timer delivery.
            replies.append(rpc("workflow.tunables", {"run_id": "running", "batch": []}))
            panel = window.tuning_panel
            assert panel is not None
            panel.panel.state.write({("guess_us",): {"expr": "10 * 3"}}, FormEnv())
            control = panel.panel.editor.findChild(QLineEdit, "form.input.sweep_factor")
            assert isinstance(control, QLineEdit)
            control.setText("-")
            replies.append(
                rpc(
                    "workflow.tunables",
                    {"run_id": "running", "batch": [{"path": ["reps"], "value": 200}]},
                )
            )
            assert panel.panel.editor.local_parse_errors() is None
            assert panel.host.accepted_input.data["guess_us"] == {"expr": "10 * 3"}
            assert panel.panel.state.input.data == panel.panel.state.base.data
        observed.append((tun.guess_us, tun.reps))
        return Next(tun.guess_us, state + 1)

    monkeypatch.setattr(window_module, "t1_fluxdep", step)
    assert isinstance(window.start("running"), int)
    harness.complete_segment()
    assert harness.background.error is None
    assert [reply["ok"] for reply in replies] == [True, True]
    for reply, revision in zip(replies, (1, 2), strict=True):
        result = reply["result"]
        assert isinstance(result, dict) and isinstance(result["status"], dict)
        assert (
            result["run_id"] == "running" and result["status"]["revision"] == revision
        )
    assert observed[0][0] == 20.0 and observed[1] == (30.0, 200)
    controller = window.controller
    assert controller is not None and controller.engine is not None
    records = controller.engine.records("running")
    assert [r.revision for r in records] == [0, 2]
    assert [r.encoded_record.value for r in records] == [20.0, 30.0]
    events = [
        json.loads(line)
        for line in (entry.result_path / "workflows/running/journal.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    updates = [e for e in events if e["kind"] == "tunables_changed"]
    assert len(updates) == 2
    assert all(e["actor"] == {"kind": "agent", "name": "rpc"} for e in updates)
    assert not harness.gate.snapshot()


@pytest.mark.parametrize("editor", ["gui", "rpc"])
def test_paused_tunables_apply_then_resume_repeats_uncommitted_step(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    paused_tuning: list[tuple[int, float, int]],
    rpc: RpcRequest,
    monkeypatch: pytest.MonkeyPatch,
    editor: str,
) -> None:
    window, harness, entry = attachment
    panel, controller = window.tuning_panel, window.controller
    assert (
        panel is not None and controller is not None and controller.engine is not None
    )
    engine = controller.engine
    token = controller.operation_id
    assert token is not None
    lease = tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot())
    assert lease and harness.handles.known_outcome(token) is None
    prior = engine.records("tuning")
    assert len(prior) == 1 and prior[0].revision == 0
    panel.panel.state.write(
        {("guess_us",): {"expr": "general.ext.guess_us * 2"}}, FormEnv()
    )
    # External Apply resolves the accepted capture, not later live parameters.
    context = harness.services.context
    context.set_parameter(context.read_context().target, "general.ext.guess_us", 99.0)
    if editor == "gui":
        applied = panel.apply()
        assert not isinstance(applied, FormErrors) and applied.revision == 1
    else:
        reply = rpc("workflow.tunables", {"run_id": "tuning", "batch": []})
        assert reply["ok"] is True
        result = reply["result"]
        assert isinstance(result, dict) and isinstance(result["status"], dict)
        assert result["status"]["revision"] == 1
    assert engine.records("tuning") == prior
    assert panel.host.accepted_input.data["guess_us"] == {
        "expr": "general.ext.guess_us * 2"
    }
    assert panel.panel.state.base.data == panel.panel.state.input.data
    # Resume also retains the original point/focus.
    entry.new_point("later-tuning")
    context.use_point(PointUseRequest("later-tuning"))
    captures: list[bool] = []

    def capture() -> dict[str, DeviceInfo]:
        captures.append(bool(harness.gate.snapshot()))
        return {"flux": FakeDeviceInfo(address="none", value=0.56)}

    monkeypatch.setattr(
        harness.services.device, "capture_live_native_device_info", capture
    )
    reply = rpc("workflow.resume", {"run_id": "tuning"})
    assert reply["ok"] is True and captures == [True]
    result = reply["result"]
    assert isinstance(result, dict) and result["operation_id"] == token
    assert controller.operation_id == token and harness.handles.live_count() == 1
    assert (
        tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot()) == lease
    )
    harness.complete_segment()
    assert harness.background.error is None
    assert paused_tuning[0][0] == 0
    assert paused_tuning[1][0:2] == (1, 20.0)
    assert paused_tuning[2][0:2] == (1, 40.0)
    assert controller.engine is engine
    records = engine.records("tuning")
    assert records[0] == prior[0] and records[1].revision == 1
    assert records[1].encoded_record.value == 40.0
    status = controller.status()
    assert status is not None and status.lifecycle == "done"
    journal = entry.result_path / "workflows/tuning/journal.jsonl"
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    resumed_event = next(event for event in events if event["kind"] == "resumed")
    assert resumed_event["devices"]["items"]["flux"]["value"] == 0.56
    assert harness.handles.known_outcome(token) is not None
    assert not harness.gate.snapshot()


@pytest.mark.parametrize("value", [-1.0, {"expr": "missing_source * 2"}])
def test_rpc_tunables_invalid_input_retains_draft_without_publishing(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    paused_tuning: list[tuple[int, float, int]],
    rpc: RpcRequest,
    value: JsonValue,
) -> None:
    del paused_tuning
    window, _harness, _entry = attachment
    panel, controller = window.tuning_panel, window.controller
    assert (
        panel is not None and controller is not None and controller.engine is not None
    )
    before = controller.engine.tunables("tuning")
    accepted = panel.host.accepted_input.data
    reply = rpc(
        "workflow.tunables",
        {
            "run_id": "tuning",
            "batch": [
                {"path": ["reps"], "value": 200},
                {"path": ["guess_us"], "value": value},
            ],
        },
    )
    error = reply["error"]
    assert reply["ok"] is False and isinstance(error, dict)
    assert (error["code"], error["reason"]) == ("invalid_params", "form_errors")
    data = error["data"]
    assert isinstance(data, dict) and isinstance(data["messages"], list)
    assert any(m["path"] == ["guess_us"] for m in data["messages"])
    assert panel.panel.state.input.data["guess_us"] == value
    assert panel.panel.state.input.data["reps"] == 200
    assert panel.host.accepted_input.data == panel.panel.state.base.data == accepted
    assert controller.engine.tunables("tuning") == before
    panel.discard()
    assert panel.panel.state.input.data == accepted


@pytest.mark.parametrize(
    "batch",
    [
        {},
        [{"path": [], "value": 200}],
        [{"path": ["reps"], "value": 200, "extra": 1}],
        [{"path": ["reps"], "value": 200}, {"path": ["reps"], "value": 300}],
        [{"path": ["calibration", 0], "value": 200}],
        [{"path": ["unknown"], "value": 200}],
        [
            {"path": ["calibration"], "value": {}},
            {"path": ["calibration", "reps"], "value": 200},
        ],
        [{"path": ["calibration", "reps", "0"], "value": 200}],
    ],
)
def test_rpc_tunables_rejects_wire_and_structural_batches_before_draft_write(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    paused_tuning: list[tuple[int, float, int]],
    rpc: RpcRequest,
    batch: object,
) -> None:
    del paused_tuning
    window, harness, _entry = attachment
    panel, controller = window.tuning_panel, window.controller
    assert (
        panel is not None and controller is not None and controller.engine is not None
    )
    before = (
        panel.panel.state.input.data,
        panel.panel.state.base.data,
        panel.host.accepted_input.data,
        controller.engine.tunables("tuning"),
    )
    token = controller.operation_id
    assert token is not None
    lease = tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot())
    assert lease and harness.handles.known_outcome(token) is None
    reply = rpc("workflow.tunables", {"run_id": "tuning", "batch": batch})
    error = reply["error"]
    assert reply["ok"] is False and isinstance(error, dict)
    assert error["code"] in {"invalid_params", "controller_error"}
    assert (
        panel.panel.state.input.data,
        panel.panel.state.base.data,
        panel.host.accepted_input.data,
        controller.engine.tunables("tuning"),
    ) == before
    assert controller.operation_id == token and harness.handles.live_count() == 1
    assert (
        tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot()) == lease
    )
    assert harness.handles.known_outcome(token) is None


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"run_id": "tuning"},
        {"run_id": "", "batch": []},
        {"run_id": None, "batch": []},
        {"run_id": 3, "batch": []},
    ],
)
def test_rpc_tunables_requires_explicit_target_and_batch(
    rpc: RpcRequest, params: dict[str, object]
) -> None:
    with pytest.raises(RemoteError) as raised:
        rpc("workflow.tunables", params)
    assert raised.value.code is ErrorCode.INVALID_PARAMS


def test_rpc_tunables_absent_and_pending_target_cannot_edit_start_draft(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    rpc: RpcRequest,
) -> None:
    window, harness, entry = attachment
    before = window.tunables_panel.state.input.data
    for actual in (None, "pending"):
        reply = rpc(
            "workflow.tunables",
            {"run_id": "old", "batch": [{"path": ["guess_us"], "value": 80.0}]},
        )
        error = reply["error"]
        assert reply["ok"] is False and isinstance(error, dict)
        assert (error["reason"], error["data"]) == (
            "run_mismatch",
            {"expected": "old", "actual": actual},
        )
        assert window.tunables_panel.state.input.data == before
        assert window.tuning_panel is None
        if actual is None:
            assert isinstance(window.start("pending"), int)
            before = window.tunables_panel.state.input.data
    with pytest.raises(RuntimeError, match="accepted"):
        window.apply_tunables("pending", {}, actor=Actor("agent", "rpc"))
    reply = rpc("workflow.tunables", {"run_id": "pending", "batch": []})
    error = reply["error"]
    assert reply["ok"] is False and isinstance(error, dict)
    assert error["code"] == "controller_error" and "accepted" in error["message"]
    assert (
        window.tunables_panel.state.input.data == before and window.tuning_panel is None
    )
    assert not (entry.result_path / "workflows/pending").exists()
    assert harness.gate.snapshot()
    window.stop("pending")
    harness.complete_segment()


def test_rpc_tunables_old_target_preserves_accepted_panel_and_native_run(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    paused_tuning: list[tuple[int, float, int]],
    rpc: RpcRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del paused_tuning
    window, harness, _entry = attachment
    panel, controller = window.tuning_panel, window.controller
    assert (
        panel is not None and controller is not None and controller.engine is not None
    )
    token = controller.operation_id
    assert token is not None
    lease = tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot())
    assert lease and harness.handles.known_outcome(token) is None
    panel.panel.state.write({("guess_us",): 35.0}, FormEnv())
    before = (
        panel.panel.state.input.data,
        panel.panel.state.base.data,
        panel.host.accepted_input.data,
        controller.engine.tunables("tuning"),
        controller.status(),
        controller.operation_id,
    )
    captures: list[str] = []
    original = harness.services.device.capture_live_native_device_info

    def capture() -> dict[str, DeviceInfo]:
        captures.append("devices")
        return original()

    monkeypatch.setattr(
        harness.services.device, "capture_live_native_device_info", capture
    )
    with pytest.raises(RunMismatch):
        window.apply_tunables("old", {("reps",): 200}, actor=Actor("agent", "rpc"))
    reply = rpc(
        "workflow.tunables",
        {"run_id": "old", "batch": [{"path": ["reps"], "value": 200}]},
    )
    error = reply["error"]
    assert reply["ok"] is False and isinstance(error, dict)
    assert error["data"] == {"expected": "old", "actual": "tuning"}
    assert (
        panel.panel.state.input.data,
        panel.panel.state.base.data,
        panel.host.accepted_input.data,
        controller.engine.tunables("tuning"),
        controller.status(),
        controller.operation_id,
    ) == before
    assert captures == [] and controller.operation_id == token
    assert (
        tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot()) == lease
    )
    assert harness.handles.live_count() == 1
    assert harness.handles.known_outcome(token) is None


@pytest.mark.parametrize("failure", ["terminal", "journal"])
def test_rpc_tunables_native_failure_keeps_base_revision_and_local_text(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    paused_tuning: list[tuple[int, float, int]],
    rpc: RpcRequest,
    failure: str,
) -> None:
    del paused_tuning
    window, harness, entry = attachment
    panel, controller = window.tuning_panel, window.controller
    assert (
        panel is not None and controller is not None and controller.engine is not None
    )
    before = controller.engine.tunables("tuning")
    accepted = panel.host.accepted_input.data
    control = panel.panel.editor.findChild(QLineEdit, "form.input.sweep_factor")
    assert isinstance(control, QLineEdit)
    control.setText("-")
    journal = entry.result_path / "workflows/tuning/journal.jsonl"
    original_journal = journal.read_bytes()
    if failure == "terminal":
        window.stop("tuning")
    else:
        journal.unlink()
        journal.mkdir()
    try:
        reply = rpc(
            "workflow.tunables",
            {"run_id": "tuning", "batch": [{"path": ["guess_us"], "value": 45.0}]},
        )
        error = reply["error"]
        assert reply["ok"] is False and isinstance(error, dict)
        if failure == "terminal":
            assert (error["code"], error["reason"]) == (
                "precondition_failed",
                "invalid_run_state",
            )
            data = error["data"]
            assert isinstance(data, dict) and data["lifecycle"] == "stopped"
        else:
            assert error["code"] == "controller_error"
            assert "Is a directory" in error["message"]
    finally:
        if failure == "journal":
            # Restore the fault injection even when the request/test fails.
            journal.rmdir()
            journal.write_bytes(original_journal)
    assert controller.engine.tunables("tuning") == before
    assert panel.host.accepted_input.data == panel.panel.state.base.data == accepted
    assert panel.panel.state.input.data["guess_us"] == 45.0
    assert panel.panel.editor.local_parse_errors() is not None
    assert control.text() == "-" and not harness.gate.snapshot()
    assert window.plan_panel.isEnabled() and window.tunables_panel.isEnabled()
    start_button = window.findChild(QPushButton, "workflow.run.start")
    assert start_button is not None and start_button.isEnabled()
    assert not harness.progress.bars_for_owner("t1-workflow")
    # A failed idle update must not leave unfinished presentation behind.
    assert isinstance(window.start("after-failure"), int)
    harness.complete_segment()
    assert harness.background.error is None
    window.close()
    window.close()
    assert not harness.progress.bars_for_owner("t1-workflow")


def test_apply_click_isolates_native_journal_failure_without_accepting_draft(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    paused_tuning: list[tuple[int, float, int]],
    escaped_qt_errors: list[BaseException],
    settlements: list[int],
    caplog: pytest.LogCaptureFixture,
) -> None:
    del paused_tuning
    window, harness, entry = run_attachment
    controller, panel = window.controller, window.tuning_panel
    assert panel is not None and controller.engine is not None
    token = controller.operation_id
    assert token is not None
    native = controller.engine.tunables("tuning")
    accepted = panel.host.accepted_input.data
    panel.panel.state.write({("guess_us",): 45.0}, panel.host.env)
    control = panel.panel.editor.findChild(QLineEdit, "form.input.sweep_factor")
    button = panel.findChild(QPushButton, "workflow.tuning.apply")
    message = panel.findChild(QLabel, "workflow.tuning.message")
    assert isinstance(control, QLineEdit) and isinstance(button, QPushButton)
    assert isinstance(message, QLabel)
    control.setText("8.00")
    before_click_text = control.text()
    journal = entry.result_path / "workflows/tuning/journal.jsonl"
    original = journal.read_bytes()
    journal.unlink()
    journal.mkdir()
    try:
        button.click()
    finally:
        journal.rmdir()
        journal.write_bytes(original)

    assert escaped_qt_errors == []
    status = controller.status()
    assert status is not None and status.lifecycle == "failed"
    outcome = harness.handles.known_outcome(token)
    assert outcome is not None and outcome.status == "failed"
    assert outcome.error is not None and str(journal) in outcome.error
    assert str(journal) in message.text() and str(journal) in caplog.text
    assert any(
        record.name == "zcu_lab.workflows.gui.tuning_panel"
        and record.exc_info is not None
        and isinstance(record.exc_info[1], IsADirectoryError)
        and record.exc_info[1].filename == str(journal)
        for record in caplog.records
    )
    assert controller.engine.tunables("tuning") == native
    assert panel.host.accepted_input.data == panel.panel.state.base.data == accepted
    assert panel.panel.state.input.data["guess_us"] == 45.0
    assert panel.panel.state.input.data["sweep_factor"] == 8.0
    assert control.text() == before_click_text
    assert settlements == [token]
    assert harness.handles.live_count() == 0 and not harness.gate.snapshot()
    assert not harness.progress.bars_for_owner("t1-workflow")
    assert window.plan_panel.isEnabled() and not controller.has_pending_segment
    window.stop("tuning")
    assert window.close() and window.close()
    assert settlements == [token] and escaped_qt_errors == []


@pytest.mark.parametrize("phase", ["paused", "active"])
def test_close_isolates_stop_journal_failure_and_retains_pending_canvas(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
    escaped_qt_errors: list[BaseException],
    settlements: list[int],
    caplog: pytest.LogCaptureFixture,
    phase: str,
) -> None:
    window, harness, entry = run_attachment

    @workflow(
        "close-fault", plan=T1Plan, tunables=T1Tunables, init=lambda _env, _plan: None
    )
    def step(
        env: WorkflowEnv[Calibrated], plan: T1Plan, tun: T1Tunables, state: None
    ) -> Step[None, int]:
        del plan, tun, state
        env.plots.subplots("live")

        def draw(figure: Figure) -> None:
            figure.clear()
            figure.subplots().plot([0, 1], [0, 1])

        env.plots.draw("live", draw)
        env.pbar("work", 2).refresh()
        window.pause("close-fault")
        yield from env.wait_until(env.started_at)
        return Done()

    monkeypatch.setattr(window_module, "t1_fluxdep", step)
    token = window.start("close-fault")
    assert isinstance(token, int)
    harness.complete_segment()
    controller = window.controller
    panel = window.tuning_panel
    stack = next(
        child
        for child in window.findChildren(QStackedWidget)
        if child.parent() is window
    )
    message = window.findChild(QLabel, "workflow.run.message")
    assert isinstance(stack, QStackedWidget) and isinstance(message, QLabel)
    assert panel is not None and stack.count() == 2
    canvas = stack.currentWidget()
    if phase == "active":
        assert window.resume("close-fault") == token
    journal = entry.result_path / "workflows/close-fault/journal.jsonl"
    original = journal.read_bytes()
    journal.unlink()
    journal.mkdir()
    try:
        # Keep this close rejected so the user can see the native diagnostic.
        closed = window.close()
        assert escaped_qt_errors == []
        assert not closed
        status = controller.status()
        assert status is not None and status.lifecycle == "failed"
        assert str(journal) in message.text() and str(journal) in caplog.text
        assert any(
            record.name == "zcu_lab.workflows.gui.window"
            and record.exc_info is not None
            and isinstance(record.exc_info[1], IsADirectoryError)
            and record.exc_info[1].filename == str(journal)
            for record in caplog.records
        )
        if phase == "active":
            assert controller.has_pending_segment and controller.binding is panel.host
            assert stack.count() == 2 and stack.currentWidget() is canvas
            assert not window.plan_panel.isEnabled()
            assert harness.background.work is not None and harness.gate.snapshot()
            assert harness.handles.live_count() == 1
            assert harness.handles.known_outcome(token) is None and settlements == []
        else:
            assert not controller.has_pending_segment
            assert harness.background.work is None
            assert window.plan_panel.isEnabled() and settlements == [token]
    finally:
        journal.rmdir()
        journal.write_bytes(original)

    if phase == "active":
        harness.background.execute()
        assert harness.gate.snapshot() and settlements == []
        assert stack.count() == 2 and stack.currentWidget() is canvas
        harness.background.deliver()
    outcome = harness.handles.known_outcome(token)
    assert outcome is not None and outcome.status == "failed"
    assert outcome.error is not None and str(journal) in outcome.error
    assert settlements == [token]
    assert harness.handles.live_count() == 0 and not harness.gate.snapshot()
    assert not harness.progress.bars_for_owner("t1-workflow")
    assert not controller.has_pending_segment
    assert window.close() and window.close()
    assert stack.count() == 1
    assert settlements == [token] and escaped_qt_errors == []


def test_window_tunables_journal_failure_reraises_the_original_cause(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    paused_tuning: list[tuple[int, float, int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del paused_tuning
    window, harness, entry = attachment
    panel = window.tuning_panel
    assert panel is not None
    native_errors: list[Exception] = []
    original_apply = panel.apply_external

    def observe_native_error(
        batch: Mapping[FormWritePath, JsonValue],
        *,
        actor: Actor,
    ) -> TunablesSnapshot | FormErrors:
        try:
            return original_apply(batch, actor=actor)
        except Exception as error:
            native_errors.append(error)
            raise

    # Observe the borrowed public panel boundary without replacing its behavior.
    monkeypatch.setattr(panel, "apply_external", observe_native_error)
    journal = entry.result_path / "workflows/tuning/journal.jsonl"
    original = journal.read_bytes()
    journal.unlink()
    journal.mkdir()
    try:
        with pytest.raises(IsADirectoryError) as caught:
            window.apply_tunables(
                "tuning", {("guess_us",): 45.0}, actor=Actor("agent", "rpc")
            )
    finally:
        journal.rmdir()
        journal.write_bytes(original)
    assert len(native_errors) == 1 and caught.value is native_errors[0]
    assert window.controller is not None
    status = window.controller.status()
    assert status is not None and status.lifecycle == "failed"
    assert not harness.progress.bars_for_owner("t1-workflow")
    window.stop("tuning")
    window.close()
    window.close()


def _start_plan_batch(_inputs: Inputs) -> list[FormWrite]:
    """Select the fixture's native modules through captured public library roots.

    Waveform's reference/inline union has no writable discriminator child path;
    reference selections use existing supported leaves without a tree encoder.
    Omit points so Start must use defaults rather than the old GUI draft.
    """
    return [
        {"path": ["flux_dev"], "value": "flux"},
        {"path": ["fluxes", "kind"], "value": "list"},
        {"path": ["fluxes", "values"], "value": [1.0, 1.1]},
        {"path": ["pi_length_us"], "value": {"expr": "0.06 * 2"}},
        {"path": ["qub_pulse", "kind"], "value": "ref"},
        {"path": ["qub_pulse", "name"], "value": "Q1.pulses.pi"},
        {"path": ["readout", "kind"], "value": "ref"},
        {"path": ["readout", "name"], "value": "Q1.readout.ro"},
    ]


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
            "plan": _start_plan_batch(inputs),
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
    plan = _start_plan_batch(inputs)
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
        {"run_id": "clean-rpc", "plan": _start_plan_batch(inputs), "tunables": []},
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
                "plan": _start_plan_batch(inputs),
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
        {"run_id": "failed-rpc", "plan": _start_plan_batch(inputs), "tunables": []},
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
        {"run_id": "empty-source", "plan": _start_plan_batch(inputs), "tunables": []},
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
        {"run_id": "replacement", "plan": _start_plan_batch(inputs), "tunables": []},
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


def test_rpc_running_stop_retains_lease_until_owner_delivery(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    rpc: RpcRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, harness, _entry = attachment
    receipts: list[dict[str, object]] = []

    def init(_env: InitEnv[Calibrated], _plan: T1Plan) -> None:
        return None

    @workflow("rpc-stop", plan=T1Plan, tunables=T1Tunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated],
        plan: T1Plan,
        tun: T1Tunables,
        state: None,
    ) -> Step[None, int]:
        del plan, tun, state
        receipts.append(rpc("workflow.stop", {"run_id": "stop-rpc"}))
        assert harness.gate.snapshot()
        yield from env.wait_until(env.started_at)
        return Done()

    monkeypatch.setattr(window_module, "t1_fluxdep", step)
    assert isinstance(window.start("stop-rpc"), int)
    harness.background.execute()
    assert harness.gate.snapshot() and not window.plan_panel.isEnabled()
    assert receipts[0]["ok"] is True
    requested = receipts[0]["result"]
    assert isinstance(requested, dict) and isinstance(requested["status"], dict)
    assert requested["status"]["lifecycle"] == "stopping"
    harness.owner.pump_all()
    harness.background.deliver()
    terminal = rpc("workflow.status", {})["result"]
    assert isinstance(terminal, dict) and isinstance(terminal["status"], dict)
    assert (
        terminal["status"]["lifecycle"] == "stopped"
        and terminal["operation_id"] is None
    )
    assert not harness.gate.snapshot() and window.plan_panel.isEnabled()
