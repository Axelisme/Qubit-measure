"""Concrete Qt startup uses one fixed native capture and no hardware."""

import json
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from time import monotonic

import pytest
from matplotlib.figure import Figure
from pydantic import JsonValue
from qtpy.QtWidgets import QApplication, QLabel, QLineEdit, QPushButton
from zcu_tools.datafile import RunSnapshot
from zcu_tools.device import DeviceInfo
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
from zcu_tools.gui.remote.wire import Request
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.events import GateChangedPayload
from zcu_tools.gui.session.services.device import ConnectDeviceRequest
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


def test_start_live_read_failure_names_device_before_admission(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, harness, entry = attachment
    driver = harness.services.device.get_connected_devices()["flux"]
    before = window.plan_panel.state.input

    def reject_read() -> DeviceInfo:
        raise OSError("instrument read failed")

    monkeypatch.setattr(driver, "get_info", reject_read)
    with pytest.raises(RuntimeError, match="flux"):
        window.start("live-read-failure")

    assert window.plan_panel.state.input == before
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
    assert window.controller is None and harness.background.work is None
    assert not (entry.result_path / "workflows").exists()
    assert not (entry.database_path / "workflows").exists()


def test_startup_source_failure_is_not_replaced_by_old_constructor_capture(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
) -> None:
    window, harness, entry = attachment
    harness.services.context.clear_context()
    with pytest.raises(FailedPreconditionError):
        window.start("empty")
    assert window.controller is None and harness.background.work is None
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
    assert window.controller is None and harness.handles.live_count() == 0
    assert not harness.gate.snapshot()
    assert not (entry.result_path / "workflows").exists()
    assert not (entry.database_path / "workflows").exists()


def test_close_pending_start_requests_stop_without_destroying_producer_host(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
) -> None:
    window, harness, entry = attachment
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
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    previous_start: bool,
    notification: str,
) -> None:
    window, harness, entry = attachment
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
        window.stop("previous")
        harness.background.execute()
        harness.background.deliver()
        assert not harness.gate.snapshot()

    close_results: list[bool] = []

    def cancel_on_notification() -> None:
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
    if action == "close":
        assert close_results == [False]
    controller = window.controller
    assert controller is not None and controller.operation_id == token
    assert controller.status() is None and harness.gate.snapshot()
    harness.background.execute()
    assert calls == []
    assert controller.status() is None and harness.gate.snapshot()
    harness.background.deliver()
    assert controller.operation_id is None and harness.handles.live_count() == 0
    assert not harness.gate.snapshot()
    assert not (entry.result_path / "workflows").exists()
    assert not (entry.database_path / "workflows").exists()
    assert window.close()


@pytest.mark.parametrize("previous_start", [False, True])
def test_gate_registration_reentrant_start_preserves_outer_controller(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
    previous_start: bool,
) -> None:
    window, harness, entry = attachment
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
        if outer is None:
            return
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
    assert observed == [("outer", None, True)]
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
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, harness, entry = attachment
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

    # Resume must not capture this later point/focus as new run inputs.
    entry.new_point("later")
    harness.services.context.use_point(PointUseRequest("later"))
    harness.services.context.set_component_focus(None)
    captures: list[str] = []
    capture_native = harness.services.device.capture_native_device_info

    def capture() -> dict[str, DeviceInfo]:
        assert harness.gate.snapshot()
        captures.append("cached devices")
        return capture_native()

    monkeypatch.setattr(harness.services.device, "capture_native_device_info", capture)
    resume_button = window.findChild(QPushButton, "workflow.run.resume")
    assert resume_button is not None and resume_button.isEnabled()
    resume_button.click()
    resumed = controller.operation_id
    assert resumed is not None and resumed != token and captures == ["cached devices"]
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
    assert not harness.gate.snapshot()


@pytest.mark.parametrize("action", ["stop", "close"])
def test_resume_register_stop_or_close_keeps_window_locked_until_delivery(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    window, harness, _entry = attachment
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

    def stop_on_register(payload: GateChangedPayload) -> None:
        if payload.active:
            if action == "close":
                close_results.append(window.close())
            else:
                window.stop("register-window")
            observations.append(
                (window.plan_panel.isEnabled(), active_controller.has_pending_segment)
            )

    subscription = harness.bus.subscribe(GateChangedPayload, stop_on_register)
    try:
        window.resume("register-window")
    finally:
        subscription.unsubscribe()
    assert observations == [(False, True)]
    assert not active_controller.has_pending_segment
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
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    window, harness, _entry = attachment
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
    capture_native = harness.services.device.capture_native_device_info
    capture_error = ValueError("cached snapshot failed after Stop")

    def capture() -> dict[str, DeviceInfo]:
        window.stop("reject-window")
        if failure == "capture":
            raise capture_error
        return capture_native()

    monkeypatch.setattr(harness.services.device, "capture_native_device_info", capture)
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
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    window, harness, entry = attachment

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
    assert window.controller is None and harness.background.work is None
    assert harness.handles.live_count() == 0 and not harness.gate.snapshot()


@pytest.mark.parametrize("phase", ["pending", "paused"])
def test_old_window_commands_cannot_change_replacement_run_and_gui_stops_current(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    window, harness, entry = attachment
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
    original_capture = harness.services.device.capture_native_device_info

    def capture() -> dict[str, DeviceInfo]:
        captures.append("devices")
        return original_capture()

    monkeypatch.setattr(harness.services.device, "capture_native_device_info", capture)
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


@pytest.fixture
def rpc(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    inputs: Inputs,
) -> Iterator[RpcRequest]:
    """Use public route and encoded replies; pump the real owner queue explicitly."""
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

            def request(method: str, params: dict[str, object]) -> dict[str, object]:
                pending = io.submit(
                    service.route, link, Request("request", method, params)
                )
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


def test_rpc_pending_status_and_stop(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    rpc: RpcRequest,
) -> None:
    window, harness, entry = attachment
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


def test_rpc_pause_resume_uses_same_native_run_and_cached_capture(
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
    assert paused["run_id"] == "rpc-run" and paused["operation_id"] is None
    assert paused["status"]["lifecycle"] == "paused"
    assert window.controller is not None
    native = window.controller.status()
    assert native is not None
    assert paused["status"] == json.loads(json.dumps(asdict(native)))
    assert paused["status"]["progress"][0]["name"] == "rpc-work"
    assert not harness.gate.snapshot()

    entry.new_point("later-rpc")
    harness.services.context.use_point(PointUseRequest("later-rpc"))
    harness.services.context.set_component_focus(None)
    captures: list[bool] = []
    original = harness.services.device.capture_native_device_info

    def capture() -> dict[str, DeviceInfo]:
        captures.append(bool(harness.gate.snapshot()))
        return original()

    monkeypatch.setattr(harness.services.device, "capture_native_device_info", capture)
    resumed = rpc("workflow.resume", {"run_id": "rpc-run"})["result"]
    assert isinstance(resumed, dict)
    assert resumed["run_id"] == "rpc-run"
    assert isinstance(resumed["operation_id"], int) and resumed["operation_id"] != token
    assert captures == [True] and not window.plan_panel.isEnabled()
    harness.complete_segment()
    terminal = rpc("workflow.status", {})["result"]
    assert isinstance(terminal, dict) and isinstance(terminal["status"], dict)
    assert (
        terminal["operation_id"] is None
        and terminal["status"]["lifecycle"] == "done"
        and terminal["status"]["run_id"] == "rpc-run"
    )
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert harness.background.error is None
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
    original = harness.services.device.capture_native_device_info

    def capture() -> dict[str, DeviceInfo]:
        captures.append("devices")
        return original()

    monkeypatch.setattr(harness.services.device, "capture_native_device_info", capture)
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
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[int, float, int]]:
    """One committed step, followed by an owner-delivered paused step."""
    window, harness, _entry = attachment
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
    assert isinstance(window.start("tuning"), int)
    harness.complete_segment()
    assert window.controller is not None
    status = window.controller.status()
    assert status is not None and status.lifecycle == "paused"
    assert window.tuning_panel is not None
    assert not harness.gate.snapshot()
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
    original = harness.services.device.capture_native_device_info

    def capture() -> dict[str, DeviceInfo]:
        captures.append(bool(harness.gate.snapshot()))
        return original()

    monkeypatch.setattr(harness.services.device, "capture_native_device_info", capture)
    reply = rpc("workflow.resume", {"run_id": "tuning"})
    assert reply["ok"] is True and captures == [True]
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
    assert not harness.gate.snapshot()


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
    original = harness.services.device.capture_native_device_info

    def capture() -> dict[str, DeviceInfo]:
        captures.append("devices")
        return original()

    monkeypatch.setattr(harness.services.device, "capture_native_device_info", capture)
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
    assert captures == [] and not harness.gate.snapshot()


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
    assert window.controller is None and harness.background.work is None
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
    assert window.controller is None and harness.background.work is None
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
    assert window.controller is None and harness.handles.live_count() == 0
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
    assert window.controller is None and harness.background.work is None
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
