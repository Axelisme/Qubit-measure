"""Concrete Qt startup uses one fixed native capture and no hardware."""

from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from matplotlib.figure import Figure
from qtpy.QtWidgets import QApplication, QPushButton
from zcu_tools.datafile import RunSnapshot
from zcu_tools.device import DeviceInfo
from zcu_tools.experiment.workflows import (
    Done,
    InitEnv,
    InvalidRunState,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.form import FormEnv, FormErrors, FormInput
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
from zcu_lab.workflows.gui.tunables import T1TunablesForm
from zcu_lab.workflows.gui.window import T1WorkflowWindow
from zcu_lab.workflows.t1 import T1Plan, T1Tunables

from ._plan_inputs import inline_input


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
                window.stop()
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
    assert window.tunables_panel.state.input.data["guess_us"] == 77.0
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
        window.stop()
        harness.background.execute()
        harness.background.deliver()
        assert not harness.gate.snapshot()

    close_results: list[bool] = []

    def cancel_on_notification() -> None:
        if action == "close":
            close_results.append(window.close())
        else:
            window.stop()

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
            window.pause()
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
    resumed = window.resume()
    assert resumed != token
    assert captures == ["cached devices"]
    assert not window.plan_panel.isEnabled()
    harness.complete_segment()
    terminal = controller.status()
    assert terminal is not None and terminal.lifecycle == "done"
    assert harness.background.error is None
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
        window.pause()
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
                window.stop()
            observations.append(
                (window.plan_panel.isEnabled(), active_controller.has_pending_segment)
            )

    subscription = harness.bus.subscribe(GateChangedPayload, stop_on_register)
    try:
        window.resume()
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
        window.pause()
        yield from env.wait_until(env.started_at)
        return Done()

    monkeypatch.setattr(window_module, "t1_fluxdep", step)
    assert isinstance(window.start("reject-window"), int)
    harness.complete_segment()
    capture_native = harness.services.device.capture_native_device_info
    capture_error = ValueError("cached snapshot failed after Stop")

    def capture() -> dict[str, DeviceInfo]:
        window.stop()
        if failure == "capture":
            raise capture_error
        return capture_native()

    monkeypatch.setattr(harness.services.device, "capture_native_device_info", capture)
    harness.background.fail_submit = failure == "submit"
    with pytest.raises(ValueError if failure == "capture" else OSError) as raised:
        window.resume()
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
        window.pause()
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
        status = window.stop()
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
