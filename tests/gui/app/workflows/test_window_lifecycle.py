"""Window lifecycle contracts through public native/GUI seams."""

from collections.abc import Callable

import pytest
from qtpy.QtWidgets import QPushButton
from zcu_tools.experiment.workflows import (
    Actor,
    Done,
    InitEnv,
    InvalidRunState,
    RunMismatch,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.operation_handles import OperationOutcome
from zcu_tools.resources.entry import (
    ResultEntry,
)

from tests._filesystem import DIRECTORY_OPEN_ERROR
from tests.gui.app.workflows._start_fakes import StartHarness

from ._reload_source import InjectedCatalog
from ._sample_input import sample_input
from ._synthetic import Context, Plan, Tunables
from ._window_scenarios import install_pause_plot_workflow
from ._window_support import form_editable


def test_close_pending_start_requests_stop_without_destroying_producer_host(
    run_attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
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


def test_window_pause_keeps_plan_plots_and_capture_then_resumes_with_lease(
    run_attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    catalog_source: InjectedCatalog,
) -> None:
    window, harness, entry = run_attachment
    attempts, figures = install_pause_plot_workflow(window, catalog_source)
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
    assert not form_editable(window.plan_panel)
    harness.complete_segment()
    terminal = controller.status()
    assert (
        terminal is not None
        and terminal.lifecycle == "done"
        and harness.background.error is None
    )
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert len(figures) == 2 and figures[0] == figures[1]
    assert form_editable(window.plan_panel)
    assert harness.handles.known_outcome(token) is not None
    assert not harness.gate.snapshot()


@pytest.mark.parametrize("action", ["stop", "close"])
def test_resume_submission_stop_or_close_keeps_window_locked_until_delivery(
    run_attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    catalog_source: InjectedCatalog,
    action: str,
) -> None:
    window, harness, _entry = run_attachment
    attempts: list[str] = []

    def init(_env: InitEnv[Context], _plan: Plan) -> None:
        return None

    @workflow("register-window", plan=Plan, tunables=Tunables, init=init)
    def step(
        env: WorkflowEnv[Context],
        plan: Plan,
        tun: Tunables,
        state: None,
    ) -> Step[None, int]:
        del plan, tun, state
        attempts.append("step")
        env.pbar("work", 2).refresh()
        window.pause("register-window")
        yield from env.wait_until(env.started_at)
        return Done()

    catalog_source.install(step, window)
    assert isinstance(window.start("register-window"), int)
    harness.complete_segment()
    active_controller = window.controller
    assert active_controller is not None
    observations: list[tuple[bool, bool]] = []
    close_results: list[bool] = []

    token = active_controller.operation_id
    assert token is not None

    def during_submit() -> None:
        harness.background.on_submit = None
        if action == "close":
            close_results.append(window.close())
        else:
            window.stop("register-window")
        observations.append(
            (form_editable(window.plan_panel), active_controller.has_pending_segment)
        )

    harness.background.on_submit = during_submit
    assert window.resume("register-window") == token
    harness.background.on_submit = None
    assert observations == [(False, True)]
    assert active_controller.has_pending_segment
    assert close_results == ([False] if action == "close" else [])
    assert not form_editable(window.plan_panel) and harness.gate.snapshot()
    harness.background.execute()
    assert not form_editable(window.plan_panel) and harness.gate.snapshot()
    harness.background.deliver()
    controller = window.controller
    assert controller is not None
    status = controller.status()
    assert status is not None and status.lifecycle == "stopped"
    assert attempts == ["step"]
    assert form_editable(window.plan_panel) and form_editable(window.tunables_panel)
    assert harness.handles.live_count() == 0 and not harness.gate.snapshot()
    assert harness.progress.bars_for_owner("workflow") == ()
    assert window.close()


def test_resume_stop_rejection_unlocks_window_without_a_producer(
    run_attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    catalog_source: InjectedCatalog,
) -> None:
    window, harness, _entry = run_attachment
    attempts: list[str] = []

    def init(_env: InitEnv[Context], _plan: Plan) -> None:
        return None

    @workflow("reject-window", plan=Plan, tunables=Tunables, init=init)
    def step(
        env: WorkflowEnv[Context],
        plan: Plan,
        tun: Tunables,
        state: None,
    ) -> Step[None, int]:
        del plan, tun, state
        attempts.append("step")
        env.pbar("work", 2).refresh()
        window.pause("reject-window")
        yield from env.wait_until(env.started_at)
        return Done()

    catalog_source.install(step, window)
    assert isinstance(window.start("reject-window"), int)
    harness.complete_segment()

    def during_submit() -> None:
        harness.background.on_submit = None
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
    assert form_editable(window.plan_panel) and form_editable(window.tunables_panel)
    assert start is not None and start.isEnabled()
    assert pause is not None and not pause.isEnabled()
    assert resume is not None and not resume.isEnabled()
    assert harness.progress.bars_for_owner("workflow") == ()
    assert not harness.gate.snapshot() and harness.handles.live_count() == 0
    assert not controller.has_pending_segment
    harness.complete_segment()  # Deliver terminal device readback, not a producer.
    assert harness.background.work is None and attempts == ["step"]
    assert window.close()


@pytest.mark.parametrize("action", ["stop", "close"])
def test_paused_window_stop_or_close_ends_run_without_another_worker(
    run_attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    catalog_source: InjectedCatalog,
    action: str,
) -> None:
    window, harness, entry = run_attachment
    attempts: list[str] = []

    def init(_env: InitEnv[Context], _plan: Plan) -> None:
        return None

    @workflow("paused-close", plan=Plan, tunables=Tunables, init=init)
    def step(
        env: WorkflowEnv[Context],
        plan: Plan,
        tun: Tunables,
        state: None,
    ) -> Step[None, int]:
        del plan, tun, state
        attempts.append("step")
        env.pbar("work", 2).refresh()
        window.pause("paused-stop")
        yield from env.wait_until(env.started_at)
        return Done()

    catalog_source.install(step, window)
    assert isinstance(window.start("paused-stop"), int)
    harness.complete_segment()
    assert harness.background.work is None
    assert harness.progress.bars_for_owner("workflow")
    assert not form_editable(window.plan_panel) and not form_editable(
        window.tunables_panel
    )
    assert window.tuning_panel is not None and window.tuning_panel.isEnabled()
    start = window.findChild(QPushButton, "workflow.run.start")
    pause = window.findChild(QPushButton, "workflow.run.pause")
    resume = window.findChild(QPushButton, "workflow.run.resume")
    assert start is not None and not start.isEnabled()
    assert pause is not None and not pause.isEnabled()
    assert resume is not None and resume.isEnabled()
    with pytest.raises(InvalidRunState):
        window.start("replacement")
    assert not (entry.result_path / "workflows/paused-close_replacement").exists()
    if action == "close":
        assert window.close()
    else:
        status = window.stop("paused-stop")
        assert status is not None and status.lifecycle == "stopped"
        assert form_editable(window.plan_panel) and start.isEnabled()
        assert not pause.isEnabled() and not resume.isEnabled()
    controller = window.controller
    assert controller is not None
    status = controller.status()
    assert status is not None and status.lifecycle == "stopped"
    harness.complete_segment()  # Stop schedules device readback, not a new step.
    assert harness.background.work is None and attempts == ["step"]
    assert not harness.gate.snapshot()
    assert harness.progress.bars_for_owner("workflow") == ()


def test_resume_failed_tunables_unwind_unlocks_window_and_preserves_error(
    run_attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    catalog_source: InjectedCatalog,
) -> None:
    window, harness, entry = run_attachment
    native_errors: list[OSError] = []
    observed: list[tuple[bool, bool]] = []

    def init(_env: InitEnv[Context], _plan: Plan) -> None:
        return None

    @workflow("failed-read", plan=Plan, tunables=Tunables, init=init)
    def step(
        env: WorkflowEnv[Context],
        plan: Plan,
        tun: Tunables,
        state: None,
    ) -> Step[None, int]:
        del plan, tun, state
        env.pbar("retained", 2).refresh()
        window.pause("failed-read")
        yield from env.wait_until(env.started_at)
        return Done()

    catalog_source.install(step, window)
    token = window.start("failed-read")
    assert isinstance(token, int)
    harness.complete_segment()
    controller, panel = window.controller, window.tuning_panel
    assert panel is not None
    accepted = panel.host.accepted_input.data
    journal = entry.result_path / "workflows/failed-read_failed-read/journal.jsonl"
    original = journal.read_bytes()
    journal.unlink()
    journal.mkdir()

    def during_submit() -> None:
        harness.background.on_submit = None
        try:
            window.apply_tunables(
                "failed-read", {("sample", "value"): 45.0}, actor=Actor("agent", "rpc")
            )
        except DIRECTORY_OPEN_ERROR as error:
            native_errors.append(error)
        observed.append(
            (form_editable(window.plan_panel), controller.has_pending_segment)
        )

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
    assert sample_input(panel.panel.state.input.data)["value"] == 45.0
    assert form_editable(window.plan_panel) and form_editable(window.tunables_panel)
    assert controller.operation_id is None and not controller.has_pending_segment
    assert harness.handles.known_outcome(token) == OperationOutcome(
        "failed", failed.reason
    )
    harness.complete_segment()  # Deliver readback after the rejected producer.
    assert harness.background.work is None and not harness.gate.snapshot()
    assert not harness.progress.bars_for_owner("workflow")
    assert window.close()


def test_prepare_forms_rejects_pending_and_paused_run(
    run_attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    catalog_source: InjectedCatalog,
) -> None:
    window, harness, _entry = run_attachment

    def init(_env: InitEnv[Context], _plan: Plan) -> None:
        return None

    @workflow("prepare-guard", plan=Plan, tunables=Tunables, init=init)
    def step(
        env: WorkflowEnv[Context],
        plan: Plan,
        tun: Tunables,
        state: None,
    ) -> Step[None, int]:
        del plan, tun, state
        window.pause("prepare-guard")
        with pytest.raises(InvalidRunState):
            window.prepare_forms()
        yield from env.wait_until(env.started_at)
        return Done()

    catalog_source.install(step, window)
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
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
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
    run_attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    catalog_source: InjectedCatalog,
    phase: str,
) -> None:
    window, harness, entry = run_attachment
    attempts: list[str] = []

    def init(_env: InitEnv[Context], _plan: Plan) -> None:
        return None

    @workflow("replacement-target", plan=Plan, tunables=Tunables, init=init)
    def step(
        env: WorkflowEnv[Context],
        plan: Plan,
        tun: Tunables,
        state: None,
    ) -> Step[None, int]:
        del plan, tun, state
        attempts.append("step")
        env.plots.subplots("live")
        env.pbar("work", 2).refresh()
        window.pause("run-B")
        yield from env.wait_until(env.started_at)
        return Done()

    catalog_source.install(step, window)
    assert isinstance(window.start("run-A"), int)
    window.stop("run-A")
    harness.complete_segment()
    harness.complete_segment()  # Deliver run-A's device readback before replacement.
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
    bars = harness.progress.bars_for_owner("workflow")
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
            harness.progress.bars_for_owner("workflow"),
        ) == (gate, bars)
        assert (
            window.tuning_panel,
            form_editable(window.plan_panel),
            form_editable(window.tunables_panel),
        ) == (tuning, False, False)
    assert attempts == ([] if phase == "pending" else ["step"])
    stop_button = window.findChild(QPushButton, "workflow.run.stop")
    assert stop_button is not None
    stop_button.click()
    if phase == "pending":
        assert harness.gate.snapshot()
        harness.complete_segment()
        assert not (entry.result_path / "workflows/replacement-target_run-B").exists()
    else:
        stopped = controller.status()
        assert stopped is not None and stopped.lifecycle == "stopped"
    assert controller.run_id == "run-B" and form_editable(window.plan_panel)
    assert harness.progress.bars_for_owner("workflow") == ()
    assert not harness.gate.snapshot() and harness.handles.live_count() == 0
