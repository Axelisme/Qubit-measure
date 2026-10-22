"""Workflow hosting through caller registrations and visible mouse/keyboard input."""

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from threading import Event

import pytest
from pydantic import BaseModel, ConfigDict
from qtpy.QtWidgets import QApplication
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.experiment.workflows import Live, WorkflowEnv, workflow
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.workflow_attachment import WorkflowAttachment
from zcu_tools.gui.app.measure.services.workflow_control import WorkflowControlFacet
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.app.measure.ui.workflow_pane import WorkflowPane
from zcu_tools.gui.app.measure.ui.workflow_tabs import WorkflowTabs
from zcu_tools.gui.control_identity import ControlStatus
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.form import FormModel
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.gui.session.types import SessionEnv

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._synthetic import SampleAdapter
from tests.gui.app.measure._workflow_ask import (
    AskAuthor,
    Choose,
    Native,
    OtherAnswer,
    Values,
)


class Plan(FormModel):
    cycles: int = 1


class Tunables(FormModel):
    gain: float = 0.25


class NativePlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    cycles: int


class NativeTunables(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    gain: float


@workflow("synthetic", "Synthetic retained-worker wait")
async def _waiting(
    env: WorkflowEnv[None], plan: NativePlan, tun: Live[NativeTunables]
) -> None:
    # A public wait keeps this invocation live until the scenario requests Stop.
    await env.wait_until(env.started_at + timedelta(days=1))


@pytest.fixture
def workflow_window(
    qapp: QApplication, tmp_path: Path
) -> Iterator[tuple[Controller, MainWindow]]:
    registry = Registry()
    registry.register("experiment", SampleAdapter)
    ctrl = Controller(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        registry,
        None,
        BaseEventBus(),
        project_root=str(tmp_path),
        progress_transport=DirectProgressTransport(),
    )
    yield from _host_window(qapp, ctrl)


def _host_window(
    qapp: QApplication, ctrl: Controller
) -> Iterator[tuple[Controller, MainWindow]]:
    ctrl.workflow_control().configure(
        (
            WorkflowAttachment(
                script=_waiting,
                plan_model=Plan,
                knobs_model=Tunables,
                plan_to_cfg=lambda form, env, snapshot: NativePlan(cycles=form.cycles),
                knobs_to_cfg=lambda form: NativeTunables(gain=form.gain),
                seed=lambda plan, form: None,
            ),
        )
    )
    ctrl.setup_control.new_entry(EntryRequest("workflow-ui"))
    ctrl.setup_control.new_point(PointCreateRequest("source", "", CommandOrigin("gui")))
    board = white_noise()
    ctrl.install_soc(board.soc, board.soccfg)
    window = MainWindow(
        ctrl, templates=lambda: {}, dialog_presenter=RecordingDialogPresenter()
    )
    ctrl.add_view(window)
    window.refresh_context_panel()
    window.show()
    qapp.processEvents()
    destroyed = Event()
    window.destroyed.connect(destroyed.set)
    try:
        yield ctrl, window
    finally:
        # close() drains but does not cancel a retained Ask/wait after an assertion
        # failure. Release each live invocation before waiting for owner delivery.
        control = ctrl.workflow_control()
        for tab in control.tabs():
            if tab.busy and tab.run_id is not None:
                control.stop(tab.run_id)
        ctrl.close()
        if not destroyed.is_set():
            window.deleteLater()
        qapp.processEvents()


@dataclass
class CleanupAttempt:
    unsubscribe: Callable[[], None]
    failures_left: int = 0
    attempts: int = 0
    successes: int = 0

    def __call__(self) -> None:
        self.attempts += 1
        if self.failures_left:
            self.failures_left -= 1
            raise RuntimeError("subscription release failed")
        self.unsubscribe()
        self.successes += 1


class CleanupFacet(WorkflowControlFacet):
    cleanups: list[CleanupAttempt]
    closed: list[str]

    def subscribe(self, listener: Callable[[], None]) -> Callable[[], None]:
        cleanup = CleanupAttempt(super().subscribe(listener))
        self.cleanups.append(cleanup)
        return cleanup

    def close(self, tab_id: str) -> bool:
        self.closed.append(tab_id)
        return super().close(tab_id)


class CleanupController(Controller):
    def __init__(self, root: Path) -> None:
        state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
        super().__init__(
            state,
            Registry(),
            None,
            BaseEventBus(),
            project_root=str(root),
            progress_transport=DirectProgressTransport(),
        )
        self.workflow = CleanupFacet(state, self.session_services)
        self.workflow.cleanups = []
        self.workflow.closed = []

    def workflow_control(self) -> CleanupFacet:
        return self.workflow


@pytest.fixture
def cleanup_window(
    qapp: QApplication, tmp_path: Path
) -> Iterator[tuple[CleanupFacet, MainWindow]]:
    ctrl = CleanupController(tmp_path)
    for _, window in _host_window(qapp, ctrl):
        ctrl.workflow.open("synthetic")
        yield ctrl.workflow, window


def _workflow_tabs(window: MainWindow) -> WorkflowTabs:
    return window.findChild(WorkflowTabs)


def _workflow_pane(tabs: WorkflowTabs) -> WorkflowPane:
    pane = tabs.widget(0)
    if not isinstance(pane, WorkflowPane):
        raise LookupError("No workflow pane mounted")
    return pane


def test_pane_dispose_retries_failed_subscription(cleanup_window):
    control, window = cleanup_window
    cleanup = control.cleanups[-1]
    cleanup.failures_left = 1
    pane = _workflow_pane(_workflow_tabs(window))
    # The owner is stopped; only this view's cleanup can now fail.
    with pytest.raises(RuntimeError, match="subscription release failed"):
        pane.dispose()
    pane.dispose()
    pane.dispose()
    assert cleanup.successes == 1
    assert cleanup.attempts == 2


def test_tabs_dispose_retries_without_repeating_successful_cleanup(cleanup_window):
    control, window = cleanup_window
    control.open("synthetic")
    cleanup = control.cleanups[-1]
    cleanup.failures_left = 1
    tabs = _workflow_tabs(window)
    with pytest.raises(RuntimeError, match="subscription release failed"):
        tabs.dispose()
    tabs.dispose()
    tabs.dispose()
    assert [c.successes for c in control.cleanups] == [1, 1, 1]
    assert cleanup.attempts == 2


def test_close_retries_view_cleanup_after_owner_has_closed(cleanup_window, tmp_path):
    control, window = cleanup_window
    cleanup = control.cleanups[-1]
    cleanup.failures_left = 1
    driver = ComponentDriver(window)
    _shot(driver, tmp_path, "01-before-close")
    driver.click_tooltip("Close synthetic (1)")
    _shot(driver, tmp_path, "02-cleanup-failed-pane-retained")
    assert driver.tabs("synthetic (1)").captions == ("synthetic (1)",)
    driver.click_tooltip("Close synthetic (1)")
    _shot(driver, tmp_path, "03-cleanup-retried-pane-removed")
    assert not any(w.object_name == "workflow.status" for w in driver.visible().widgets)
    assert cleanup.successes == 1
    assert cleanup.attempts == 2
    assert len(control.closed) == 1


def test_window_dispose_retries_workflow_cleanup(cleanup_window):
    control, window = cleanup_window
    cleanup = control.cleanups[-1]
    cleanup.failures_left = 1
    with pytest.raises(RuntimeError, match="subscription release failed"):
        window.dispose_control_watches()
    window.dispose_control_watches()
    window.dispose_control_watches()
    assert all(c.successes == 1 for c in control.cleanups)
    assert cleanup.attempts == 2


def test_full_window_dispose_retries_cleanup_then_destroys_tree(cleanup_window):
    control, window = cleanup_window
    control.open("synthetic")
    cleanup = control.cleanups[-1]
    cleanup.failures_left = 1
    destroyed = Event()
    window.destroyed.connect(destroyed.set)

    with pytest.raises(RuntimeError, match="subscription release failed"):
        window.dispose()
    assert not destroyed.is_set()
    assert [c.successes for c in control.cleanups] == [1, 1, 0]

    window.dispose()
    assert [c.successes for c in control.cleanups] == [1, 1, 1]
    assert [c.attempts for c in control.cleanups] == [1, 1, 2]
    assert destroyed.is_set()


def _ready(driver: ComponentDriver) -> None:
    assert any(w.object_name == "workflow.toggle" for w in driver.visible().widgets), (
        "MainWindow must expose the single Experiment/Workflow toggle"
    )


def _shot(driver: ComponentDriver, root: Path, step: str) -> None:
    driver.screenshot(root / f"{step}.png")


def _text(driver: ComponentDriver, name: str) -> str:
    return next(w.text for w in driver.visible().widgets if w.object_name == name)


def _open(driver: ComponentDriver, evidence: Path) -> None:
    driver.select_menu("New Tab ▾", ("synthetic",))
    driver.screenshot(evidence)


def _status(driver: ComponentDriver, text: str) -> bool:
    return any(
        w.object_name == "workflow.status" and w.text == text
        for w in driver.visible().widgets
    )


def _controls(driver: ComponentDriver) -> set[str]:
    return {
        widget.text
        for widget in driver.visible().widgets
        if widget.kind == "QPushButton"
        and widget.text in {"Refresh", "Start", "Pause", "Resume", "Stop"}
    }


def test_toggle_follows_reload_with_compact_unclipped_captions(
    workflow_window, tmp_path
):
    _, window = workflow_window
    driver = ComponentDriver(window)
    for step, caption in (
        ("01-experiment-toolbar", "Workflow tabs"),
        ("02-workflow-toolbar", "Experiment tabs"),
    ):
        _ask_shot(driver, tmp_path, step)
        reload = driver.observe("Reload experiments")
        toggle = driver.observe(object_name="workflow.toggle")
        assert toggle.text == caption
        assert reload.bounds is not None and toggle.bounds is not None
        x, y, width, height = reload.bounds
        tx, ty, _tw, th = toggle.bounds
        assert tx > x + width, "Toggle must follow Reload, not New Tab"
        assert (ty, th) == (y, height)
        assert toggle.caption_fits and toggle.fully_visible
        driver.click(object_name="workflow.toggle")


def test_toggle_same_name_tabs_forms_and_idle_close(workflow_window, tmp_path):
    _, window = workflow_window
    driver = ComponentDriver(window)
    _ready(driver)
    _shot(driver, tmp_path, "01-experiments")
    driver.click(object_name="workflow.toggle")
    _shot(driver, tmp_path, "02-empty-workflows")
    _open(driver, tmp_path / "03-first-tab.png")
    assert driver.tabs("synthetic (1)").current == "synthetic (1)"
    assert driver.tabs("Plan").captions == ("Plan", "Tunables")
    driver.enter_text(value="7", object_name="form.input.cycles")
    _shot(driver, tmp_path, "04-plan-edited")
    _open(driver, tmp_path / "05-second-tab.png")
    assert driver.tabs("synthetic (2)").captions == ("synthetic (1)", "synthetic (2)")
    assert _text(driver, "form.input.cycles") == "1"
    driver.select_tab("synthetic (1)")
    _shot(driver, tmp_path, "06-independent-plan")
    assert _text(driver, "form.input.cycles") == "7"
    driver.select_tab("Tunables")
    _shot(driver, tmp_path, "07-tunables")
    assert _text(driver, "form.input.gain") == "0.25"
    driver.click_tooltip("Close synthetic (1)")
    _shot(driver, tmp_path, "09-one-closed")
    assert driver.tabs("synthetic (2)").captions == ("synthetic (2)",)
    driver.click_tooltip("Close synthetic (2)")
    _shot(driver, tmp_path, "10-last-closed")
    _open(driver, tmp_path / "11-reopened.png")
    assert driver.observe("Start").enabled
    driver.click(object_name="workflow.toggle")
    _shot(driver, tmp_path, "12-experiments-again")
    assert driver.observe(object_name="empty_tabs_hint").visible


def test_workflow_pane_displays_active_chain_and_live_leaves(workflow_window, tmp_path):
    _, window = workflow_window
    driver = ComponentDriver(window)
    driver.click(object_name="workflow.toggle")
    _open(driver, tmp_path / "scope-01-open.png")
    driver.click("Start")
    qt_until(lambda: _status(driver, "running"), label="workflow accepted")
    _shot(driver, tmp_path, "scope-02-running")
    active = driver.observe(object_name="workflow.active_path")
    leaves = driver.observe(object_name="workflow.live_paths")
    assert active.visible and active.text == "workflow:synthetic/operation#1"
    assert leaves.visible and leaves.text == "workflow:synthetic/operation#1"
    driver.click("Stop")
    qt_until(lambda: _status(driver, "cancelled"), label="scope root cancelled")
    _shot(driver, tmp_path, "scope-03-cancelled")
    assert driver.observe(object_name="workflow.active_path").text == "workflow:synthetic"


def test_running_paused_close_lock_and_badge_navigation(workflow_window, tmp_path):
    _, window = workflow_window
    driver = ComponentDriver(window)
    _ready(driver)
    driver.select_menu("New Tab ▾", ("experiment",))
    _shot(driver, tmp_path, "01-experiment")
    driver.click(object_name="workflow.toggle")
    _shot(driver, tmp_path, "01a-workflows")
    _open(driver, tmp_path / "01b-first-tab.png")
    _open(driver, tmp_path / "01c-second-tab.png")
    driver.select_tab("synthetic (1)")
    _shot(driver, tmp_path, "01d-selected-first")
    driver.click("Start")
    qt_until(lambda: _status(driver, "running"), label="workflow accepted")
    _shot(driver, tmp_path, "02-running")
    driver.click_tooltip("Close synthetic (1)")
    _shot(driver, tmp_path, "03-running-close-refused")
    assert driver.tabs("synthetic (1)").captions == ("synthetic (1)", "synthetic (2)")
    driver.select_tab("synthetic (2)")
    _shot(driver, tmp_path, "04-other-locked")
    assert not driver.observe("Start").enabled
    driver.click(object_name="workflow.toggle")
    _shot(driver, tmp_path, "05-experiment-locked")
    assert not driver.observe("Run").enabled
    driver.click(object_name="workflow.badge")
    _shot(driver, tmp_path, "06-badge-navigation")
    assert driver.tabs("synthetic (1)").current == "synthetic (1)"
    driver.click("Pause")
    qt_until(lambda: _status(driver, "paused"), label="native pause gate")
    _shot(driver, tmp_path, "07-paused")
    driver.click_tooltip("Close synthetic (1)")
    _shot(driver, tmp_path, "08-paused-close-refused")
    assert driver.tabs("synthetic (1)").captions == ("synthetic (1)", "synthetic (2)")
    driver.select_tab("Tunables")
    _shot(driver, tmp_path, "08a-tunables")
    driver.enter_text(value="0.5", object_name="form.input.gain")
    _shot(driver, tmp_path, "08b-edited-tunables")
    driver.click("Apply")
    _shot(driver, tmp_path, "09-tunables-applied")
    assert driver.observe(object_name="workflow.message").text == "Applied revision 1"
    driver.click("Resume")
    qt_until(lambda: _status(driver, "running"), label="resumed")
    _shot(driver, tmp_path, "10-resumed")
    driver.click("Stop")
    qt_until(lambda: _status(driver, "cancelled"), label="stop delivered")
    _shot(driver, tmp_path, "11-stopped")
    driver.click_tooltip("Close synthetic (1)")
    _shot(driver, tmp_path, "12-closed-after-stop")
    assert driver.tabs("synthetic (2)").captions == ("synthetic (2)",)
    assert driver.observe("Start").enabled


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("idle", {"Refresh", "Start"}),
        ("running", {"Pause", "Stop"}),
        ("paused", {"Resume", "Stop"}),
        ("waiting", {"Stop"}),
    ],
)
def test_run_state_shows_only_meaningful_controls(
    workflow_window, tmp_path, state, expected
):
    ctrl, window = workflow_window
    if state == "waiting":
        ctrl.workflow_control().configure((AskAuthor().attachment(),))
    driver = ComponentDriver(window)
    driver.click(object_name="workflow.toggle")
    driver.select_menu("New Tab ▾", ("asking" if state == "waiting" else "synthetic",))
    _ask_shot(driver, tmp_path, "controls-01-idle")
    if state != "idle":
        driver.click("Start")
        # The retained invocation delivers Ask or the synthetic public wait.
        qt_until(
            lambda: _status(driver, "paused" if state == "waiting" else "running"),
            label="control scenario start delivery",
        )
    if state == "paused":
        driver.click("Pause")
        # Pause interrupts the wait and retains the invocation at its native gate.
        qt_until(lambda: _status(driver, "paused"), label="control scenario paused")
    _ask_shot(driver, tmp_path, f"controls-02-{state}")
    try:
        assert _controls(driver) == expected
    finally:
        if state != "idle":
            driver.click("Stop")
            # Stop releases the synthetic wait or retained Ask before teardown.
            qt_until(
                lambda: _status(driver, "cancelled"), label="control scenario stop"
            )
            _ask_shot(driver, tmp_path, "controls-03-stopped")


@pytest.fixture
def asking_window(workflow_window):
    ctrl, window = workflow_window
    author = AskAuthor()
    ctrl.workflow_control().configure((author.attachment(),))
    return window


def _ask_shot(driver: ComponentDriver, root: Path, step: str) -> None:
    driver.screenshot(root / f"{step}.png")
    (root / f"{step}.txt").write_text(repr(driver.visible()), encoding="utf-8")


@pytest.mark.parametrize("reopen", [False, True])
def test_idle_ask_content_is_hidden_after_open_and_reopen(
    asking_window, tmp_path, reopen
):
    driver = ComponentDriver(asking_window)
    driver.click(object_name="workflow.toggle")
    driver.select_menu("New Tab ▾", ("asking",))
    _ask_shot(driver, tmp_path, "idle-ask-01-opened")
    if reopen:
        driver.click_tooltip("Close asking (1)")
        _ask_shot(driver, tmp_path, "idle-ask-02-closed")
        driver.select_menu("New Tab ▾", ("asking",))
        _ask_shot(driver, tmp_path, "idle-ask-03-reopened")
    assert driver.tabs("Plan").captions == ("Plan", "Tunables")
    assert not any(
        widget.object_name
        in {"workflow.ask", "workflow.answer", "workflow.answer_status"}
        for widget in driver.visible().widgets
    ), "Removing the Ask tab must also hide its page contents"


@pytest.mark.parametrize("cancel", [False, True])
def test_ask_answer_or_cancel_preserves_visible_exchange(
    asking_window, tmp_path, cancel
):
    driver = ComponentDriver(asking_window)
    _ask_shot(driver, tmp_path, "ask-01-experiments")
    driver.click(object_name="workflow.toggle")
    _ask_shot(driver, tmp_path, "ask-02-workflows")
    driver.select_menu("New Tab ▾", ("asking",))
    _ask_shot(driver, tmp_path, "ask-03-opened")
    assert driver.tabs("Plan").captions == ("Plan", "Tunables")
    driver.click("Start")
    # Start must yield its question. Badge and pane project separate snapshots;
    # native paused alone does not establish that both have rendered the Ask.
    qt_until(
        lambda: (
            _status(driver, "paused")
            and any(
                w.object_name == "workflow.badge" and "等待回答" in w.text
                for w in driver.visible().widgets
            )
            and any(
                w.object_name == "workflow.ask" and "authored evidence" in w.text
                for w in driver.visible().widgets
            )
        ),
        label="Ask pane and badge presentation",
    )
    _ask_shot(driver, tmp_path, "ask-04-question")
    assert driver.tabs("Ask").current == "Ask", "Pending question must select Ask"
    assert "authored evidence" in driver.observe(object_name="workflow.ask").text
    assert "value" in driver.observe(object_name="workflow.answer_schema").text
    assert "等待回答" in driver.observe(object_name="workflow.badge").text
    assert _controls(driver) == {"Stop"}
    driver.select_tab("Tunables")
    _ask_shot(driver, tmp_path, "ask-05-tunables")
    driver.click(object_name="workflow.toggle")
    _ask_shot(driver, tmp_path, "ask-06-experiments")
    driver.click(object_name="workflow.badge")
    _ask_shot(driver, tmp_path, "ask-07-badge-question")
    assert driver.tabs("Ask").current == "Ask"
    if cancel:
        driver.click("Stop")
        # Stop releases Ask; Start becomes available after terminal owner delivery,
        # which may follow the native cancelled status.
        qt_until(
            lambda: (
                _status(driver, "cancelled")
                and any(
                    w.text == "Start" and w.enabled for w in driver.visible().widgets
                )
            ),
            label="Ask cancellation and Start presentation",
        )
    else:
        driver.enter_text(object_name="workflow.answer", value='{"value": 17}')
        _ask_shot(driver, tmp_path, "ask-08-answer-edited")
        driver.click("Answer")
        # Answer releases the coroutine. Native done can precede owner delivery;
        # the next Start also requires that delivery to release the RUN lease.
        qt_until(
            lambda: (
                _status(driver, "done")
                and any(
                    w.text == "Start" and w.enabled for w in driver.visible().widgets
                )
            ),
            label="answer completion and Start presentation",
        )
    _ask_shot(driver, tmp_path, "ask-09-retained-exchange")
    assert "authored evidence" in driver.observe(object_name="workflow.ask").text
    assert not driver.observe("Answer").enabled
    assert not driver.observe(object_name="workflow.answer").enabled
    if cancel:
        assert driver.observe(object_name="workflow.answer_status").text == "提問已取消"
    else:
        assert driver.observe(object_name="workflow.answer").text == '{"value": 17}'
        assert "已回答" in driver.observe(object_name="workflow.answer_status").text
    assert driver.observe("Start").enabled
    assert driver.tabs("Ask").captions == ("Plan", "Tunables", "Ask")
    driver.click_tooltip("Close asking (1)")
    _ask_shot(driver, tmp_path, "ask-10-closed")
    driver.select_menu("New Tab ▾", ("asking",))
    _ask_shot(driver, tmp_path, "ask-11-reopened")
    assert driver.tabs("Plan").captions == ("Plan", "Tunables")


@pytest.fixture
def restart_questions_window(workflow_window):
    ctrl, window = workflow_window

    @workflow("restart-questions", "Synthetic fresh-run Ask presentation")
    async def questions(
        env: WorkflowEnv[None], plan: Native, tun: Live[Native]
    ) -> None:
        # An explicit zero plan makes the entire run question-free.
        if plan.value:
            await env.ask(Choose(point=plan.value))

    ctrl.workflow_control().configure(
        (
            WorkflowAttachment(
                script=questions,
                plan_model=Values,
                knobs_model=Values,
                plan_to_cfg=lambda form, env, snapshot: Native(value=form.value),
                knobs_to_cfg=lambda form: Native(value=form.value),
                seed=lambda plan, form: None,
            ),
        )
    )
    return window


def test_new_run_hides_retained_exchange_until_new_question(
    restart_questions_window, tmp_path
):
    driver = ComponentDriver(restart_questions_window)
    driver.click(object_name="workflow.toggle")
    _ask_shot(driver, tmp_path, "restart-01-workflows")
    driver.select_menu("New Tab ▾", ("restart-questions",))
    _ask_shot(driver, tmp_path, "restart-02-opened")
    driver.enter_text(object_name="form.input.value", value="1")
    _ask_shot(driver, tmp_path, "restart-03-first-plan")
    driver.click("Start")
    # The author coroutine yields the explicitly requested first question.
    qt_until(lambda: _status(driver, "paused"), label="first run question")
    _ask_shot(driver, tmp_path, "restart-04-first-question")
    assert driver.tabs("Ask").current == "Ask"
    assert '"point": 1' in driver.observe(object_name="workflow.ask").text
    driver.enter_text(object_name="workflow.answer", value='{"value": 17}')
    _ask_shot(driver, tmp_path, "restart-05-answer-edited")
    driver.click("Answer")
    # Answer resumes the author coroutine, which returns.
    qt_until(lambda: _status(driver, "done"), label="first run answered")
    _ask_shot(driver, tmp_path, "restart-06-retained-exchange")
    assert driver.tabs("Ask").captions == ("Plan", "Tunables", "Ask")
    assert '"point": 1' in driver.observe(object_name="workflow.ask").text
    assert driver.observe(object_name="workflow.answer").text == '{"value": 17}'
    assert driver.observe(object_name="workflow.answer_status").text == "已回答"
    assert not driver.observe("Answer").enabled
    driver.select_tab("Plan")
    _ask_shot(driver, tmp_path, "restart-07-plan")
    driver.enter_text(object_name="form.input.value", value="0")
    _ask_shot(driver, tmp_path, "restart-08-question-free-plan")
    driver.click("Start")
    # The zero plan returns without any Ask, so this is a stable no-question state.
    qt_until(lambda: _status(driver, "done"), label="question-free run completed")
    _ask_shot(driver, tmp_path, "restart-09-question-free-run")
    assert driver.tabs("Plan").captions == ("Plan", "Tunables")
    assert not any(
        widget.object_name
        in {"workflow.ask", "workflow.answer", "workflow.answer_status"}
        for widget in driver.visible().widgets
    ), "A new run must clear the old exchange and hide its Ask page"
    driver.enter_text(object_name="form.input.value", value="2")
    _ask_shot(driver, tmp_path, "restart-10-new-question-plan")
    driver.click("Start")
    # The nonzero plan yields a fresh question on this same tab's third run.
    qt_until(lambda: _status(driver, "paused"), label="new run question")
    _ask_shot(driver, tmp_path, "restart-11-new-question")
    assert driver.tabs("restart-questions (1)").captions == ("restart-questions (1)",)
    assert driver.tabs("Ask").captions == ("Plan", "Tunables", "Ask")
    assert driver.tabs("Ask").current == "Ask"
    assert '"point": 2' in driver.observe(object_name="workflow.ask").text
    assert driver.observe(object_name="workflow.answer").text == "{}"
    assert driver.observe(object_name="workflow.answer_status").text == "等待回答"
    assert driver.observe("Answer").enabled
    driver.click("Stop")
    # Stop drains the retained question before the shared window fixture closes.
    qt_until(lambda: _status(driver, "cancelled"), label="new run question cancelled")
    _ask_shot(driver, tmp_path, "restart-12-stopped")


@pytest.fixture
def two_questions_window(workflow_window):
    ctrl, window = workflow_window
    release = Event()

    @workflow("two-questions", "Synthetic successive Ask presentation")
    async def questions(
        env: WorkflowEnv[None], plan: Native, tun: Live[Native]
    ) -> None:
        first = await env.ask(Choose(point=1))
        if first is None:
            return
        if not release.wait(30):
            raise TimeoutError("Second question was not released by the scenario")
        await env.ask(Choose(point=2))

    ctrl.workflow_control().configure(
        (
            WorkflowAttachment(
                script=questions,
                plan_model=Values,
                knobs_model=Values,
                plan_to_cfg=lambda form, env, snapshot: Native(value=form.value),
                knobs_to_cfg=lambda form: Native(value=form.value),
                seed=lambda plan, form: None,
            ),
        )
    )
    try:
        yield window, release
    finally:
        release.set()


def test_next_question_replaces_answer_and_selects_ask(two_questions_window, tmp_path):
    window, release = two_questions_window
    driver = ComponentDriver(window)
    driver.click(object_name="workflow.toggle")
    driver.select_menu("New Tab ▾", ("two-questions",))
    _ask_shot(driver, tmp_path, "next-01-opened")
    driver.click("Start")
    # The retained invocation must yield the first synthetic question.
    qt_until(lambda: _status(driver, "paused"), label="first question")
    _ask_shot(driver, tmp_path, "next-02-first-question")
    assert '"point": 1' in driver.observe(object_name="workflow.ask").text
    driver.enter_text(object_name="workflow.answer", value='{"value": 17}')
    _ask_shot(driver, tmp_path, "next-03-first-answer")
    driver.click("Answer")
    driver.select_tab("Plan")
    _ask_shot(driver, tmp_path, "next-04-plan-during-continuation")
    release.set()
    # Releasing the author lets the author continuation yield its second question.
    qt_until(
        lambda: any(
            w.object_name == "workflow.ask" and '"point": 2' in w.text
            for w in driver.visible().widgets
        ),
        label="second question visible",
    )
    _ask_shot(driver, tmp_path, "next-05-second-question")
    assert driver.tabs("Ask").current == "Ask"
    assert driver.observe(object_name="workflow.answer").text == "{}"
    assert driver.observe(object_name="workflow.answer_status").text == "等待回答"
    driver.click("Stop")
    # Stop drains the second retained Ask before fixture shutdown.
    qt_until(lambda: _status(driver, "cancelled"), label="second question cancelled")
    _ask_shot(driver, tmp_path, "next-06-cancelled")


@pytest.mark.parametrize("reject_first", [False, True])
def test_external_answer_projects_accepted_exchange_to_window(
    asking_window, workflow_window, tmp_path, reject_first
):
    ctrl, _window = workflow_window
    control = ctrl.workflow_control()
    driver = ComponentDriver(asking_window)
    driver.click(object_name="workflow.toggle")
    driver.select_menu("New Tab ▾", ("asking",))
    _ask_shot(driver, tmp_path, "external-01-opened")
    driver.click("Start")
    # The retained Ask must publish answerable controls.
    qt_until(
        lambda: any(w.text == "Answer" and w.enabled for w in driver.visible().widgets),
        label="external Ask owner delivery",
    )
    _ask_shot(driver, tmp_path, "external-02-question")
    run_id = control.tabs()[0].run_id
    assert run_id is not None
    question = control.pending_question(run_id)
    assert question is not None
    if reject_first:
        # External callers receive synchronous validation; no worker segment is submitted.
        with pytest.raises(
            TypeError, match="Answer must match the question's declared model"
        ):
            control.answer(
                run_id, OtherAnswer(value=99), request_id=question["request_id"]
            )
        _ask_shot(driver, tmp_path, "external-03-rejected")
        assert control.pending_question(run_id) == question
        assert driver.observe(object_name="workflow.answer_status").text == "等待回答"
        assert driver.observe("Answer").enabled
    control.answer(run_id, {"value": 29}, request_id=question["request_id"])
    # This is deliberately an external-caller seam, not a replacement for the
    # ComponentDriver local Answer/Stop scenarios. The answered coroutine returns.
    qt_until(lambda: _status(driver, "done"), label="external answer delivery")
    _ask_shot(driver, tmp_path, "external-04-accepted")
    assert json.loads(driver.observe(object_name="workflow.answer").text) == {
        "value": 29
    }
    assert driver.observe(object_name="workflow.answer_status").text == "已回答"
    assert not driver.observe("Answer").enabled
    assert "authored evidence" in driver.observe(object_name="workflow.ask").text
    assert not any(
        w.object_name == "workflow.message"
        and "Answer must match the question's declared model" in w.text
        for w in driver.visible().widgets
    ), "An accepted exchange must not display an answer validation error"


def _check_remote_workflow_controls(driver: ComponentDriver, root: Path) -> None:
    """Only emergency Stop bypasses the visible Remote mutation gate."""
    _ask_shot(driver, root, "remote-05-remote-question")
    assert driver.observe(object_name="control_indicator").text == (
        "Remote · workflow-agent · online"
    )
    assert driver.observe("Stop").enabled, "Remote must not disable local workflow Stop"
    assert _controls(driver) == {"Stop"}
    assert not driver.observe("Answer").enabled
    driver.select_tab("Tunables")
    _ask_shot(driver, root, "remote-06-remote-tunables")
    assert not driver.observe("Apply").enabled
    assert driver.text_input(object_name="form.input.value").read_only
    driver.select_tab("Ask")
    _ask_shot(driver, root, "remote-07-remote-ask")


def _check_workflow_cancelled(
    driver: ComponentDriver,
    ctrl: Controller,
    run_id: str,
    token: int,
    author: AskAuthor,
) -> None:
    """Terminal delivery settles the original operation and restores admission."""
    assert driver.observe("Start").enabled
    assert driver.observe(object_name="workflow.answer_status").text == "提問已取消"
    assert ctrl.workflow_control().tabs()[0].run_id == run_id
    assert ctrl.workflow_control().tabs()[0].operation_id is None
    assert not any(p.kind == "run" for p in ctrl.get_hardware_gate_presence())
    completed = ctrl.await_operation(token, 0)
    assert completed is not None and completed.reason == "completed"
    assert completed.outcome is not None
    assert completed.outcome.status == "cancelled"
    assert [answer for answer, _thread in author.deliveries] == [None]


def test_remote_workflow_stop_reclaims_before_cancelling_same_operation(
    workflow_window, tmp_path
):
    ctrl, window = workflow_window
    control = ctrl.workflow_control()
    cleanup_entered, release_cleanup = Event(), Event()

    def hold_cleanup() -> None:
        cleanup_entered.set()
        # The scenario releases the synthetic author's finite cleanup in finally.
        assert release_cleanup.wait(30)

    author = AskAuthor(on_cancel=hold_cleanup)
    control.configure((author.attachment(),))
    driver = ComponentDriver(window)
    _ask_shot(driver, tmp_path, "remote-01-experiments")
    driver.click(object_name="workflow.toggle")
    _ask_shot(driver, tmp_path, "remote-02-workflows")
    driver.select_menu("New Tab ▾", ("asking",))
    _ask_shot(driver, tmp_path, "remote-03-opened")
    driver.click("Start")
    # The submitted retained invocation must deliver its synthetic Ask pause.
    qt_until(lambda: _status(driver, "paused"), label="remote Stop Ask pause")
    _ask_shot(driver, tmp_path, "remote-04-question")
    token = control.tabs()[0].operation_id
    run_id = control.tabs()[0].run_id
    assert token is not None and run_id is not None
    # Gate presence reports live age; compare only its stable public facts.
    lease = tuple(
        (p.kind, p.origin_kind, p.note) for p in ctrl.get_hardware_gate_presence()
    )
    assert any(kind == "run" for kind, _origin, _note in lease)
    ctrl.control_identity.connect("workflow-client", "workflow-agent", "owner")
    observed: list[tuple[str, bool, str | None, int | None]] = []

    def observe_authority(status: ControlStatus) -> None:
        tab = control.tabs()[0]
        observed.append(
            (
                status.holder,
                status.locked,
                tab.status.lifecycle if tab.status is not None else None,
                tab.operation_id,
            )
        )

    unwatch = ctrl.control_identity.watch(observe_authority)
    try:
        _check_remote_workflow_controls(driver, tmp_path)
        driver.click("Stop")
        # Stop must drain the retained Ask into the controlled author cleanup.
        qt_until(cleanup_entered.is_set, label="remote Stop author cancellation")
        _ask_shot(driver, tmp_path, "remote-08-local-cancelling")
        assert driver.observe(object_name="control_indicator").text == "Local locked"
        assert _controls(driver) == {"Stop"}
        assert observed == [
            ("rpc:workflow-agent", False, "paused", token),
            ("gui", True, "paused", token),
        ], "Reclaim must precede Stop's lifecycle change on the original operation"
        assert control.tabs()[0].operation_id == token
        assert control.tabs()[0].run_id == run_id
        assert (
            tuple(
                (p.kind, p.origin_kind, p.note)
                for p in ctrl.get_hardware_gate_presence()
            )
            == lease
        )
        pending = ctrl.await_operation(token, 0)
        assert pending is not None and pending.reason == "timeout"
        assert pending.outcome is None
        release_cleanup.set()
        # Released cleanup returns; the original owner must deliver cancellation.
        qt_until(
            lambda: _status(driver, "cancelled"), label="remote Stop terminal delivery"
        )
        _ask_shot(driver, tmp_path, "remote-09-local-stopped")
        _check_workflow_cancelled(driver, ctrl, run_id, token, author)
    finally:
        release_cleanup.set()
        unwatch()


def test_external_facet_open_is_projected_without_gui_selection(
    workflow_window, tmp_path
):
    ctrl, window = workflow_window
    driver = ComponentDriver(window)
    _ready(driver)
    # This is an external-caller seam test, not a replacement for the driven scenarios above.
    ctrl.workflow_control().open("synthetic")
    _shot(driver, tmp_path, "01-external-open")
    assert driver.tabs("synthetic (1)").current == "synthetic (1)"
    ctrl.workflow_control().open("synthetic")
    _shot(driver, tmp_path, "02-external-second-open")
    assert driver.tabs("synthetic (2)").current == "synthetic (2)"
    driver.click_tooltip("Close synthetic (2)")
    _shot(driver, tmp_path, "03-external-tab-closed-locally")
    assert driver.tabs("synthetic (1)").captions == ("synthetic (1)",)
