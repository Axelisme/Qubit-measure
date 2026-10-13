"""Run presentation: status bar, fixed Plan, live Tunables tab, banner and Stop.

Synthetic workflows only. The assertions read what a user sees: the status
bar's labels left to right, the diagnostics line under it, the form tabs'
titles and shown forms, enabled states and the paused banner. The wide
workflow's worker hook observes the Window while a point is live; the host
never sees a workflow name.
"""

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

from qtpy.QtCore import QEventLoop, QPoint, QRect, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTabWidget,
    QWidget,
)
from zcu_tools.gui.app.workflows.panel import WorkflowFormPanel
from zcu_tools.gui.app.workflows.window import WorkflowWindow

from ._window_support import (
    FigurePoint,
    WindowSession,
    bind_point,
    expand_all,
    install_soc,
)

RUN_ID = "presented"
PLAN_TAB, TUNABLES_TAB = 0, 1
STATUS_PARTS = (
    "workflow.run.previous",
    "workflow.run.badge",
    "workflow.run.workflow",
    "workflow.run.detail",
)


def child[W: QWidget](window: QWidget, kind: type[W], name: str) -> W:
    found = window.findChildren(kind, name)
    assert len(found) == 1, name
    return found[0]


def wait_for(qapp: QApplication, condition: Callable[[], bool]) -> None:
    """Run the Qt event loop in short slices until condition holds or 5 s pass."""
    deadline = monotonic() + 5.0
    while not condition():
        assert monotonic() < deadline, "condition not reached within 5 s"
        loop = QEventLoop()
        QTimer.singleShot(20, loop.quit)
        loop.exec()
    qapp.processEvents()


def settle(qapp: QApplication, seconds: float = 0.2) -> None:
    """Let several 50 ms status polls run."""
    deadline = monotonic() + seconds
    while monotonic() < deadline:
        loop = QEventLoop()
        QTimer.singleShot(20, loop.quit)
        loop.exec()


def status_text(window: QWidget) -> str:
    """The status bar read left to right: its shown labels, space-joined."""
    bar = child(window, QWidget, "workflow.run.status")
    parts = [child(window, QLabel, name) for name in STATUS_PARTS]
    for part in parts:
        assert bar.isAncestorOf(part)
    return " ".join(part.text() for part in parts if part.isVisibleTo(bar))


def badge(window: QWidget) -> str:
    return child(window, QLabel, "workflow.run.badge").text()


def message(window: QWidget) -> str:
    return child(window, QLabel, "workflow.run.message").text()


def banner(window: QWidget) -> QLabel:
    return child(window, QLabel, "workflow.run.paused_note")


def button(window: QWidget, name: str) -> QPushButton:
    return child(window, QPushButton, f"workflow.run.{name}")


def tabs(window: QWidget) -> QTabWidget:
    return child(window, QTabWidget, "workflow.forms")


def note_shown(window: QWidget, name: str, index: int) -> bool:
    """Whether the tab's note shows when a user opens that tab."""
    page = tabs(window).widget(index)
    assert page is not None
    note = child(window, QLabel, f"workflow.forms.{name}.note")
    return page.isAncestorOf(note) and note.isVisibleTo(page)


def shown_forms(window: QWidget, index: int) -> list[QWidget]:
    """Every form panel a user sees in that tab."""
    page = tabs(window).widget(index)
    assert page is not None
    return [
        widget
        for widget in page.findChildren(QWidget)
        if isinstance(widget, WorkflowFormPanel) and widget.isVisibleTo(page)
    ]


@dataclass(frozen=True)
class Seen:
    """What the Window showed at one moment of a run."""

    status: str
    badge: str
    stop: bool
    pause: bool
    resume: bool
    start: bool
    plan_editable: bool
    plan_note: bool
    tunables_title: str
    tunables_note: bool
    live_only: bool
    current_tab: int
    banner: bool


def see(window: WorkflowWindow[None]) -> Seen:
    tuning = window.tuning_panel
    shown = shown_forms(window, TUNABLES_TAB)
    return Seen(
        status=status_text(window),
        badge=badge(window),
        stop=button(window, "stop").isEnabled(),
        pause=button(window, "pause").isEnabled(),
        resume=button(window, "resume").isEnabled(),
        start=button(window, "start").isEnabled(),
        plan_editable=window.plan_panel.isEnabled(),
        plan_note=note_shown(window, "plan", PLAN_TAB),
        tunables_title=tabs(window).tabText(TUNABLES_TAB),
        tunables_note=note_shown(window, "tunables", TUNABLES_TAB),
        live_only=tuning is not None and shown == [tuning.panel],
        current_tab=tabs(window).currentIndex(),
        banner=banner(window).isVisibleTo(window),
    )


def bound(session: WindowSession, qapp: QApplication) -> WorkflowWindow[None]:
    """Bind the point and SoC, then show the Window at 1600x1000."""
    bind_point(session.harness, session.root)
    install_soc(session.harness)
    window = session.window
    window.resize(1600, 1000)
    window.show()
    qapp.processEvents()
    return window


def observe_at(
    session: WindowSession,
    qapp: QApplication,
    index: int,
    until: str,
    action: Callable[[], object] | None = None,
) -> list[Seen]:
    """At the wide point index, run action, wait for the until badge, then see."""
    seen: list[Seen] = []

    def hook(point: int) -> None:
        if point != index:
            return
        if action is not None:
            action()
        wait_for(qapp, lambda: badge(session.window) == until)
        seen.append(see(session.window))

    session.during_point = hook
    return seen


def records(window: WorkflowWindow[None], run_id: str) -> list[object]:
    engine = window.controller.engine
    assert engine is not None
    return [record.encoded_record.value for record in engine.records(run_id)]


def assert_start_tunables(window: WorkflowWindow[None]) -> None:
    """Terminal: the Start panels are editable again, with no live copy shown."""
    seen = see(window)
    assert seen.tunables_title == "Tunables"
    assert shown_forms(window, TUNABLES_TAB) == [window.tunables_panel]
    assert window.tunables_panel.isEnabled() and seen.plan_editable
    assert not seen.plan_note and not seen.tunables_note and not seen.banner
    assert seen.start and not seen.stop and not seen.pause and not seen.resume


def test_idle_status_is_ready_for_the_selection_and_stop_is_disabled(
    window_session: WindowSession, qapp: QApplication
) -> None:
    window = bound(window_session, qapp)
    assert status_text(window) == "ready synthetic_wide not started"
    seen = see(window)
    assert seen.start and not seen.stop and not seen.pause and not seen.resume
    assert seen.tunables_title == "Tunables" and seen.plan_editable
    assert not seen.plan_note and not seen.banner

    window.select_workflow("synthetic_small")
    assert status_text(window) == "ready synthetic_small not started"
    assert not button(window, "stop").isEnabled()


def test_pending_startup_keeps_stop_and_stop_ends_before_startup(
    window_session: WindowSession, qapp: QApplication
) -> None:
    window = bound(window_session, qapp)
    harness = window_session.harness
    assert isinstance(window.start(RUN_ID), int)

    pending = see(window)
    assert pending.status == f"pending synthetic_wide startup pending · run {RUN_ID}"
    assert pending.stop and not pending.pause and not pending.resume
    assert not pending.start and not pending.plan_editable and pending.plan_note
    button(window, "stop").click()
    assert button(window, "stop").isEnabled() and badge(window) == "pending"
    harness.complete_segment()
    qapp.processEvents()

    assert status_text(window) == (
        f"stopped synthetic_wide stopped before startup · run {RUN_ID}"
    )
    assert window.controller.status() is None
    assert not any((window_session.root / "result").glob("**/workflows/*"))
    assert_start_tunables(window)


def test_running_counts_commits_and_shows_only_the_live_tunables(
    window_session: WindowSession, qapp: QApplication
) -> None:
    window = bound(window_session, qapp)
    seen = observe_at(window_session, qapp, 1, "running")
    assert isinstance(window.start(RUN_ID), int)
    window_session.harness.complete_segment()
    qapp.processEvents()

    assert len(seen) == 1
    running = seen[0]
    assert running.status == f"running synthetic_wide 1 committed · run {RUN_ID}"
    assert running.stop and running.pause and not running.resume
    assert not running.start
    assert not running.plan_editable and running.plan_note
    assert running.tunables_title == "Tunables (live)" and running.tunables_note
    assert running.live_only, "the Start tunables must not be shown beside live ones"
    assert not running.banner

    assert status_text(window) == f"done synthetic_wide 3 committed · run {RUN_ID}"
    assert_start_tunables(window)
    assert window.tuning_panel is not None and not window.tuning_panel.isVisibleTo(
        window
    )


def test_pause_switches_to_live_tunables_and_banner_until_resume(
    window_session: WindowSession, qapp: QApplication
) -> None:
    window = bound(window_session, qapp)
    tabs(window).setCurrentIndex(PLAN_TAB)
    seen = observe_at(window_session, qapp, 1, "pausing", lambda: window.pause(RUN_ID))
    assert isinstance(window.start(RUN_ID), int)
    window_session.harness.complete_segment()
    qapp.processEvents()

    assert len(seen) == 1
    pausing = seen[0]
    assert pausing.status == f"pausing synthetic_wide 1 committed · run {RUN_ID}"
    assert pausing.stop and not pausing.pause and not pausing.resume
    assert pausing.current_tab == PLAN_TAB and not pausing.banner
    assert pausing.tunables_title == "Tunables (live)" and pausing.live_only

    paused = see(window)
    assert paused.status == f"paused synthetic_wide 1 committed · run {RUN_ID}"
    assert paused.current_tab == TUNABLES_TAB
    assert paused.tunables_title == "Tunables (live)" and paused.live_only
    assert paused.banner
    note = banner(window).text()
    assert "Resume" in note and "Apply" in note and "uncommitted" in note
    assert paused.stop and paused.resume and not paused.pause
    assert not paused.plan_editable and paused.plan_note

    window_session.during_point = None
    button(window, "resume").click()
    wait_for(qapp, lambda: not banner(window).isVisibleTo(window))
    window_session.harness.complete_segment()
    qapp.processEvents()
    assert status_text(window) == f"done synthetic_wide 3 committed · run {RUN_ID}"
    assert_start_tunables(window)


def test_paused_apply_redoes_the_point_and_terminal_keeps_accepted_input(
    window_session: WindowSession, qapp: QApplication
) -> None:
    window = bound(window_session, qapp)
    window_session.figure_points = [FigurePoint(("a/one",))] * 3
    window_session.figure_pauses = {1}
    window.select_workflow("synthetic_figures")
    assert isinstance(window.start(RUN_ID), int)
    window_session.harness.complete_segment()
    qapp.processEvents()
    tuning = window.tuning_panel
    assert tuning is not None and badge(window) == "paused"

    tuning.panel.state.write({("value",): 7}, tuning.host.env)
    child(window, QPushButton, "workflow.tuning.apply").click()
    assert child(window, QLabel, "workflow.tuning.message").text() == (
        "Applied revision 1"
    )
    button(window, "resume").click()
    window_session.harness.complete_segment()
    qapp.processEvents()

    assert records(window, RUN_ID) == [2, 7, 7], "the uncommitted point is redone"
    assert badge(window) == "done"
    assert_start_tunables(window)
    assert window.tunables_panel.state.input.data["value"] == 7
    assert window.tunables_panel.state.base.data["value"] == 7

    assert isinstance(window.start("again"), int)
    window_session.harness.complete_segment()
    assert records(window, "again") == [7, 7, 7]


def test_stopping_disables_stop_until_the_stopped_terminal(
    window_session: WindowSession, qapp: QApplication
) -> None:
    window = bound(window_session, qapp)
    seen = observe_at(window_session, qapp, 1, "stopping", lambda: window.stop(RUN_ID))
    assert isinstance(window.start(RUN_ID), int)
    window_session.harness.complete_segment()
    qapp.processEvents()

    assert len(seen) == 1
    stopping = seen[0]
    assert stopping.status == f"stopping synthetic_wide 1 committed · run {RUN_ID}"
    assert not stopping.stop and not stopping.pause and not stopping.resume
    assert not stopping.start and not stopping.plan_editable

    assert status_text(window) == f"stopped synthetic_wide 1 committed · run {RUN_ID}"
    assert_start_tunables(window)


def test_paused_stop_button_ends_the_run_as_stopped(
    window_session: WindowSession, qapp: QApplication
) -> None:
    window = bound(window_session, qapp)
    window_session.pause_at = 0
    assert isinstance(window.start(RUN_ID), int)
    window_session.harness.complete_segment()
    qapp.processEvents()
    assert badge(window) == "paused" and button(window, "stop").isEnabled()

    button(window, "stop").click()
    qapp.processEvents()
    assert status_text(window) == f"stopped synthetic_wide 0 committed · run {RUN_ID}"
    assert_start_tunables(window)


def test_failed_run_shows_failed_badge_and_its_reason_below(
    window_session: WindowSession, qapp: QApplication
) -> None:
    window = bound(window_session, qapp)

    def fail(index: int) -> None:
        if index == 1:
            raise RuntimeError("synthetic failure")

    window_session.during_point = fail
    assert isinstance(window.start(RUN_ID), int)
    window_session.harness.complete_segment()
    qapp.processEvents()

    assert status_text(window) == f"failed synthetic_wide 1 committed · run {RUN_ID}"
    assert "synthetic failure" in message(window)
    assert_start_tunables(window)


def test_start_errors_stay_below_the_status_through_refreshes(
    window_session: WindowSession, qapp: QApplication
) -> None:
    window = bound(window_session, qapp)
    window.select_workflow("synthetic_small")
    assert isinstance(window.start(RUN_ID), int)
    window_session.harness.complete_segment()
    qapp.processEvents()
    finished = f"done synthetic_small 0 committed · run {RUN_ID}"
    assert status_text(window) == finished

    window.plan_panel.state.write({("count",): 0}, window.plan_panel.env)
    button(window, "start").click()
    error = message(window)
    assert "count" in error.lower() or "greater" in error.lower()
    assert status_text(window) == finished

    window.select_workflow("synthetic_wide")
    window.select_workflow("synthetic_small")
    settle(qapp)
    assert message(window) == error
    assert status_text(window) == finished


def test_start_exception_stays_below_the_ready_status(
    window_session: WindowSession, qapp: QApplication
) -> None:
    bind_point(window_session.harness, window_session.root)
    window = window_session.window
    window.show()
    button(window, "start").click()
    settle(qapp)
    assert "SoC" in message(window)
    assert status_text(window) == "ready synthetic_wide not started"


def test_switching_workflow_marks_the_previous_run_until_next_start(
    window_session: WindowSession, qapp: QApplication
) -> None:
    window = bound(window_session, qapp)
    assert isinstance(window.start(RUN_ID), int)
    window_session.harness.complete_segment()
    qapp.processEvents()
    finished = f"done synthetic_wide 3 committed · run {RUN_ID}"
    assert status_text(window) == finished

    window.select_workflow("synthetic_small")
    assert status_text(window) == f"Previous run: {finished}"
    assert window.figure_names(RUN_ID) == ("synthetic/trend",)
    window.select_workflow("synthetic_wide")
    assert status_text(window) == finished

    window.select_workflow("synthetic_small")
    assert isinstance(window.start("small"), int)
    assert not status_text(window).startswith("Previous run:")
    window_session.harness.complete_segment()
    qapp.processEvents()
    assert status_text(window) == "done synthetic_small 0 committed · run small"


def area(window: QWidget, widget: QWidget) -> QRect:
    return QRect(widget.mapTo(window, QPoint(0, 0)), widget.size())


def test_paused_live_tunables_stay_in_their_tab_within_800(
    window_session: WindowSession, qapp: QApplication
) -> None:
    window = bound(window_session, qapp)
    expand_all(window.plan_panel, qapp)
    expand_all(window.tunables_panel, qapp)
    window_session.pause_at = 1
    assert isinstance(window.start(RUN_ID), int)
    window_session.harness.complete_segment()
    qapp.processEvents()
    tuning = window.tuning_panel
    assert tuning is not None and badge(window) == "paused"
    expand_all(tuning, qapp)
    assert window.minimumSizeHint().height() <= 800

    page = tabs(window).widget(TUNABLES_TAB)
    assert page is not None and page.isAncestorOf(tuning)
    scroll = tuning.parentWidget()
    while scroll is not None and not isinstance(scroll, QScrollArea):
        scroll = scroll.parentWidget()
    assert scroll is not None and page.isAncestorOf(scroll)
    for name in ("workflow.tuning.apply", "workflow.tuning.discard"):
        assert page.isAncestorOf(child(window, QPushButton, name))

    window.resize(1600, 1000)
    qapp.processEvents()
    bar = child(window, QWidget, "workflow.run.status")
    figures = child(window, QStackedWidget, "workflow.figures")
    for widget in (bar, banner(window), child(window, QLabel, "workflow.run.message")):
        assert widget.isVisible()
        assert area(window, widget).bottom() <= area(window, figures).top()
    status_row = child(window, QLabel, "workflow.run.badge")
    assert area(window, status_row).bottom() <= area(window, banner(window)).top()
    assert figures.height() >= 600
