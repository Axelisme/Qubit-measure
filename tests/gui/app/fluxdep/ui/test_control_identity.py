"""Fluxdep Qt write admission and viewing under application control authority."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from qtpy.QtCore import QEvent, Qt, QTimer
from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QGroupBox,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTabWidget,
)
from zcu_tools.analysis.fluxdep.models import TransitionDict
from zcu_tools.analysis.fluxdep.search import SearchCancelled
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.search import FluxDepSearchRuntime
from zcu_tools.gui.app.fluxdep.ui.analyze_panel import AnalyzePanelWidget
from zcu_tools.gui.app.fluxdep.ui.interactive.line_picker import LinePickerWidget
from zcu_tools.gui.app.fluxdep.ui.interactive.selector import SelectorWidget
from zcu_tools.gui.app.fluxdep.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import EventOrigin
from zcu_tools.gui.session.adapters.qt_progress_transport import QtProgressTransport
from zcu_tools.gui.session.services.progress import ProgressService


def button(widget, text):
    return next(b for b in widget.findChildren(QPushButton) if b.text() == text)


def indicator(window):
    found = window.findChild(QPushButton, "control_indicator")
    assert found is not None, "Fluxdep must display the shared control indicator"
    return found


@pytest.fixture
def window(qapp, cross_controller):
    view = MainWindow(cross_controller)
    view.show()
    yield view, cross_controller
    cross_controller.control_identity.reclaim_local()
    view.close()
    view.deleteLater()
    qapp.processEvents()


def test_indicator_projects_online_offline_reclaim_lock_and_unlock(window):
    view, ctrl = window
    lamp = indicator(view)
    assert "Local" in lamp.text()
    ctrl.control_identity.connect("a", "flux-agent", "owner")
    assert "Remote" in lamp.text() and "flux-agent" in lamp.text()
    assert "online" in lamp.text().lower()
    ctrl.control_identity.disconnect("a")
    assert "offline" in lamp.text().lower()
    lamp.click()
    assert ctrl.control_identity.status().holder == "gui"
    assert ctrl.control_identity.status().locked
    assert "locked" in lamp.text().lower()
    lamp.click()
    assert not ctrl.control_identity.status().locked
    ctrl.control_identity.connect("b", "flux-agent", "owner")
    assert "Remote" in lamp.text()
    ctrl.control_identity.release("b")
    assert "Local" in lamp.text() and "locked" not in lamp.text().lower()


def test_remote_blocks_spectrum_writes_but_keeps_result_canvas_available(window):
    view, ctrl = window
    ctrl.control_identity.connect("a", "flux-agent", "owner")
    for text in (
        "Project…",
        "Add…",
        "Remove",
        "Restore",
        "Export",
        "Analyze…",
        "Re-pick lines",
        "Re-select points",
    ):
        assert not button(view, text).isEnabled(), text
    spectra = view.findChild(QListWidget)
    assert spectra is not None and not spectra.isEnabled()
    canvas = view.findChild(FigureCanvasQTAgg)
    assert canvas is not None and canvas.isEnabled()
    assert view.centralWidget().isEnabled()
    button(view, "Remove").click()
    assert "a" in ctrl.state.spectrums
    indicator(view).click()
    assert spectra.isEnabled()
    button(view, "Remove").click()
    assert "a" not in ctrl.state.spectrums


def test_remote_gates_open_dialog_and_restores_its_fields(window):
    view, ctrl = window
    button(view, "Project…").click()
    dialogs = view.findChildren(QDialog)
    assert len(dialogs) == 1
    dialog = dialogs[0]
    edits = dialog.findChildren(QLineEdit)
    assert edits and all(edit.isEnabled() for edit in edits)
    ctrl.control_identity.connect("a", "flux-agent", "owner")
    assert all(not edit.isEnabled() for edit in edits)
    indicator(view).click()
    assert all(edit.isEnabled() for edit in edits)
    dialog.reject()


def test_remote_filter_switch_cannot_cancel_until_reclaim(window):
    view, ctrl = window
    button(view, "Analyze…").click()
    panel = view.findChild(AnalyzePanelWidget)
    assert panel is not None
    active = ctrl.interactive.inspect()
    assert active is not None and active.kind == "selection"
    tabs = panel.findChild(QTabWidget)
    assert tabs is not None
    ctrl.control_identity.connect("a", "flux-agent", "owner")
    assert not tabs.tabBar().isEnabled()
    selector = panel.findChild(SelectorWidget)
    assert selector is not None
    assert not button(selector, "Apply").isEnabled()
    assert ctrl.interactive.inspect() == active
    indicator(view).click()
    assert tabs.tabBar().isEnabled()
    tabs.setCurrentIndex(1)
    assert ctrl.interactive.inspect() is None


def test_remote_filter_keyboard_navigation_cannot_cancel_session(window):
    view, ctrl = window
    button(view, "Analyze…").click()
    panel = view.findChild(AnalyzePanelWidget)
    assert panel is not None
    tabs = panel.findChild(QTabWidget)
    assert tabs is not None
    active = ctrl.interactive.inspect()
    assert active is not None and active.kind == "selection"
    ctrl.control_identity.connect("a", "flux-agent", "owner")
    QApplication.sendEvent(
        tabs,
        QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Tab, Qt.KeyboardModifier.ControlModifier
        ),
    )
    assert panel.current_tab == "filter"
    assert ctrl.interactive.inspect() == active
    assert ctrl.control_identity.status().holder == "rpc:flux-agent"
    indicator(view).click()
    QApplication.sendEvent(
        tabs,
        QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Tab, Qt.KeyboardModifier.ControlModifier
        ),
    )
    assert panel.current_tab == "search"
    assert ctrl.interactive.inspect() is None


def test_agent_mounts_new_remote_picker_without_enabling_controls_or_pointer(window):
    view, ctrl = window
    ctrl.control_identity.connect("a", "flux-agent", "owner")
    with ctrl.bus.origin(EventOrigin(kind="agent")):
        context = ctrl.interactive.begin_line_pick("a")
    picker = view.findChild(LinePickerWidget)
    assert picker is not None
    assert not button(picker, "Swap Lines").isEnabled()
    assert picker.canvas.isEnabled()
    before = context.session.snapshot()
    picker.canvas.draw()
    axes = picker.figure.axes[0]
    for kind, x in (
        ("button_press_event", before.flux_half),
        ("motion_notify_event", 0.2),
        ("button_release_event", 0.2),
    ):
        px, py = axes.transData.transform((x, sum(axes.get_ylim()) / 2))
        event = MouseEvent(kind, picker.canvas, px, py, button=MouseButton.LEFT)
        picker.canvas.callbacks.process(kind, event)
    assert context.session.snapshot() == before
    indicator(view).click()
    button(picker, "Swap Lines").click()
    assert context.session.snapshot().flux_half == before.flux_int


class DeferredSearch:
    """Hold a worker until the test requests native cooperative cancellation."""

    def __init__(self):
        self.work: Callable[[], object] | None = None
        self.error: Callable[[Exception], None] | None = None

    def submit(self, work, *, run_in_pool, on_done, on_error):
        self.work, self.error = work, on_error

    def complete_cancel(self):
        assert self.work is not None and self.error is not None
        with (
            ThreadPoolExecutor(max_workers=1) as pool,
            pytest.raises(SearchCancelled) as caught,
        ):
            pool.submit(self.work).result()
        self.error(caught.value)


@pytest.fixture
def pending_window(qapp, cross_controller, tmp_path):
    background = DeferredSearch()
    ctrl = Controller(
        cross_controller.state,
        search_runtime=FluxDepSearchRuntime(
            background, ProgressService(QtProgressTransport())
        ),
    )
    path = tmp_path / "db.h5"
    path.touch()
    ctrl.set_fit_params(
        str(path),
        (2, 15),
        (0.2, 2),
        (0.1, 2),
        TransitionDict({"transitions": [(0, 1)]}),
        None,
        None,
    )
    view = MainWindow(ctrl)
    view.show()
    button(view, "Analyze…").click()
    panel = view.findChild(AnalyzePanelWidget)
    assert panel is not None
    panel.show_tab("search")
    token = ctrl.search.start()
    yield view, ctrl, panel, background, token
    ctrl.control_identity.reclaim_local()
    if ctrl.search.active_token is not None:
        ctrl.search.cancel(token)
        background.complete_cancel()
    view.close()
    view.deleteLater()
    qapp.processEvents()


def test_remote_search_cancel_disabled_reclaim_restores_domain_gate(pending_window):
    view, ctrl, panel, background, token = pending_window
    cancel = button(panel, "Cancel search")
    assert cancel.isEnabled()
    ctrl.control_identity.connect("a", "flux-agent", "owner")
    assert not cancel.isEnabled()
    form = next(
        g for g in panel.findChildren(QGroupBox) if g.title() == "Search parameters"
    )
    assert not form.isEnabled()
    tabs = panel.findChild(QTabWidget)
    assert tabs is not None and tabs.tabBar().isTabEnabled(2)
    tabs.setCurrentIndex(2)
    assert panel.current_tab == "show"
    assert next(
        g for g in panel.findChildren(QGroupBox) if g.title() == "Display"
    ).isEnabled()
    cancel.click()
    assert ctrl.search.outcome(token) is None
    assert ctrl.control_identity.status().holder == "rpc:flux-agent"
    indicator(view).click()
    assert cancel.isEnabled() and not button(panel, "Search database").isEnabled()
    cancel.click()
    background.complete_cancel()
    outcome = ctrl.search.outcome(token)
    assert outcome is not None and outcome.status == "cancelled"
    assert not cancel.isEnabled() and button(panel, "Search database").isEnabled()


def answer_close(answer, seen):
    def respond():
        box = QApplication.activeModalWidget()
        assert isinstance(box, QMessageBox)
        seen.append(box.text())
        choice = (
            QMessageBox.StandardButton.Yes if answer else QMessageBox.StandardButton.No
        )
        response_button = box.button(choice)
        assert response_button is not None
        response_button.click()

    QTimer.singleShot(0, respond)


def test_remote_close_decline_has_no_lifecycle_effect_accept_waits_for_search(
    pending_window,
):
    view, ctrl, panel, background, token = pending_window
    ctrl.control_identity.connect("a", "flux-agent", "owner")
    seen = []
    indicator(view)
    answer_close(False, seen)
    assert not view.close(), "Remote close must ask before beginning shutdown"
    assert seen and "flux-agent" in seen[0]
    assert view.isVisible() and ctrl.search.active_token == token
    answer_close(True, seen)
    assert not view.close(), "Accepted close must retain the pending search owner"
    assert ctrl.control_identity.status().holder == "rpc:flux-agent"
    background.complete_cancel()
    assert ctrl.search.outcome(token).status == "cancelled"
    assert panel.figures == {}
    answer_close(True, seen)
    assert view.close()
