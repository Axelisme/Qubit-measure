"""Fluxdep Qt write admission and viewing under application control authority."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from qtpy.QtCore import QEvent, QMetaObject, Qt, QTimer
from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QGroupBox,
    QLineEdit,
    QListWidget,
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
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.adapters.qt_progress_transport import QtProgressTransport
from zcu_tools.gui.session.services.progress import ProgressService

from tests.gui._dialog_fakes import RecordingDialogPresenter


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


def test_closed_window_stops_indicator_authority_projection(window):
    view, ctrl = window
    display = indicator(view)
    assert display.text() == "Local"
    assert view.close()

    ctrl.control_identity.connect("retired-view", "flux-agent", "owner")

    assert display.text() == "Local"


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


def choose_file(path, activate, *, cancel=False):
    """Accept or cancel the real Local file chooser."""
    errors = []
    seen = []

    def respond():
        chooser = QApplication.activeModalWidget()
        try:
            assert isinstance(chooser, QFileDialog)
            seen.append(chooser.windowTitle())
            chooser.selectFile(str(path))
            QMetaObject.invokeMethod(
                chooser,
                "reject" if cancel else "accept",
                Qt.ConnectionType.DirectConnection,
            )
        except (AssertionError, RuntimeError, TypeError) as exc:
            errors.append(exc)
            if isinstance(chooser, QDialog):
                chooser.reject()

    def abort():
        chooser = QApplication.activeModalWidget()
        errors.append(AssertionError("Static chooser did not finish within 2 seconds"))
        if isinstance(chooser, QDialog):
            chooser.reject()

    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(abort)
    timeout.start(2000)
    QTimer.singleShot(0, respond)
    activate()
    timeout.stop()
    if errors:
        raise errors[0]
    assert seen, "The action must present a real modal file chooser"


@pytest.mark.parametrize("owner", ["load", "export", "result", "database", "search"])
@pytest.mark.parametrize("cancel", [False, True])
def test_local_chooser_preserves_path_and_cancel_contract(
    window, tmp_path, spectrum_hdf5, owner, cancel
):
    view, ctrl = window
    target = tmp_path / "chosen.hdf5"
    if owner in ("result", "database"):
        target = tmp_path
    elif owner != "export":
        target = Path(spectrum_hdf5[0])
    if owner == "search":
        button(view, "Analyze…").click()
        panel = view.findChild(AnalyzePanelWidget)
        assert panel is not None
        panel.show_tab("search")
        container = panel
    else:
        command = {"load": "Add…", "export": "Export"}.get(owner, "Project…")
        button(view, command).click()
        container = QApplication.activeModalWidget()
        assert isinstance(container, QDialog)
    edits = container.findChildren(QLineEdit)
    before = [edit.text() for edit in edits]
    browse_buttons = [
        b for b in container.findChildren(QPushButton) if b.text() == "Browse…"
    ]
    choose_file(
        target,
        browse_buttons[1 if owner == "database" else 0].click,
        cancel=cancel,
    )
    after = [edit.text() for edit in edits]
    if cancel:
        assert after == before
    else:
        assert str(target) in after
        if owner == "load":
            assert button(container, "OK").isEnabled()
    assert target.exists() == (owner != "export")
    assert ctrl.control_identity.status().holder == "gui"
    if isinstance(container, QDialog):
        container.reject()


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
def pending_window(qapp, cross_controller, search_database_hdf5):
    background = DeferredSearch()
    ctrl = Controller(
        cross_controller.state,
        search_runtime=FluxDepSearchRuntime(
            background, ProgressService(QtProgressTransport())
        ),
    )
    ctrl.set_fit_params(
        search_database_hdf5,
        (2, 15),
        (0.2, 2),
        (0.1, 2),
        TransitionDict({"transitions": [(0, 1)]}),
        None,
        None,
    )
    dialogs = RecordingDialogPresenter(
        confirm_answers=[False, False], defer_confirm=True
    )
    view = MainWindow(ctrl, dialog_presenter=dialogs)
    view.show()
    button(view, "Analyze…").click()
    panel = view.findChild(AnalyzePanelWidget)
    assert panel is not None
    panel.show_tab("search")
    token = ctrl.search.start()
    yield view, ctrl, panel, background, token, dialogs
    ctrl.control_identity.reclaim_local()
    if ctrl.search.active_token is not None:
        ctrl.search.cancel(token)
        background.complete_cancel()
    view.close()
    view.deleteLater()
    qapp.processEvents()


def test_remote_search_cancel_disabled_reclaim_restores_domain_gate(pending_window):
    view, ctrl, panel, background, token, _dialogs = pending_window
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


def test_remote_close_decline_has_no_lifecycle_effect_accept_waits_for_search(
    pending_window,
):
    view, ctrl, panel, background, token, dialogs = pending_window
    ctrl.control_identity.connect("a", "flux-agent", "owner")
    assert not view.close(), "Remote close must return while the decision is pending"
    assert len(dialogs.calls) == 1 and "flux-agent" in dialogs.calls[0].message
    assert not view.close(), "A pending decision must not open another confirmation"
    assert len(dialogs.calls) == 1
    # Search admission stays open until acceptance, not merely close intent.
    with pytest.raises(FailedPreconditionError) as busy:
        ctrl.search.start()
    assert busy.value.reason_code == "search_busy"
    assert view.isVisible() and ctrl.search.active_token == token
    dialogs.decide_confirm(False)
    with pytest.raises(FailedPreconditionError) as still_busy:
        ctrl.search.start()
    assert still_busy.value.reason_code == "search_busy"
    assert view.isVisible()
    ctrl.control_identity.disconnect("a")
    assert indicator(view).text().endswith("offline")
    ctrl.control_identity.connect("a", "flux-agent", "owner")
    assert indicator(view).text().endswith("online")
    assert not view.close()
    assert len(dialogs.calls) == 2
    dialogs.decide_confirm(True)
    assert view.isVisible(), "Accepted close must retain the pending search owner"
    assert ctrl.control_identity.status().holder == "rpc:flux-agent"
    ctrl.control_identity.disconnect("a")
    assert indicator(view).text().endswith("offline")
    ctrl.control_identity.connect("a", "flux-agent", "owner")
    assert indicator(view).text().endswith("online")
    assert not view.close(), "Drain refusal must not ask for confirmation again"
    assert len(dialogs.calls) == 2
    background.complete_cancel()
    assert ctrl.search.outcome(token).status == "cancelled"
    assert panel.figures == {}
    assert view.close()
    assert len(dialogs.calls) == 2
