"""Fluxdep Qt write admission and viewing under application control authority."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QToolBar,
    QWidget,
)
from zcu_tools.analysis.fluxdep.models import TransitionDict
from zcu_tools.analysis.fluxdep.search import ParamBounds, SearchCancelled
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.search import FluxDepSearchRuntime
from zcu_tools.gui.app.fluxdep.ui.interactive.line_picker import LinePickerWidget
from zcu_tools.gui.app.fluxdep.ui.interactive.selector import SelectorWidget
from zcu_tools.gui.app.fluxdep.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import EventOrigin
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.gui.session.adapters.qt_progress_transport import QtProgressTransport
from zcu_tools.gui.session.services.progress import ProgressService

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui.app.fluxdep._search_result import (
    perfect_fit_result,
    synthetic_search_input,
)
from tests.gui.app.fluxdep._window import (
    choose_file,
    command,
    current_page,
    fit_panel,
    named,
    open_chooser,
    picker_button,
    show_canvas,
    show_page,
    spectra_list,
    stage_button,
)
from tests.gui.app.fluxdep.remote._route_harness import RouteHarness


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
    for name in (
        "fluxdep.project",
        "fluxdep.spectrum.add",
        "fluxdep.spectrum.remove",
        "fluxdep.spectra.restore",
        "fluxdep.spectra.export",
    ):
        assert not command(view, name).isEnabled(), name
    # Stages stay clickable for viewing, but a non-holder click opens no input.
    for stage in ("align", "sample"):
        stage_button(view, stage).click()
        assert ctrl.interactive.inspect("a") is None, stage
    spectra = spectra_list(view)
    assert spectra.isEnabled()
    canvas = view.findChild(FigureCanvasQTAgg)
    assert canvas is not None and canvas.isEnabled()
    assert view.centralWidget().isEnabled()
    command(view, "fluxdep.spectrum.remove").click()
    assert "a" in ctrl.state.spectrums
    indicator(view).click()
    assert spectra.isEnabled()
    command(view, "fluxdep.spectrum.remove").click()
    assert "a" not in ctrl.state.spectrums


def _record_stale_fit(ctrl, search_db):
    ctrl.set_fit_params(
        search_db,
        (3.0, 12.0),
        (0.3, 1.5),
        (0.2, 1.2),
        TransitionDict({"transitions": [(0, 1)]}),
        None,
        None,
    )
    bounds = ParamBounds(EJ=(3.0, 12.0), EC=(0.3, 1.5), EL=(0.2, 1.2))
    fluxs, freqs = ctrl.selected_pointcloud()
    ctrl.record_search_result(
        perfect_fit_result(fluxs, freqs, bounds),
        synthetic_search_input(ctrl.state, fluxs, freqs, bounds),
    )
    ctrl.set_selection(np.array([True, False, True, True]))
    assert ctrl.state.fit_stale_reasons == ["filter changed"]


def test_remote_viewer_reads_fit_page_without_opening_filter(
    window, search_database_hdf5
):
    view, ctrl = window
    _record_stale_fit(ctrl, search_database_hdf5)
    ctrl.control_identity.connect("a", "flux-agent", "owner")
    show_page(view, "fit")
    panel = fit_panel(view)
    assert panel is not None
    assert ctrl.interactive.current_cross_selection() is None
    freshness = named(panel, QLabel, "fluxdep.fit.freshness")
    assert freshness.text() == "Outdated: filter changed"
    form = next(
        g for g in panel.findChildren(QGroupBox) if g.title() == "Search parameters"
    )
    assert not form.isEnabled()
    tabs = panel.findChild(QTabWidget)
    assert tabs is not None and not tabs.isTabEnabled(0)
    panel.show_tab("show")
    title = show_canvas(panel).figure.axes[0].get_title()
    assert "Outdated: filter changed" in title
    assert ctrl.interactive.current_cross_selection() is None
    assert ctrl.control_identity.status().holder == "rpc:flux-agent"


def _show_display_inputs(panel) -> list[QWidget]:
    """Every input of Show's Display group: limits, Auto limits, lines, subset, Apply."""
    display = next(g for g in panel.findChildren(QGroupBox) if g.title() == "Display")
    return [
        control
        for control in display.findChildren(QWidget)
        if isinstance(control, (QDoubleSpinBox, QCheckBox, QLineEdit, QPushButton))
    ]


def test_remote_viewer_show_display_is_disabled_but_toolbar_zooms_until_reclaim(
    window, search_database_hdf5
):
    view, ctrl = window
    _record_stale_fit(ctrl, search_database_hdf5)
    show_page(view, "fit")
    panel = fit_panel(view)
    assert panel is not None
    panel.show_tab("show")
    inputs = _show_display_inputs(panel)
    kinds = {type(control) for control in inputs}
    assert {QDoubleSpinBox, QCheckBox, QLineEdit, QPushButton} <= kinds
    assert {"fluxdep.show.apply", "fluxdep.show.auto_limits"} <= {
        control.objectName() for control in inputs
    }
    assert all(control.isEnabled() for control in inputs)

    ctrl.control_identity.connect("a", "flux-agent", "owner")
    still_enabled = [control for control in inputs if control.isEnabled()]
    assert still_enabled == [], "a non-holder must not edit Show's display choices"
    # The toolbar only changes this viewer's own view, so it stays usable.
    toolbar = named(panel, QToolBar, "fluxdep.show.toolbar")
    assert toolbar.isEnabled()
    tools = {action.text(): action for action in toolbar.actions() if action.text()}
    # Back/Forward follow matplotlib's view history, not control authority.
    assert all(tools[name].isEnabled() for name in ("Home", "Pan", "Zoom"))

    indicator(view).click()
    assert all(control.isEnabled() for control in inputs)
    assert toolbar.isEnabled()


def test_remote_page_switch_keeps_agent_filter_session(window):
    view, ctrl = window
    ctrl.control_identity.connect("a", "flux-agent", "owner")
    with ctrl.bus.origin(EventOrigin(kind="agent")):
        context = ctrl.interactive.begin_cross_selection()
    # The agent's Filter waits on the Fit page; the viewer stays on Spectra.
    assert current_page(view) == "spectra"
    show_page(view, "fit")
    panel = fit_panel(view)
    assert panel is not None and panel.current_tab == "filter"
    assert panel.findChild(SelectorWidget) is not None
    active = ctrl.interactive.inspect_selection()
    assert active is not None and active.context is context
    show_page(view, "spectra")
    # Leaving Fit retires the Filter view (deferred Qt deletion) but not its input.
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert panel.findChild(SelectorWidget) is None
    assert ctrl.interactive.inspect_selection() == active
    show_page(view, "fit")
    assert panel.current_tab == "filter"
    assert panel.findChild(SelectorWidget) is not None
    assert ctrl.interactive.inspect_selection() == active


def test_ended_agent_filter_no_longer_routes_the_next_fit_entry(window):
    view, ctrl = window
    show_page(view, "fit")
    panel = fit_panel(view)
    assert panel is not None
    panel.show_tab("search")
    show_page(view, "spectra")
    ctrl.control_identity.connect("a", "flux-agent", "owner")
    with ctrl.bus.origin(EventOrigin(kind="agent")):
        ctrl.interactive.begin_cross_selection()
        ctrl.interactive.cancel_selection()
    indicator(view).click()
    assert ctrl.control_identity.status().holder == "gui"
    show_page(view, "fit")
    # The agent's Filter has ended, so its event no longer picks the tab.
    assert panel.current_tab == "search"
    assert ctrl.interactive.current_cross_selection() is None


@pytest.mark.parametrize("owner", ["load", "export", "result", "database"])
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
    name = {"load": "fluxdep.spectrum.add", "export": "fluxdep.spectra.export"}
    command(view, name.get(owner, "fluxdep.project")).click()
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
    container.reject()


@pytest.mark.parametrize("cancel", [False, True])
def test_search_db_chooser_is_non_blocking_and_holds_local_while_open(
    qapp, tmp_path, search_database_hdf5, cancel
):
    harness = RouteHarness.create(str(tmp_path))
    view = MainWindow(harness.ctrl)
    view.show()
    try:
        show_page(view, "fit")
        panel = fit_panel(view)
        assert panel is not None
        panel.show_tab("search")
        field = named(panel, QLineEdit, "fluxdep.fit.search_db")
        before = field.text()
        browse = button(panel, "Browse…")
        chooser = open_chooser(browse.click)
        # An open Local chooser refuses an agent's owner connection.
        link = ClientLink("agent", token_required=False)
        harness.service.on_client_open(link)
        harness.links.append(link)
        refused = harness.request(link, "control.connect", label="agent", mode="owner")
        assert refused["ok"] is False
        assert refused["error"].get("reason") == "local_busy"
        assert harness.ctrl.control_identity.status().holder == "gui"
        if cancel:
            chooser.reject()
            assert field.text() == before
        else:
            chooser.selectFile(search_database_hdf5)
            chooser.accept()
            assert field.text() == search_database_hdf5
        assert QApplication.activeModalWidget() is None
        accepted = harness.request(link, "control.connect", label="agent", mode="owner")
        assert accepted["ok"], accepted
    finally:
        harness.ctrl.control_identity.reclaim_local()
        view.close()
        view.deleteLater()
        qapp.processEvents()
        harness.close()


def _selector_count(panel) -> int:
    """Selector views under panel once deferred Qt deletions have run."""
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    return len(panel.findChildren(SelectorWidget))


def test_remote_filter_switch_detaches_and_cannot_cancel_until_reclaim(window):
    view, ctrl = window
    show_page(view, "fit")
    panel = fit_panel(view)
    assert panel is not None
    active = ctrl.interactive.inspect()
    assert active is not None and active.kind == "selection"
    tabs = panel.findChild(QTabWidget)
    assert tabs is not None
    ctrl.control_identity.connect("a", "flux-agent", "owner")
    selector = panel.findChild(SelectorWidget)
    assert selector is not None
    assert not button(selector, "Apply").isEnabled()
    # A non-holder leaving Filter only retires the view; the Session stays.
    tabs.setCurrentIndex(1)
    assert panel.current_tab == "search"
    assert _selector_count(panel) == 0
    assert ctrl.interactive.inspect() == active
    # Coming back reattaches the same Session, still read-only.
    tabs.setCurrentIndex(0)
    assert panel.current_tab == "filter"
    assert _selector_count(panel) == 1
    selector = panel.findChild(SelectorWidget)
    assert selector is not None and not button(selector, "Apply").isEnabled()
    assert ctrl.interactive.inspect() == active
    indicator(view).click()
    tabs.setCurrentIndex(1)
    assert ctrl.interactive.inspect() is None


def test_remote_filter_keyboard_navigation_cannot_cancel_session(window):
    view, ctrl = window
    show_page(view, "fit")
    panel = fit_panel(view)
    assert panel is not None
    tabs = panel.findChild(QTabWidget)
    assert tabs is not None
    active = ctrl.interactive.inspect()
    assert active is not None and active.kind == "selection"
    ctrl.control_identity.connect("a", "flux-agent", "owner")
    next_tab = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Tab, Qt.KeyboardModifier.ControlModifier
    )
    QApplication.sendEvent(tabs, next_tab)
    assert panel.current_tab == "search"
    assert ctrl.interactive.inspect() == active
    assert ctrl.control_identity.status().holder == "rpc:flux-agent"
    indicator(view).click()
    panel.show_tab("filter")
    assert ctrl.interactive.inspect() == active
    QApplication.sendEvent(tabs, next_tab)
    assert panel.current_tab == "search"
    assert ctrl.interactive.inspect() is None


def test_remote_viewer_reads_search_while_the_agent_filter_runs(
    window, search_database_hdf5
):
    view, ctrl = window
    _record_stale_fit(ctrl, search_database_hdf5)
    ctrl.control_identity.connect("a", "flux-agent", "owner")
    with ctrl.bus.origin(EventOrigin(kind="agent")):
        context = ctrl.interactive.begin_cross_selection()
    show_page(view, "fit")
    panel = fit_panel(view)
    assert panel is not None and panel.current_tab == "filter"
    tabs = panel.findChild(QTabWidget)
    assert tabs is not None
    tabs.setCurrentIndex(1)
    assert panel.current_tab == "search"
    freshness = named(panel, QLabel, "fluxdep.fit.freshness")
    assert freshness.text().startswith("Outdated: ")
    active = ctrl.interactive.inspect_selection()
    assert active is not None and active.context is context
    tabs.setCurrentIndex(2)
    assert panel.current_tab == "show"
    assert tabs.isTabEnabled(0), "the live agent Filter can be viewed again"
    tabs.setCurrentIndex(0)
    assert panel.current_tab == "filter"
    assert _selector_count(panel) == 1
    assert ctrl.interactive.inspect_selection() == active
    tabs.setCurrentIndex(1)
    with ctrl.bus.origin(EventOrigin(kind="agent")):
        ctrl.interactive.cancel_selection()
    # Without a live Filter a viewer cannot enter Filter, which would begin one.
    assert not tabs.isTabEnabled(0)
    assert ctrl.interactive.current_cross_selection() is None


def test_agent_mounts_new_remote_picker_without_enabling_controls_or_pointer(window):
    view, ctrl = window
    ctrl.control_identity.connect("a", "flux-agent", "owner")
    with ctrl.bus.origin(EventOrigin(kind="agent")):
        context = ctrl.interactive.begin_line_pick("a")
    picker = view.findChild(LinePickerWidget)
    assert picker is not None
    assert not picker_button(picker, "swap").isEnabled()
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
    picker_button(picker, "swap").click()
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
    show_page(view, "fit")
    panel = fit_panel(view)
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
    cancel = named(panel, QPushButton, "fluxdep.fit.cancel")
    search = named(panel, QPushButton, "fluxdep.fit.search")
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
    assert not next(
        g for g in panel.findChildren(QGroupBox) if g.title() == "Display"
    ).isEnabled()
    cancel.click()
    assert ctrl.search.outcome(token) is None
    assert ctrl.control_identity.status().holder == "rpc:flux-agent"
    indicator(view).click()
    assert cancel.isEnabled() and not search.isEnabled()
    cancel.click()
    background.complete_cancel()
    outcome = ctrl.search.outcome(token)
    assert outcome is not None and outcome.status == "cancelled"
    assert not cancel.isEnabled() and search.isEnabled()


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
