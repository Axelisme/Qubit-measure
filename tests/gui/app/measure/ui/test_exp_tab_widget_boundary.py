"""Behavior tests for the MainWindow-to-ExpTabWidget view boundary."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import pytest
from matplotlib.figure import Figure
from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtWidgets import (
    QLineEdit,
    QPushButton,
    QTabWidget,
    QWidget,
)
from zcu_tools.experiment.artifacts import ArtifactRef
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    ParameterWriteback,
    ParamMeta,
)
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.services import create_persistence_caretaker
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.measure.ui.interactive_frontend import InteractiveFrontend
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.plotting.figures import FigureCollection
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._native_save_support import (
    NativeSaveSession,
    PaneParams,
    PaneSummary,
)
from tests.gui.app.measure.ui._main_window_fakes import (
    editor_wiring_ctrl,
    make_tab_snapshot,
    mock_ctrl,
    tab_pane_tabs,
)
from tests.gui.widgets.form._text_input import commit_text


class _RecordingTabActions:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str] | tuple[str, str, ArtifactRef]] = []

    def refresh_interaction(self, tab_id: str) -> None:
        self.calls.append(("refresh_interaction", tab_id))

    def run_or_stop(self, tab_id: str) -> None:
        self.calls.append(("run_or_stop", tab_id))

    def load_data(self, tab_id: str) -> None:
        self.calls.append(("load_data", tab_id))

    def analyze(self, tab_id: str) -> None:
        self.calls.append(("analyze", tab_id))

    def post_analyze(self, tab_id: str) -> None:
        self.calls.append(("post_analyze", tab_id))

    def apply_writeback(self, tab_id: str) -> None:
        self.calls.append(("apply_writeback", tab_id))

    def apply_post_writeback(self, tab_id: str) -> None:
        self.calls.append(("apply_post_writeback", tab_id))

    def save_artifact(self, tab_id: str, ref: ArtifactRef) -> None:
        self.calls.append(("save_artifact", tab_id, ref))

    def save_all(self, tab_id: str) -> None:
        self.calls.append(("save_all", tab_id))


def test_exp_tab_disables_local_buttons_while_analyzing(
    main_window_control_factory, qapp
):
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        mock_ctrl(main_window_control_factory),
        AdapterCapabilities(
            analysis=AnalysisMode.FIT, post_analysis=False, load_data=True
        ),
    )
    tab.update_writeback_items(
        [ParameterWriteback("r_f", "freq", proposed_value=6100.0)]
    )
    tab.update_interaction_state(
        make_tab_snapshot(
            "tab-1",
            global_run_active=False,
            is_running=False,
            is_analyzing=True,
            has_context=True,
            has_active_context=True,
            has_soc=True,
            has_run_result=True,
            has_analyze_result=True,
            has_figure=True,
        )
    )

    assert tab.analyze_btn.isEnabled() is False
    load = next(
        button
        for button in tab.findChildren(QPushButton)
        if button.text() == "Load Data"
    )
    assert load.isHidden() is False
    assert load.isEnabled() is False
    assert tab.writeback_widget.isEnabled() is False
    saves = [
        button for button in tab.findChildren(QPushButton) if button.text() == "Save"
    ]
    assert len(saves) == 2
    assert all(not button.isEnabled() for button in saves)


def test_exp_tab_keeps_analyze_enabled_while_other_tab_running(
    main_window_control_factory, qapp
):
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        mock_ctrl(main_window_control_factory),
        AdapterCapabilities(
            analysis=AnalysisMode.FIT, post_analysis=False, load_data=True
        ),
    )
    tab.update_interaction_state(
        make_tab_snapshot(
            "tab-1",
            global_run_active=True,
            is_running=False,
            is_analyzing=False,
            has_context=True,
            has_active_context=True,
            has_soc=True,
            has_run_result=True,
            has_analyze_result=True,
            has_figure=True,
        )
    )

    assert tab.run_btn.isEnabled() is False
    assert tab.analyze_btn.isEnabled() is True
    saves = [
        button for button in tab.findChildren(QPushButton) if button.text() == "Save"
    ]
    assert len(saves) == 2
    assert all(button.isEnabled() for button in saves)


def test_exp_tab_run_tooltip_shows_no_soc_reason(main_window_control_factory, qapp):
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        mock_ctrl(main_window_control_factory),
        AdapterCapabilities(
            analysis=AnalysisMode.FIT, post_analysis=False, load_data=True
        ),
    )
    tab.update_interaction_state(
        make_tab_snapshot(
            "tab-1",
            global_run_active=False,
            is_running=False,
            is_analyzing=False,
            has_context=True,
            has_active_context=True,
            has_soc=False,
            has_run_result=False,
            has_analyze_result=False,
            has_figure=False,
        )
    )

    assert tab.run_btn.isEnabled() is False
    assert tab.run_btn.toolTip() == "No SoC connection"


def test_exp_tab_run_tooltip_shows_cfg_invalid_reason(
    main_window_control_factory, qapp
):
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        mock_ctrl(main_window_control_factory),
        AdapterCapabilities(
            analysis=AnalysisMode.FIT, post_analysis=False, load_data=True
        ),
    )
    reps = tab.findChild(QLineEdit, "form.input.reps")
    assert reps is not None
    commit_text(reps, "many")
    tab.update_interaction_state(
        make_tab_snapshot(
            "tab-1",
            global_run_active=False,
            is_running=False,
            is_analyzing=False,
            has_context=True,
            has_active_context=True,
            has_soc=True,
            has_run_result=False,
            has_analyze_result=False,
            has_figure=False,
        )
    )

    assert tab.run_btn.isEnabled() is False
    assert tab.run_btn.toolTip().startswith("Config invalid: reps: ")


def test_exp_tab_unselected_context_allows_analysis_but_disables_run_and_save(
    main_window_control_factory, qapp
):
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    snap = make_tab_snapshot(
        "tab-1",
        global_run_active=False,
        is_running=False,
        is_analyzing=False,
        has_context=True,
        has_active_context=False,
        has_soc=True,
        has_run_result=True,
        has_analyze_result=True,
        has_figure=True,
    )
    assert snap.capabilities is not None
    tab = ExpTabWidget(
        "tab-1", mock_ctrl(main_window_control_factory), snap.capabilities
    )
    tab.update_writeback_items(
        [ParameterWriteback("r_f", "freq", proposed_value=6100.0)]
    )
    tab.update_interaction_state(snap)

    assert tab.run_btn.isEnabled() is False
    assert tab.run_btn.toolTip() == "Select or create a working point first"
    load = next(
        button
        for button in tab.findChildren(QPushButton)
        if button.text() == "Load Data"
    )
    assert load.isHidden() is False
    assert load.isEnabled() is True
    assert tab.analyze_btn.isEnabled() is True
    assert tab.writeback_widget.isEnabled() is True
    saves = [
        button for button in tab.findChildren(QPushButton) if button.text() == "Save"
    ]
    assert len(saves) == 2
    assert all(not button.isEnabled() for button in saves)


def test_non_analysis_adapter_hides_analysis_widgets_but_keeps_save(
    main_window_control_factory, qapp
):
    """flux_dep / power_dep adapters (analysis=NONE) hide only the
    analysis widgets, never the Save section. Regression: the whole second tab
    used to be hidden, so the user could not save a 2D-sweep run at all."""
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        mock_ctrl(main_window_control_factory),
        AdapterCapabilities(
            analysis=AnalysisMode.NONE, post_analysis=False, load_data=True
        ),
    )
    tab.update_interaction_state(
        make_tab_snapshot(
            "tab-1",
            has_run_result=True,
            has_active_context=True,
            supports_analysis=False,
            analyze_params=None,
        )
    )

    # Fixed order Run | Analysis? | Post? | Data | Guide: Analysis not constructed, Data always visible
    left_tabs = tab_pane_tabs(tab)
    visible = [left_tabs.tabText(i) for i in range(left_tabs.count())]
    assert visible == ["Run", "Data", "Guide"]
    # Prove Analysis page and controls were never constructed, not only hidden
    assert not hasattr(tab, "_analysis_panel")
    assert not hasattr(tab, "analyze_form")
    assert not hasattr(tab, "_analyze_section")
    # ... but Save stays reachable and usable (run result + active context).
    saves = [
        button for button in tab.findChildren(QPushButton) if button.text() == "Save"
    ]
    assert len(saves) == 1
    assert all(button.isEnabled() for button in saves)


def test_analysis_adapter_shows_analysis_widgets_and_labels_tab(
    main_window_control_factory, qapp
):
    """An analysis adapter keeps the analysis widgets visible and the second tab
    labelled 'Analysis' — the counterpart to the non-analysis case."""
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        mock_ctrl(main_window_control_factory),
        AdapterCapabilities(
            analysis=AnalysisMode.FIT, post_analysis=False, load_data=True
        ),
    )
    tab.update_interaction_state(
        make_tab_snapshot("tab-1", has_run_result=True, supports_analysis=True)
    )

    assert tab_pane_tabs(tab).tabText(1) == "Analysis"
    analysis_pane = tab_pane_tabs(tab).widget(1)
    assert analysis_pane is not None
    assert tab.analyze_form.isVisibleTo(analysis_pane) is True
    assert tab.analyze_btn.isHidden() is False
    saves = [
        button for button in tab.findChildren(QPushButton) if button.text() == "Save"
    ]
    assert len(saves) == 2


def test_exp_tab_load_button_requires_context_but_not_soc(
    main_window_control_factory, qapp
):
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    snap1 = make_tab_snapshot(
        "tab-1",
        has_context=True,
        has_active_context=False,
        has_soc=False,
        has_run_result=False,
        has_analyze_result=False,
        has_figure=False,
    )
    assert snap1.capabilities is not None
    tab = ExpTabWidget(
        "tab-1", mock_ctrl(main_window_control_factory), snap1.capabilities
    )
    tab.update_interaction_state(snap1)
    load = next(
        button
        for button in tab.findChildren(QPushButton)
        if button.text() == "Load Data"
    )
    assert load.isHidden() is False
    assert load.isEnabled() is True

    tab.update_interaction_state(
        make_tab_snapshot(
            "tab-1",
            has_context=False,
            has_active_context=False,
            has_soc=False,
            has_run_result=False,
            has_analyze_result=False,
            has_figure=False,
        )
    )
    assert load.isEnabled() is False


def test_exp_tab_attach_projects_existing_resource(main_window_control_factory, qapp):
    ctrl = editor_wiring_ctrl(main_window_control_factory)
    cfg = ctrl.cfg_resources.lookup("tab-1")
    tab = ExpTabWidget(
        "tab-1", ctrl, AdapterCapabilities(analysis=AnalysisMode.FIT, load_data=True)
    )
    tab.attach(make_tab_snapshot("tab-1"), _RecordingTabActions())
    cfg.write({("gain",): 0.45})
    entry = tab.findChild(QLineEdit, "form.input.gain")
    assert entry is not None and float(entry.text()) == 0.45


def test_exp_tab_detach_preserves_resource_for_later_views(
    main_window_control_factory, qapp
):
    ctrl = editor_wiring_ctrl(main_window_control_factory)
    cfg = ctrl.cfg_resources.lookup("tab-1")
    tab = ExpTabWidget(
        "tab-1", ctrl, AdapterCapabilities(analysis=AnalysisMode.FIT, load_data=True)
    )
    tab.attach(make_tab_snapshot("tab-1"), _RecordingTabActions())
    tab.detach()
    cfg.write({("gain",): 0.45})
    other = ExpTabWidget(
        "tab-1", ctrl, AdapterCapabilities(analysis=AnalysisMode.FIT, load_data=True)
    )
    other.attach(make_tab_snapshot("tab-1"), _RecordingTabActions())
    entry = other.findChild(QLineEdit, "form.input.gain")
    assert entry is not None and float(entry.text()) == 0.45
    assert other.cfg_invalid_reason() is None


def test_exp_tab_buttons_dispatch_public_tab_actions(main_window_control_factory, qapp):
    import dataclasses

    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    ctrl = editor_wiring_ctrl(main_window_control_factory)
    tab = ExpTabWidget(
        "tab-1",
        ctrl,
        AdapterCapabilities(
            analysis=AnalysisMode.FIT, post_analysis=True, load_data=True
        ),
    )
    actions = _RecordingTabActions()
    snapshot = dataclasses.replace(
        make_tab_snapshot(
            "tab-1",
            has_run_result=True,
            has_analyze_result=True,
            has_figure=True,
            has_post_analyze_result=True,
            supports_post_analysis=True,
        ),
        cfg_input={},
    )
    tab.attach(snapshot, actions)

    assert tab.run_btn.isEnabled() is True
    load = next(
        button
        for button in tab.findChildren(QPushButton)
        if button.text() == "Load Data"
    )
    assert load.isHidden() is False
    assert load.isEnabled() is True
    assert tab.analyze_btn.isEnabled() is True
    assert tab.post_analyze_btn.isEnabled() is True
    saves = [
        button for button in tab.findChildren(QPushButton) if button.text() == "Save"
    ]
    assert len(saves) == 3
    assert all(button.isEnabled() for button in saves)

    actions.calls.clear()
    tab.run_btn.click()
    load.click()
    tab.analyze_btn.click()
    tab.post_analyze_btn.click()
    tab.writeback_widget.apply_requested.emit()
    tab.post_writeback_widget.apply_requested.emit()
    for button in saves:
        button.click()
    save_all = next(
        button
        for button in tab.findChildren(QPushButton)
        if button.text() == "Save All"
    )
    assert save_all.isEnabled()
    save_all.click()

    assert actions.calls == [
        ("run_or_stop", "tab-1"),
        ("load_data", "tab-1"),
        ("analyze", "tab-1"),
        ("post_analyze", "tab-1"),
        ("apply_writeback", "tab-1"),
        ("apply_post_writeback", "tab-1"),
        ("save_artifact", "tab-1", ArtifactRef("run", "measurement_data")),
        ("save_artifact", "tab-1", ArtifactRef("analysis", "results")),
        ("save_artifact", "tab-1", ArtifactRef("post_analysis", "results")),
        ("save_all", "tab-1"),
    ]


def test_exp_tab_reset_publishes_defaults_on_same_resource(
    main_window_control_factory, qapp
):
    dialogs = RecordingDialogPresenter(confirm_answers=[True])
    ctrl = editor_wiring_ctrl(main_window_control_factory)
    cfg = ctrl.cfg_resources.lookup("tab-1")
    defaults = cfg.input.data
    cfg.write({("gain",): 0.45})
    tab = ExpTabWidget(
        "tab-1",
        ctrl,
        AdapterCapabilities(analysis=AnalysisMode.FIT, load_data=True),
        dialog_presenter=dialogs,
    )
    actions = _RecordingTabActions()
    tab.attach(make_tab_snapshot("tab-1"), actions)
    tab.reset_btn.click()

    assert cfg.input.data == defaults
    assert actions.calls[-1] == ("refresh_interaction", "tab-1")


def test_exp_tab_reset_confirm_no_does_not_reset(main_window_control_factory, qapp):
    dialogs = RecordingDialogPresenter(confirm_answers=[False])
    ctrl = editor_wiring_ctrl(main_window_control_factory)
    cfg = ctrl.cfg_resources.lookup("tab-1")
    before = cfg.input.data
    tab = ExpTabWidget(
        "tab-1",
        ctrl,
        AdapterCapabilities(analysis=AnalysisMode.FIT, load_data=True),
        dialog_presenter=dialogs,
    )
    actions = _RecordingTabActions()
    tab.attach(make_tab_snapshot("tab-1"), actions)
    actions.calls.clear()
    tab.reset_btn.click()
    assert cfg.input.data == before
    assert actions.calls == []


def test_exp_tab_reset_btn_idle_only_enable(main_window_control_factory, qapp):
    """reset_btn must be enabled when idle and disabled while the tab is busy."""
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        mock_ctrl(main_window_control_factory),
        AdapterCapabilities(
            analysis=AnalysisMode.FIT, post_analysis=False, load_data=True
        ),
    )

    # Idle: reset_btn should be enabled.
    tab.update_interaction_state(make_tab_snapshot("tab-1", is_running=False))
    assert tab.reset_btn.isEnabled() is True

    # Busy (running): reset_btn must be disabled.
    tab.update_interaction_state(make_tab_snapshot("tab-1", is_running=True))
    assert tab.reset_btn.isEnabled() is False

    # Back to idle: reset_btn re-enabled.
    tab.update_interaction_state(make_tab_snapshot("tab-1", is_running=False))
    assert tab.reset_btn.isEnabled() is True


def _tab(window: MainWindow, tab_id: str) -> ExpTabWidget:
    return next(
        tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == tab_id
    )


@dataclass
class _OptionalFitParams:
    delay: Annotated[float | None, ParamMeta(label="Delay")] = None


def test_optional_invalid_input_disables_fit_and_shows_recoverable_error(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str], tmp_path: Path
) -> None:
    """UI evidence reads only named visible driver state, not domain state."""
    _session, ctrl, window, tab_id = live_ui
    tab = _tab(window, tab_id)
    tab.populate_analyze_params(_OptionalFitParams())
    tab.update_interaction_state(ctrl.get_tab_snapshot(tab_id))
    window.resize(1200, 850)
    window.show()
    driver = ComponentDriver(window)
    driver.select_tab("Analysis")
    driver.screenshot(tmp_path / "01-blank.png")
    assert driver.observe("Analyze").enabled

    driver.enter_text("Delay:", "-")
    driver.screenshot(tmp_path / "02-invalid.png")
    assert driver.observe("-").text == "-"
    assert not driver.observe("Analyze").enabled
    assert "Delay" in driver.observe("-").tooltip
    assert any(
        "Delay" in item.text and "number" in item.text
        for item in driver.visible().widgets
        if item.kind == "QLabel"
    )

    driver.enter_text("Delay:", "2.5")
    driver.screenshot(tmp_path / "03-valid.png")
    assert driver.observe("Analyze").enabled
    assert driver.observe("2.5").tooltip == ""
    assert not any("number" in item.text for item in driver.visible().widgets)

    driver.enter_text("Delay:", "")
    driver.screenshot(tmp_path / "04-blank-again.png")
    fields = [item for item in driver.visible().widgets if item.kind == "QLineEdit"]
    assert len(fields) == 1
    assert fields[0].text == ""
    assert driver.observe("Analyze").enabled
    assert not any("number" in item.text for item in driver.visible().widgets)


def test_result_focus_and_panel_width_are_owned_by_tab(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
) -> None:
    _session, ctrl, window, tab_id = live_ui
    tab = _tab(window, tab_id)
    panels = tab.findChild(QTabWidget)
    assert panels is not None
    tab.select_pane("analysis")
    analysis_panel = panels.currentWidget()
    tab.select_pane("run")
    assert panels.currentWidget() is not analysis_panel

    tab.focus_result_panel()

    assert panels.currentWidget() is analysis_panel
    assert tab.left_panel_width() == ctrl.get_left_panel_width()
    assert window.current_left_panel_width() == tab.left_panel_width()


def test_window_dimensions_roundtrip_through_layout_caretaker(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str], tmp_path: Path
) -> None:
    _session, ctrl, window, _original_tab = live_ui
    ctrl.add_view(window)
    ctrl.attach_caretaker(create_persistence_caretaker(ctrl, cache_dir=tmp_path))
    original_ids = set(ctrl.list_tab_ids())
    window.resize(1100, 820)
    ctrl.persist_all()
    window.resize(900, 650)
    try:
        ctrl.restore_all()
        assert (window.width(), window.height()) == (1100, 820)
    finally:
        for tab_id in tuple(ctrl.list_tab_ids()):
            if tab_id not in original_ids:
                ctrl.close_tab(tab_id)


def test_prepare_run_container_clears_run_and_downstream_figures(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
) -> None:
    _session, _ctrl, window, tab_id = live_ui
    tab = _tab(window, tab_id)
    analysis = FigureCollection()
    analysis.adopt("fit", Figure())
    analysis.seal()
    post = FigureCollection()
    post.adopt("diagnostic", Figure())
    post.seal()
    tab.show_run_figure(Figure())
    tab.show_analysis_figures(analysis)
    tab.show_post_analysis_figures(post)
    for pane in ("run", "analysis", "post_analysis"):
        assert tab.get_current_figure_for_pane(pane) is not None
    captured = tab.get_run_container()

    container = tab.prepare_run_container()

    assert container is captured
    assert container is tab.get_run_container()
    for pane in ("run", "analysis", "post_analysis"):
        assert tab.get_current_figure_for_pane(pane) is None


class _Interactive(InteractiveFrontend):
    def __init__(self) -> None:
        super().__init__()
        self.stopped = False
        self.read_only = False
        self._preview_active = True
        self._figure = Figure()

    @property
    def figure(self) -> Figure:
        return self._figure

    @property
    def preview_active(self) -> bool:
        return self._preview_active

    def set_read_only(self, *, read_only: bool) -> None:
        self.read_only = read_only

    def cancel_preview(self) -> None:
        self._preview_active = False

    def teardown(self) -> None:
        self.stopped = True


class _SelectedInteractive(_Interactive):
    """A frontend type selected for removal while other frontends remain."""


class _RecordingInteractive(_Interactive):
    """Record authority projections delivered to a mounted plugin view."""

    def __init__(self) -> None:
        super().__init__()
        self.read_only_changes: list[bool] = []

    def set_read_only(self, *, read_only: bool) -> None:
        self.read_only_changes.append(read_only)


def test_unmounted_interactive_frontend_stops_authority_projection(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
) -> None:
    _session, ctrl, window, tab_id = live_ui
    widget = _RecordingInteractive()
    window.mount_interactive_analysis(tab_id, lambda _env: widget)
    assert widget.read_only_changes == [False]

    window.unmount_interactive_analysis(tab_id)
    ctrl.control_identity.connect("agent", "retired-view-check", "owner")

    assert widget.stopped is True
    assert widget.read_only_changes == [False]


@pytest.mark.parametrize(
    "retirement", ["close_tab", "window_close", "runtime_shutdown"]
)
def test_retired_view_stops_authority_projection(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
    retirement: str,
) -> None:
    _session, ctrl, _fixture_window, _fixture_tab = live_ui
    ctrl.attach_caretaker(create_persistence_caretaker(ctrl))
    tab_id = ctrl.new_tab("saving")
    dialogs = RecordingDialogPresenter(destructive_answers=[True, True])
    window = MainWindow(ctrl, templates=lambda: {}, dialog_presenter=dialogs)
    window.add_tab_widget(tab_id, "saving")
    window.show()
    widget = _RecordingInteractive()
    window.mount_interactive_analysis(tab_id, lambda _env: widget)
    assert widget.read_only_changes == [False]
    try:
        if retirement == "close_tab":
            ctrl.tab_control.close_tab(tab_id)
        elif retirement == "window_close":
            window.close()
            QCoreApplication.processEvents()
            assert not window.isVisible()
        else:
            window.request_shutdown()
            QCoreApplication.processEvents()
            assert not window.isVisible()

        ctrl.control_identity.connect("agent", "retired-view-check", "owner")

        assert widget.read_only_changes == [False]
    finally:
        ctrl.abort_shutdown()
        if ctrl.has_tab(tab_id):
            ctrl.tab_control.close_tab(tab_id)
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.slow
# tolerated(capability): Qt process exit must not quit the suite's QApplication.
@pytest.mark.requires_subprocess
def test_application_quit_stops_authority_projection(tmp_path: Path) -> None:
    """Exercise real Qt process exit without mutating the suite's QApplication."""
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.gui.app.measure.ui._quit_authority",
            str(tmp_path),
        ],
        env={
            **os.environ,
            "QT_QPA_PLATFORM": "offscreen",
            "QT_QPA_PLATFORMTHEME": "generic",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as child:
        try:
            stdout, stderr = child.communicate(timeout=30)
            assert child.returncode == 0, stderr
            assert "application quit retired authority projection" in stdout
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)


def test_interactive_widget_lifecycle_is_owned_by_tab(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
) -> None:
    _session, _ctrl, window, tab_id = live_ui
    tab = _tab(window, tab_id)
    first = _SelectedInteractive()
    second = _SelectedInteractive()
    unrelated = _Interactive()
    for widget in (first, second, unrelated):
        tab.mount_interactive_widget(widget)
    assert tab.interactive_frontend() is unrelated

    tab.unmount_interactive_widgets(_SelectedInteractive)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    assert tab.get_current_figure_for_pane("analysis") is None
    remaining = tab.findChildren(_Interactive)
    assert first not in remaining and second not in remaining
    assert unrelated in remaining
    tab.mount_interactive_widget(unrelated)
    assert tab.interactive_frontend() is unrelated
    tab.unmount_interactive_widgets(_Interactive)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not tab.findChildren(_Interactive)
    assert tab.get_current_figure_for_pane("analysis") is None


@pytest.mark.parametrize("failure_stage", ["factory", "mount"])
def test_interactive_setup_failure_clears_stale_figure_and_cleans_widget(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    _session, ctrl, window, tab_id = live_ui
    tab = _tab(window, tab_id)
    figures = FigureCollection()
    figures.adopt("stale", Figure())
    figures.seal()
    tab.show_analysis_figures(figures)
    assert tab.get_current_figure_for_pane("analysis") is figures["stale"]
    captured = tab.get_analysis_container()
    widget = _Interactive()
    if failure_stage == "mount":

        def fail_mount(_widget: QWidget) -> None:
            raise RuntimeError("mount failed")

        monkeypatch.setattr(tab, "mount_interactive_widget", fail_mount)

    def frontend_factory(env: Controller) -> InteractiveFrontend:
        assert env is ctrl
        if failure_stage == "factory":
            raise RuntimeError("factory failed")
        return widget

    with pytest.raises(RuntimeError, match=failure_stage):
        window.mount_interactive_analysis(tab_id, frontend_factory)

    assert tab.get_current_figure_for_pane("analysis") is None
    assert tab.get_analysis_container() is captured
    if failure_stage == "mount":
        assert widget.stopped is True


def test_interactive_mount_and_unmount_quiesce_frontend(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
) -> None:
    _session, _ctrl, window, tab_id = live_ui
    tab = _tab(window, tab_id)
    captured = tab.get_analysis_container()
    widget = _Interactive()

    window.mount_interactive_analysis(tab_id, lambda _env: widget)
    assert tab.get_analysis_container() is captured
    assert widget in tab.findChildren(_Interactive)
    assert tab.interactive_frontend() is widget
    window.unmount_interactive_analysis(tab_id)
    assert widget.stopped is True
    assert tab.get_current_figure_for_pane("analysis") is None
    assert tab.get_analysis_container() is captured


def test_mounted_interactive_view_exposes_figure_and_discardable_preview(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
) -> None:
    _session, _ctrl, window, tab_id = live_ui
    widget = _Interactive()
    window.mount_interactive_analysis(tab_id, lambda _env: widget)

    assert window.interactive_presentation(tab_id) == (widget.figure, True)
    window.discard_interactive_preview(tab_id)
    assert window.interactive_presentation(tab_id) == (widget.figure, False)
    window.unmount_interactive_analysis(tab_id)
    assert window.interactive_presentation(tab_id) is None


def test_interactive_success_restores_committed_figure_in_analysis_pane(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
) -> None:
    session, ctrl, window, tab_id = live_ui
    tab = _tab(window, tab_id)
    widget = _Interactive()
    plots = Plots(NonPresentingHost())
    committed_figure, _axes = plots.subplots("fit")
    plots.finish(present=False)
    session.state.update_tab_analyze(
        tab_id, PaneSummary(), plots, analyze_params_instance=PaneParams()
    )
    window.refresh_tab_figure(tab_id, ctrl.get_tab_snapshot(tab_id))
    assert tab.get_current_figure_for_pane("analysis") is committed_figure
    captured = tab.get_analysis_container()

    window.mount_interactive_analysis(tab_id, lambda _env: widget)
    assert tab.get_current_figure_for_pane("analysis") is widget.figure
    window.unmount_interactive_analysis(tab_id, restore_result=True)

    assert widget.stopped is True
    assert tab.get_analysis_container() is captured
    assert tab.get_current_figure_for_pane("analysis") is committed_figure
    assert committed_figure is not widget.figure
