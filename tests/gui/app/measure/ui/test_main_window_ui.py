"""UI structure tests for ExpTabWidget layout decisions."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QCoreApplication, QEvent, Qt
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QToolButton,
)
from zcu_tools.experiment.artifacts import ArtifactRef
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    ParameterWriteback,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.artifacts import (
    ArtifactSectionSnapshot,
    ArtifactSnapshot,
    SaveStatus,
)
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services import TabSnapshot
from zcu_tools.gui.app.measure.services.tab_form import (
    AcceptedFormConfig,
    TabFormResource,
)
from zcu_tools.gui.app.measure.state import State, TabInteractionState
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.app.measure.writeback import WritebackItem
from zcu_tools.gui.control_identity import ControlIdentity
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.plotting import FigureContainer
from zcu_tools.gui.plotting.explicit import QtPlotHost
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.gui.session.events import SocChangedPayload
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.plotting.plots import Plots
from zcu_tools.resources.entry import (
    RoleResolutionError,
)

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._form_fakes import configure_form_lookup, make_form
from tests.gui.app.measure._synthetic import (
    SampleAdapter,
    SampleAnalysis,
    SampleRun,
)
from tests.gui.app.measure.artifacts._snapshots import ready_figures, with_artifacts
from tests.gui.widgets.form._text_input import commit_text


@pytest.fixture
def control_factory(qapp: QApplication) -> Iterator[Callable[[], ControlIdentity]]:
    """Keep fake application authority alive until its Qt view tree is deleted."""
    baseline = set(qapp.topLevelWidgets())
    authorities: list[ControlIdentity] = []

    def create() -> ControlIdentity:
        authority = ControlIdentity()
        authorities.append(authority)
        return authority

    yield create
    for widget in qapp.topLevelWidgets():
        if widget not in baseline:
            widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()
    authorities.clear()


def _mock_ctrl(control_factory: Callable[[], ControlIdentity]) -> MagicMock:
    ctrl = MagicMock()
    ctrl.control_identity = control_factory()
    ctrl.get_adapter_guide.return_value = AdapterGuide(
        markdown="# Synthetic guide\n", resource_dir=None
    )
    configure_form_lookup(ctrl)
    ctrl.get_left_panel_width.return_value = 500
    ctrl.writeback_control.get_writeback_values_for_pane.return_value.__getitem__.return_value.actual_path = "r_f"
    return ctrl


def _experiment_tabs(window: MainWindow) -> QTabWidget:
    tabs = window.findChild(QTabWidget)
    assert tabs is not None
    return tabs


def test_main_window_role_failure_is_presented_without_registering_tab(
    control_factory, qapp
):
    ctrl = _apply_window_defaults(_mock_ctrl(control_factory))
    ctrl.get_bus.return_value = EventBus()
    ctrl.new_tab.side_effect = RoleResolutionError(
        "qubit", "R1", "qubit/*", "wrong kind"
    )
    dialogs = RecordingDialogPresenter()
    window = MainWindow(ctrl, templates=lambda: {}, dialog_presenter=dialogs)
    window.create_tab("fake")
    assert len(dialogs.calls) == 1
    assert "qubit" in dialogs.calls[0].message
    tabs = window.findChild(QTabWidget)
    assert tabs is not None and tabs.count() == 0


def _apply_window_defaults(ctrl: MagicMock) -> MagicMock:
    """Set the minimal return values required by MainWindow.__init__ on a mock ctrl.

    MainWindow uses active_operation_count() for shutdown confirmation.
    Stub the count to keep tests independent of live operations.
    """
    ctrl.active_operation_count.return_value = 0
    ctrl.get_active_context_label.return_value = None
    ctrl.has_project.return_value = False
    ctrl.context_control.get_alias_rows.return_value = ()
    ctrl.get_session_env.return_value = SessionEnv(
        parameters=None, soc=None, soccfg=None
    )
    return ctrl


# Distinguishes "caller did not specify analyze_params" (default to a MagicMock)
# from "caller explicitly wants None" (a non-analysis adapter's snapshot).
_DEFAULT_PARAMS = object()


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


def _snapshot(
    tab_id: str,
    *,
    global_run_active: bool = False,
    is_running: bool = False,
    is_analyzing: bool = False,
    has_context: bool = True,
    has_active_context: bool = True,
    has_soc: bool = True,
    has_run_result: bool = True,
    has_analyze_result: bool = True,
    has_figure: bool = True,
    has_post_analyze_result: bool = False,
    supports_analysis: bool = True,
    supports_post_analysis: bool = False,
    supports_load_data: bool = False,
    analyze_params: object = _DEFAULT_PARAMS,
    post_analyze_params: object | None = None,
    writeback_items: tuple[WritebackItem, ...] = (),
    figure: object = _DEFAULT_PARAMS,
    post_figure: object = _DEFAULT_PARAMS,
    data_status: SaveStatus | None = None,
) -> TabSnapshot:
    from zcu_tools.gui.app.measure.services.ports import (
        AnalysisPaneSnapshot,
        PostAnalysisPaneSnapshot,
        RunPaneSnapshot,
        SavePaneSnapshot,
    )

    # Resolve analysis params
    resolved_analyze_params = (
        MagicMock() if analyze_params is _DEFAULT_PARAMS else analyze_params
    )
    # Resolve figures: allow explicit None or Figure override
    if figure is _DEFAULT_PARAMS:
        from matplotlib.figure import Figure

        figure_obj: Figure | None = (
            Figure() if has_figure and has_analyze_result else None
        )
    else:
        from matplotlib.figure import Figure

        assert figure is None or isinstance(figure, Figure)
        figure_obj = figure
    if post_figure is _DEFAULT_PARAMS:
        from matplotlib.figure import Figure as _Fig2

        post_figure_obj: Figure | None = _Fig2() if has_post_analyze_result else None
    else:
        assert post_figure is None or isinstance(post_figure, Figure)
        post_figure_obj = post_figure

    # Capabilities include load_data
    caps = AdapterCapabilities(
        analysis=AnalysisMode.FIT if supports_analysis else AnalysisMode.NONE,
        post_analysis=supports_post_analysis,
        load_data=supports_load_data,
    )

    # Pane snapshots
    run_snap = RunPaneSnapshot(
        result=object() if has_run_result else None,
        source_path=None,
    )
    analysis_snap = AnalysisPaneSnapshot(
        params=resolved_analyze_params,
        result=object() if has_analyze_result else None,
        figures=ready_figures(figure_obj),
        writeback_items=tuple(writeback_items),
    )
    post_snap = PostAnalysisPaneSnapshot(
        params=post_analyze_params,
        result=object() if has_post_analyze_result else None,
        figures=ready_figures(post_figure_obj),
        writeback_items=(),
    )
    snapshot = TabSnapshot(
        adapter_name="fake",
        tab_id=tab_id,
        interaction=TabInteractionState(
            global_run_active=global_run_active,
            is_running=is_running,
            is_analyzing=is_analyzing,
            has_context=has_context,
            has_active_context=has_active_context,
            has_soc=has_soc,
            has_run_result=has_run_result,
            has_analyze_result=has_analyze_result,
            has_figure=has_figure,
            has_post_analyze_result=has_post_analyze_result,
        ),
        cfg_input={},
        capabilities=caps,
        run=run_snap,
        analysis=analysis_snap,
        post_analysis=post_snap,
        save=SavePaneSnapshot(),
    )
    snapshot = with_artifacts(snapshot)
    if data_status is None or data_status is SaveStatus.NOT_SAVED:
        return snapshot
    # Close-guard specimens retain planned paths and project the Run save facts.
    sections: list[ArtifactSectionSnapshot] = []
    for section in snapshot.artifacts:
        if section.section != "run":
            sections.append(section)
            continue
        artifacts: list[ArtifactSnapshot] = []
        for artifact in section.artifacts:
            members = tuple(
                replace(
                    member,
                    outputs=tuple(
                        replace(output, status="saved") for output in member.outputs
                    ),
                )
                for member in artifact.members
            )
            artifacts.append(replace(artifact, status=data_status, members=members))
        sections.append(replace(section, artifacts=tuple(artifacts)))
    return replace(snapshot, artifacts=tuple(sections))


def _left_tabs(tab: ExpTabWidget) -> QTabWidget:
    """Return the left tab widget, failing if it is missing."""
    tabs = tab.findChild(QTabWidget)
    assert tabs is not None
    return tabs


def _left_panel_handle(tab: ExpTabWidget) -> QToolButton:
    """Return the tooltip-identified panel handle, failing unless it is unique."""
    handles = [
        button
        for button in tab.findChildren(QToolButton)
        if button.toolTip() in ("Collapse left panel", "Expand left panel")
    ]
    assert len(handles) == 1
    return handles[0]


def _new_tab_button(window: MainWindow) -> QPushButton:
    """Return the New Tab button by its label, failing unless it is unique."""
    buttons = [
        button
        for button in window.findChildren(QPushButton)
        if button.text() == "New Tab ▾"
    ]
    assert len(buttons) == 1
    return buttons[0]


def test_left_panel_toggle_is_attached_to_tab_bar(control_factory, qapp):
    from qtpy.QtWidgets import QApplication
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        _mock_ctrl(control_factory),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )
    tab.show()
    QApplication.processEvents()

    left_tabs = _left_tabs(tab)
    corner = left_tabs.cornerWidget(Qt.Corner.TopLeftCorner)
    assert corner is None
    assert _left_panel_handle(tab).isVisible() is True


def test_left_panel_toggle_uses_collapsed_boundary_handle(control_factory, qapp):
    from qtpy.QtWidgets import QApplication
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        _mock_ctrl(control_factory),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )
    tab.resize(1000, 700)
    tab.show()
    QApplication.processEvents()

    splitter = tab.findChild(QSplitter)
    assert splitter is not None
    handle = _left_panel_handle(tab)
    expanded_left = splitter.sizes()[0]
    expanded_handle_x = handle.x()
    assert expanded_handle_x > 0
    assert handle.toolTip() == "Collapse left panel"

    handle.click()
    QApplication.processEvents()

    assert splitter.sizes()[0] == 0
    assert handle.toolTip() == "Expand left panel"
    assert handle.x() == 0

    handle.click()
    QApplication.processEvents()

    assert handle.toolTip() == "Collapse left panel"
    assert handle.x() > 0
    assert splitter.sizes()[0] >= expanded_left


def test_left_panel_handle_tracks_splitter_boundary(control_factory, qapp):
    from qtpy.QtWidgets import QApplication
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        _mock_ctrl(control_factory),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )
    tab.resize(1000, 700)
    tab.show()
    QApplication.processEvents()

    splitter = tab.findChild(QSplitter)
    assert splitter is not None
    left_tabs = _left_tabs(tab)
    handle = _left_panel_handle(tab)
    initial_x = handle.x()
    splitter_x = splitter.geometry().x()
    left_boundary = splitter_x + left_tabs.geometry().right() + 1
    assert abs(initial_x - (left_boundary - handle.width() // 2)) <= 2

    splitter.moveSplitter(260, 1)
    QApplication.processEvents()

    moved_x = handle.x()
    splitter_x = splitter.geometry().x()
    left_boundary = splitter_x + left_tabs.geometry().right() + 1
    assert abs(moved_x - (left_boundary - handle.width() // 2)) <= 2


def test_exp_tab_disables_local_buttons_while_analyzing(control_factory, qapp):
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        _mock_ctrl(control_factory),
        AdapterCapabilities(
            analysis=AnalysisMode.FIT, post_analysis=False, load_data=True
        ),
    )
    tab.update_writeback_items(
        [ParameterWriteback("r_f", "freq", proposed_value=6100.0)]
    )
    tab.update_interaction_state(
        _snapshot(
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
            supports_load_data=True,
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


def test_exp_tab_keeps_analyze_enabled_while_other_tab_running(control_factory, qapp):
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        _mock_ctrl(control_factory),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )
    tab.update_interaction_state(
        _snapshot(
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


def test_exp_tab_run_tooltip_shows_no_soc_reason(control_factory, qapp):
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        _mock_ctrl(control_factory),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )
    tab.update_interaction_state(
        _snapshot(
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


def test_exp_tab_run_tooltip_shows_cfg_invalid_reason(control_factory, qapp):
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        _mock_ctrl(control_factory),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )
    reps = tab.findChild(QLineEdit, "form.input.reps")
    assert reps is not None
    commit_text(reps, "many")
    tab.update_interaction_state(
        _snapshot(
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
    control_factory, qapp
):
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    snap = _snapshot(
        "tab-1",
        supports_load_data=True,
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
    tab = ExpTabWidget("tab-1", _mock_ctrl(control_factory), snap.capabilities)
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
    control_factory, qapp
):
    """flux_dep / power_dep adapters (analysis=NONE) hide only the
    analysis widgets, never the Save section. Regression: the whole second tab
    used to be hidden, so the user could not save a 2D-sweep run at all."""
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        _mock_ctrl(control_factory),
        AdapterCapabilities(analysis=AnalysisMode.NONE, post_analysis=False),
    )
    tab.update_interaction_state(
        _snapshot(
            "tab-1",
            has_run_result=True,
            has_active_context=True,
            supports_analysis=False,
            analyze_params=None,
        )
    )

    # Fixed order Run | Analysis? | Post? | Data | Guide: Analysis not constructed, Data always visible
    left_tabs = _left_tabs(tab)
    visible = [left_tabs.tabText(i) for i in range(left_tabs.count())]
    assert visible == ["Run", "Data", "Guide"]
    # Prove Analysis page and controls were never constructed, not only hidden
    assert not hasattr(tab, "_analysis_panel")
    assert not hasattr(tab, "analyze_form")
    assert not hasattr(tab, "_analyze_section")
    # No Load Data capability means no control is constructed.
    assert not any(
        button.text() == "Load Data" for button in tab.findChildren(QPushButton)
    )
    # ... but Save stays reachable and usable (run result + active context).
    saves = [
        button for button in tab.findChildren(QPushButton) if button.text() == "Save"
    ]
    assert len(saves) == 1
    assert all(button.isEnabled() for button in saves)


def test_analysis_adapter_shows_analysis_widgets_and_labels_tab(control_factory, qapp):
    """An analysis adapter keeps the analysis widgets visible and the second tab
    labelled 'Analysis' — the counterpart to the non-analysis case."""
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        _mock_ctrl(control_factory),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )
    tab.update_interaction_state(
        _snapshot("tab-1", has_run_result=True, supports_analysis=True)
    )

    assert _left_tabs(tab).tabText(1) == "Analysis"
    analysis_pane = _left_tabs(tab).widget(1)
    assert analysis_pane is not None
    assert tab.analyze_form.isVisibleTo(analysis_pane) is True
    assert tab.analyze_btn.isHidden() is False
    saves = [
        button for button in tab.findChildren(QPushButton) if button.text() == "Save"
    ]
    assert len(saves) == 2


def test_exp_tab_load_button_requires_context_but_not_soc(control_factory, qapp):
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    snap1 = _snapshot(
        "tab-1",
        has_context=True,
        has_active_context=False,
        has_soc=False,
        has_run_result=False,
        has_analyze_result=False,
        has_figure=False,
        supports_load_data=True,
    )
    assert snap1.capabilities is not None
    tab = ExpTabWidget("tab-1", _mock_ctrl(control_factory), snap1.capabilities)
    tab.update_interaction_state(snap1)
    load = next(
        button
        for button in tab.findChildren(QPushButton)
        if button.text() == "Load Data"
    )
    assert load.isHidden() is False
    assert load.isEnabled() is True

    tab.update_interaction_state(
        _snapshot(
            "tab-1",
            has_context=False,
            has_active_context=False,
            has_soc=False,
            has_run_result=False,
            has_analyze_result=False,
            has_figure=False,
            supports_load_data=True,
        )
    )
    assert load.isEnabled() is False


@pytest.mark.parametrize("analysis_error", [None, "analysis defaults unavailable"])
def test_main_window_load_data_dialog_calls_controller(
    control_factory, qapp, monkeypatch, tmp_path, analysis_error
):
    from qtpy.QtWidgets import QFileDialog
    from zcu_tools.gui.app.measure.services.load import LoadTabResultOutcome
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow
    from zcu_tools.gui.app.measure.ui.run_load_dialog import RunLoadDialog

    ctrl = _apply_window_defaults(_mock_ctrl(control_factory))
    ctrl.get_bus.return_value = EventBus()
    ctrl.has_tab.return_value = True
    ctrl.get_tab_snapshot.return_value = _snapshot(
        "tab-1", supports_load_data=True, has_figure=False
    )
    database_root = tmp_path / "Database" / "Q3_2D" / "Q1"
    database_root.mkdir(parents=True)
    ctrl.save_control.list_saved_runs.return_value = ()
    ctrl.get_session_env.return_value = SessionEnv(
        parameters=None,
        soc=None,
        soccfg=None,
        database_path=str(database_root / "2026" / "06" / "Data_0625"),
    )
    ctrl.load_tab_result.return_value = LoadTabResultOutcome(
        tab_id="tab-1",
        data_path="/tmp/result.hdf5",
        result_type="Result",
        cfg_issue="cfg unavailable",
        has_analyze_params=analysis_error is None,
        analysis_error=analysis_error,
    )
    dialogs = RecordingDialogPresenter()
    window = MainWindow(ctrl, templates=lambda: {}, dialog_presenter=dialogs)
    window.add_tab_widget("tab-1", "fake")
    try:
        assert window.has_tab_widget("tab-1") is True
        window.show_status_message = MagicMock()

        captured_dir: dict[str, str] = {}

        def fake_get_open_file_name(*args, **kwargs):
            captured_dir["directory"] = args[2]
            return ("/tmp/result.hdf5", "")

        monkeypatch.setattr(
            QFileDialog,
            "getOpenFileName",
            fake_get_open_file_name,
        )

        def choose_file(dialog: RunLoadDialog) -> int:
            button = dialog.findChild(QPushButton, "load_file")
            assert button is not None
            button.click()
            return dialog.result()

        monkeypatch.setattr(RunLoadDialog, "exec", choose_file)
        window.load_tab_data_dialog("tab-1")

        ctrl.load_tab_result.assert_called_once_with("tab-1", "/tmp/result.hdf5")
        window.show_status_message.assert_called_once()
        assert captured_dir["directory"] == str(database_root)
        if analysis_error is not None:
            assert [(call.kind, call.title) for call in dialogs.calls] == [
                ("warning", "Analysis preparation failed"),
            ]
            assert "Data loaded successfully" in dialogs.consume_message_containing(
                "warning", analysis_error
            )
        dialogs.assert_no_unexpected_messages()
    finally:
        window.remove_tab_widget("tab-1")


def test_main_window_successful_writeback_relies_on_event_owned_projection(
    control_factory, qapp
):
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(_mock_ctrl(control_factory))
    ctrl.get_bus.return_value = EventBus()
    ctrl.get_tab_snapshot.return_value = _snapshot("tab-1", has_figure=False)
    ctrl.has_tab.return_value = True
    ctrl.apply_writeback_for_pane.return_value = {
        "applied_ids": ["param-1"],
        "written": {"parameter": ["r_f"], "ml_modules": [], "ml_waveforms": []},
    }
    window = MainWindow(ctrl, templates=lambda: {})
    window.add_tab_widget("tab-1", "fake")
    try:
        assert window.has_tab_widget("tab-1") is True
        window.refresh_tab_writeback = MagicMock()
        window.show_status_message = MagicMock()

        window.apply_tab_writeback("tab-1", pane="analysis")

        ctrl.apply_writeback_for_pane.assert_called_once_with("tab-1", "analysis")
        window.refresh_tab_writeback.assert_not_called()
        window.show_status_message.assert_called_once_with("Writeback applied: param-1")
    finally:
        window.remove_tab_widget("tab-1")


def test_apply_failure_preserves_edited_draft_and_selection_without_closing_window(
    tmp_path: Path, qapp: QApplication
) -> None:
    class BlockedWritebackAdapter(SampleAdapter):
        def get_writeback_items(
            self, req: WritebackRequest[SampleRun, SampleAnalysis]
        ) -> list[ParameterWriteback]:
            return [
                ParameterWriteback(
                    "general.ext.blocked.sigma", "Nested calibration", 1.0
                )
            ]

    registry = Registry()
    registry.register("blocked-writeback", BlockedWritebackAdapter)
    ctrl = Controller(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        registry,
        None,
        EventBus(),
        project_root=str(tmp_path),
    )
    ctrl.setup_control.new_entry(EntryRequest("synthetic"))
    target = ctrl.setup_control.new_point(
        PointCreateRequest("working", "", CommandOrigin("gui"))
    )
    ctrl.context_control.set_parameter(target, "general.ext.blocked", 3)
    tab_id = ctrl.new_tab("blocked-writeback")
    window = MainWindow(ctrl, templates=lambda: {})
    ctrl.add_view(window)
    window.add_tab_widget(tab_id, "blocked-writeback")
    window.show()
    driver = ComponentDriver(window)
    images = tmp_path / "apply-failure"
    try:
        driver.screenshot(images / "01-ready.png")
        driver.click("Run")
        driver.select_tab("Analysis")
        qt_until(
            lambda: driver.observe("Analyze").enabled,
            label="run result ready for analysis",
        )
        driver.click("Analyze")
        qt_until(
            lambda: driver.observe(object_name="writebackApply").enabled,
            label="analysis writeback ready",
        )
        driver.screenshot(images / "02-proposal.png")
        driver.click("Edit")
        driver.enter_text(object_name="writebackValue", value="7.25", commit=False)
        driver.screenshot(images / "03-edited.png")
        driver.click("Save")
        proposed = driver.observe(object_name="writebackProposed")
        selected = driver.observe("general.ext.blocked.sigma*")
        assert proposed.text == "7.25"
        assert selected.checked is True
        driver.screenshot(images / "04-draft.png")

        # Exercise the public Qt boundary directly first: a missing exception
        # barrier fails here without allowing the later real signal to abort pytest.
        window.apply_tab_writeback(tab_id)
        qt_until(lambda: bool(driver.dialogs()), label="direct Apply error visible")
        error = driver.dialogs()[0]
        assert error.title == "Writeback failed"
        assert any("not a field container" in text for text in error.texts)
        driver.screenshot(images / "05-boundary-failure.png")
        driver.click("OK")

        # This second attempt uses the shipped Apply signal and real mouse input.
        driver.click("Apply Selected")
        qt_until(lambda: bool(driver.dialogs()), label="clicked Apply error visible")
        error = driver.dialogs()[0]
        assert error.title == "Writeback failed"
        assert any("not a field container" in text for text in error.texts)
        driver.screenshot(images / "06-click-failure.png")
        driver.click("OK")
        assert driver.observe(object_name="writebackProposed") == proposed
        assert driver.observe("general.ext.blocked.sigma*") == selected
        assert driver.observe("Apply Selected").enabled
        assert "Writeback applied" not in driver.status().text
        assert driver.tabs("Analysis").current == "Analysis"
        driver.screenshot(images / "07-retained.png")
    finally:
        ctrl.close()
        ctrl.close_tab(tab_id)
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()


def _status_bar_labels(window: MainWindow, text: str) -> list[QLabel]:
    status_bar = window.statusBar()
    assert status_bar is not None
    return [label for label in status_bar.findChildren(QLabel) if label.text() == text]


def test_status_bar_message_is_mouse_selectable(control_factory, qapp):
    ctrl = _apply_window_defaults(_mock_ctrl(control_factory))
    ctrl.get_bus.return_value = EventBus()
    ctrl.list_tab_ids.return_value = []
    window = MainWindow(ctrl, templates=lambda: {})

    assert len(_status_bar_labels(window, "Ready")) == 1

    window.show_status_message("Writeback applied: param-1")

    labels = _status_bar_labels(window, "Writeback applied: param-1")
    assert len(labels) == 1
    flags = labels[0].textInteractionFlags()
    assert flags & Qt.TextInteractionFlag.TextSelectableByMouse
    assert window.get_view_snapshot()["status"] == "Writeback applied: param-1"


def test_main_window_toolbar_does_not_show_arb_waveforms(control_factory, qapp):
    from qtpy.QtWidgets import QPushButton
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(_mock_ctrl(control_factory))
    ctrl.get_bus.return_value = EventBus()

    window = MainWindow(ctrl, templates=lambda: {})
    texts = {button.text() for button in window.findChildren(QPushButton)}

    assert "Inspect…" in texts
    assert "Agent…" not in texts
    assert "Arb Waveforms…" not in texts


def test_main_window_named_dialog_facade_delegates_to_registry(
    control_factory, qapp, monkeypatch
):
    from zcu_tools.gui.app.measure.remote.dialogs import DialogName
    from zcu_tools.gui.app.measure.ui.main_dialog_registry import MainDialogRegistry
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(_mock_ctrl(control_factory))
    ctrl.get_bus.return_value = EventBus()
    registry = MagicMock()
    registry.visible_names.return_value = [DialogName.PREDICTOR]
    registry.take_screenshot.return_value = b"png"
    monkeypatch.setattr(MainDialogRegistry, "close", registry.close)
    monkeypatch.setattr(MainDialogRegistry, "open", registry.open)
    monkeypatch.setattr(MainDialogRegistry, "visible_names", registry.visible_names)
    monkeypatch.setattr(MainDialogRegistry, "take_screenshot", registry.take_screenshot)
    window = MainWindow(ctrl, templates=lambda: {})

    window.close_dialog(DialogName.PREDICTOR)
    window.open_dialog(DialogName.PREDICTOR)
    assert window.list_open_dialogs() == [DialogName.PREDICTOR]

    assert window.take_dialog_screenshot(DialogName.SETUP) == b"png"

    registry.close.assert_called_once_with(DialogName.PREDICTOR)
    registry.open.assert_called_once_with(DialogName.PREDICTOR, writable=True)
    registry.visible_names.assert_called_once_with()
    registry.take_screenshot.assert_called_once_with(DialogName.SETUP)


def test_show_error_dialog_retains_until_close(control_factory, qapp):
    import gc
    import weakref

    from qtpy.QtWidgets import QMessageBox

    ctrl = _apply_window_defaults(_mock_ctrl(control_factory))
    ctrl.get_bus.return_value = EventBus()
    window = MainWindow(ctrl, templates=lambda: {})

    window.show_error_dialog("Broken", "Something failed")
    qapp.processEvents()

    dialogs = [
        dialog
        for dialog in window.findChildren(QMessageBox)
        if dialog.windowTitle() == "Broken"
    ]
    assert len(dialogs) == 1
    dialog_ref = weakref.ref(dialogs[0])
    del dialogs
    gc.collect()

    dialog = dialog_ref()
    assert dialog is not None
    assert dialog.text() == "Something failed"
    assert dialog.icon() == QMessageBox.Icon.Critical
    assert dialog.isVisible()
    assert not dialog.isModal()
    assert dialog.windowModality() == Qt.WindowModality.NonModal
    assert window.list_open_dialogs() == []

    destroyed: list[bool] = []
    dialog.destroyed.connect(lambda: destroyed.append(True))
    dialog.close()
    del dialog
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()

    assert destroyed
    assert window.list_open_dialogs() == []


def test_main_window_load_data_dialog_cancel_is_noop(
    control_factory, qapp, monkeypatch
):
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow
    from zcu_tools.gui.app.measure.ui.run_load_dialog import RunLoadDialog

    ctrl = _apply_window_defaults(_mock_ctrl(control_factory))
    ctrl.get_bus.return_value = EventBus()
    ctrl.has_tab.return_value = True
    ctrl.get_tab_snapshot.return_value = _snapshot(
        "tab-1", supports_load_data=True, has_figure=False
    )
    ctrl.save_control.list_saved_runs.return_value = ()
    window = MainWindow(ctrl, templates=lambda: {})
    window.add_tab_widget("tab-1", "fake")
    try:
        assert window.has_tab_widget("tab-1") is True

        def cancel(dialog: RunLoadDialog) -> int:
            button = dialog.findChild(QPushButton, "load_cancel")
            assert button is not None
            button.click()
            return dialog.result()

        monkeypatch.setattr(RunLoadDialog, "exec", cancel)
        window.load_tab_data_dialog("tab-1")

        ctrl.load_tab_result.assert_not_called()
    finally:
        window.remove_tab_widget("tab-1")


def test_main_window_load_data_dialog_shows_user_facing_error(
    control_factory, qapp, monkeypatch
):
    from qtpy.QtWidgets import QFileDialog
    from zcu_tools.gui.app.measure.services.load import LoadDataError
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow
    from zcu_tools.gui.app.measure.ui.run_load_dialog import RunLoadDialog

    ctrl = _apply_window_defaults(_mock_ctrl(control_factory))
    ctrl.get_bus.return_value = EventBus()
    ctrl.has_tab.return_value = True
    ctrl.get_tab_snapshot.return_value = _snapshot(
        "tab-1", supports_load_data=True, has_figure=False
    )
    ctrl.save_control.list_saved_runs.return_value = ()
    ctrl.load_tab_result.side_effect = LoadDataError(
        "Cannot load this data file into the current tab.\n\nDetails: bad axes",
        reason_code="invalid_data_file",
    )
    window = MainWindow(ctrl, templates=lambda: {})
    window.add_tab_widget("tab-1", "fake")
    try:
        assert window.has_tab_widget("tab-1") is True
        window.show_error_dialog = MagicMock()
        monkeypatch.setattr(
            QFileDialog,
            "getOpenFileName",
            lambda *args, **kwargs: ("/tmp/bad.hdf5", ""),
        )

        def choose_file(dialog: RunLoadDialog) -> int:
            button = dialog.findChild(QPushButton, "load_file")
            assert button is not None
            button.click()
            return dialog.result()

        monkeypatch.setattr(RunLoadDialog, "exec", choose_file)
        window.load_tab_data_dialog("tab-1")

        window.show_error_dialog.assert_called_once()
        title, message = window.show_error_dialog.call_args.args
        assert title == "Load data failed"
        assert "Cannot load this data file" in message
        assert "bad axes" in message
    finally:
        window.remove_tab_widget("tab-1")


def test_main_window_run_lock_keeps_new_tab_available(
    control_factory, qapp, monkeypatch
):
    from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _mock_ctrl(control_factory)
    configure_form_lookup(ctrl)
    ctrl.get_bus.return_value = EventBus()
    ctrl.has_tab.return_value = True
    ctrl.get_tab_snapshot.side_effect = lambda tab_id: _snapshot(
        tab_id,
        global_run_active=tab_id != "tab-1",
        is_running=tab_id == "tab-1",
    )

    window = MainWindow(ctrl, templates=lambda: {})
    try:
        window.add_tab_widget("tab-1", "fake")
        window.add_tab_widget("tab-2", "fake")
        tabs = {tab.tab_id: tab for tab in window.findChildren(ExpTabWidget)}
        tab_one = tabs["tab-1"]
        tab_two = tabs["tab-2"]
        tab_one_update = MagicMock(wraps=tab_one.update_interaction_state)
        tab_two_update = MagicMock(wraps=tab_two.update_interaction_state)
        monkeypatch.setattr(tab_one, "update_interaction_state", tab_one_update)
        monkeypatch.setattr(tab_two, "update_interaction_state", tab_two_update)

        window.refresh_run_lock("tab-1")

        assert _new_tab_button(window).isEnabled() is True
        tab_one_update.assert_called_once()
        tab_two_update.assert_called_once()
    finally:
        window.remove_tab_widget("tab-1")
        window.remove_tab_widget("tab-2")


def test_main_window_tabs_are_movable_and_close_uses_moved_widget(
    control_factory, qapp
):
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(_editor_wiring_ctrl(control_factory))
    ctrl.get_bus.return_value = EventBus()
    ctrl.has_tab.side_effect = lambda tab_id: tab_id in {"tab-a", "tab-b"}
    ctrl.get_tab_snapshot.side_effect = lambda tab_id: _snapshot(
        tab_id, has_run_result=False
    )
    window = MainWindow(ctrl, templates=lambda: {})
    try:
        window.add_tab_widget("tab-a", "A")
        window.add_tab_widget("tab-b", "B")
        tab_a = _experiment_tabs(window).widget(0)
        assert tab_a is not None

        assert _experiment_tabs(window).isMovable() is True

        tab_bar = _experiment_tabs(window).tabBar()
        assert tab_bar is not None
        tab_bar.moveTab(0, 1)
        assert _experiment_tabs(window).widget(1) is tab_a
        ctrl.reorder_tabs.assert_called_once_with(["tab-b", "tab-a"])

        _experiment_tabs(window).tabCloseRequested.emit(1)

        ctrl.close_tab.assert_called_once_with("tab-a")
    finally:
        window.remove_tab_widget("tab-a")
        window.remove_tab_widget("tab-b")


def test_main_window_soc_changed_refreshes_run_lock(control_factory, qapp, monkeypatch):
    from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(_mock_ctrl(control_factory))
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_running_tab_id.return_value = None
    ctrl.has_tab.return_value = True
    ctrl.get_tab_snapshot.return_value = _snapshot("tab-1", has_soc=False)

    window = MainWindow(ctrl, templates=lambda: {})
    try:
        window.add_tab_widget("tab-1", "fake")
        tab = next(
            tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == "tab-1"
        )
        update_interaction_state = MagicMock(wraps=tab.update_interaction_state)
        monkeypatch.setattr(tab, "update_interaction_state", update_interaction_state)

        bus.emit(SocChangedPayload(soc=None, soccfg=None))

        update_interaction_state.assert_called()
    finally:
        window.remove_tab_widget("tab-1")


def test_main_window_content_event_queries_single_tab_snapshot(
    control_factory, qapp, monkeypatch
):
    from zcu_tools.gui.app.measure.events.tab import (
        TabContentChangedPayload,
        TabContentFact,
    )
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(_mock_ctrl(control_factory))
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    snapshot = _snapshot("tab-1")
    assert snapshot.analysis is not None
    ctrl.get_tab_snapshot.return_value = snapshot
    window = MainWindow(ctrl, templates=lambda: {})
    tab = MagicMock()
    try:
        window.add_tab_widget("tab-1", "fake")
        view = next(
            view for view in window.findChildren(ExpTabWidget) if view.tab_id == "tab-1"
        )
        monkeypatch.setattr(view.analyze_form, "sync", tab.analyze_form.sync)
        monkeypatch.setattr(view, "update_writeback_items", tab.update_writeback_items)
        ctrl.get_tab_snapshot.reset_mock()

        bus.emit(TabContentChangedPayload("tab-1", TabContentFact.RUN_RESULT_COMMITTED))

        ctrl.get_tab_snapshot.assert_called_once_with("tab-1")
        tab.analyze_form.sync.assert_called_once_with(snapshot.analysis.params)
        tab.update_writeback_items.assert_called_once_with(
            list(snapshot.analysis.writeback_items)
        )
    finally:
        window.remove_tab_widget("tab-1")


def test_main_window_interaction_event_refreshes_finished_analysis_figure(
    control_factory, qapp, monkeypatch
):
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.measure.events.tab import (
        TabInteractionChangedPayload,
        TabInteractionFact,
    )
    from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(_mock_ctrl(control_factory))
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.has_tab.return_value = True
    figure = Figure()
    writeback_item = MagicMock()
    ctrl.get_tab_snapshot.return_value = _snapshot(
        "tab-1",
        is_analyzing=False,
        has_analyze_result=True,
        has_figure=True,
        figure=figure,
        writeback_items=(writeback_item,),
    )
    window = MainWindow(ctrl, templates=lambda: {})
    try:
        window.add_tab_widget("tab-1", "fake")
        tab = next(
            tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == "tab-1"
        )
        update_writeback_items = MagicMock(wraps=tab.update_writeback_items)
        monkeypatch.setattr(tab, "update_writeback_items", update_writeback_items)
        show_analysis_figures = MagicMock(wraps=tab.show_analysis_figures)
        monkeypatch.setattr(tab, "show_analysis_figures", show_analysis_figures)
        ctrl.get_tab_snapshot.reset_mock()

        bus.emit(
            TabInteractionChangedPayload(
                "tab-1", TabInteractionFact.PRIMARY_ANALYZE_FAILED
            )
        )

        ctrl.get_tab_snapshot.assert_called_once_with("tab-1")
        update_writeback_items.assert_not_called()
        snapshot = ctrl.get_tab_snapshot.return_value
        assert snapshot.analysis is not None
        show_analysis_figures.assert_called_once_with(snapshot.analysis.figures)
    finally:
        window.remove_tab_widget("tab-1")


def test_main_window_interaction_event_does_not_restore_old_figure_on_analyze_start(
    control_factory,
    qapp,
    monkeypatch,
):
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.measure.events.tab import (
        TabInteractionChangedPayload,
        TabInteractionFact,
    )
    from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(_mock_ctrl(control_factory))
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.has_tab.return_value = True
    figure = Figure()
    ctrl.get_tab_snapshot.return_value = _snapshot(
        "tab-1",
        is_analyzing=True,
        has_analyze_result=True,
        has_figure=True,
        figure=figure,
    )
    window = MainWindow(ctrl, templates=lambda: {})
    try:
        window.add_tab_widget("tab-1", "fake")
        tab = next(
            tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == "tab-1"
        )
        show_analysis_figures = MagicMock(wraps=tab.show_analysis_figures)
        monkeypatch.setattr(tab, "show_analysis_figures", show_analysis_figures)
        ctrl.get_tab_snapshot.reset_mock()

        bus.emit(
            TabInteractionChangedPayload(
                "tab-1", TabInteractionFact.PRIMARY_ANALYZE_STARTED
            )
        )

        ctrl.get_tab_snapshot.assert_called_once_with("tab-1")
        show_analysis_figures.assert_not_called()
    finally:
        window.remove_tab_widget("tab-1")


def test_main_window_interaction_event_does_not_restore_old_figure_on_run_start(
    control_factory,
    qapp,
    monkeypatch,
):
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.measure.events.tab import (
        TabInteractionChangedPayload,
        TabInteractionFact,
    )
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(_mock_ctrl(control_factory))
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.has_tab.return_value = True
    figure = Figure()
    snapshot = _snapshot(
        "tab-1",
        is_running=True,
        is_analyzing=False,
        has_analyze_result=True,
        has_figure=True,
        figure=figure,
    )
    assert snapshot.analysis is not None
    ctrl.get_tab_snapshot.return_value = snapshot
    window = MainWindow(ctrl, templates=lambda: {})
    tab = MagicMock()
    try:
        window.add_tab_widget("tab-1", "fake")
        view = next(
            view for view in window.findChildren(ExpTabWidget) if view.tab_id == "tab-1"
        )
        monkeypatch.setattr(view.analyze_form, "sync", tab.analyze_form.sync)
        monkeypatch.setattr(view, "update_writeback_items", tab.update_writeback_items)
        monkeypatch.setattr(view, "show_analysis_figures", tab.show_analysis_figures)
        ctrl.get_tab_snapshot.reset_mock()

        bus.emit(
            TabInteractionChangedPayload("tab-1", TabInteractionFact.RUN_START_REJECTED)
        )

        ctrl.get_tab_snapshot.assert_called_once_with("tab-1")
        tab.show_analysis_figures.assert_not_called()
        tab.analyze_form.sync.assert_called_once_with(snapshot.analysis.params)
        tab.update_writeback_items.assert_called_once_with(
            list(snapshot.analysis.writeback_items)
        )
    finally:
        window.remove_tab_widget("tab-1")


def test_main_window_interaction_event_shows_post_figure_after_primary(
    control_factory, qapp, monkeypatch
):
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.measure.events.tab import (
        TabInteractionChangedPayload,
        TabInteractionFact,
    )
    from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(_mock_ctrl(control_factory))
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.has_tab.return_value = True
    primary = Figure()
    post = Figure()
    ctrl.get_tab_snapshot.return_value = _snapshot(
        "tab-1",
        is_analyzing=False,
        has_analyze_result=True,
        has_figure=True,
        has_post_analyze_result=True,
        supports_post_analysis=True,
        figure=primary,
        post_figure=post,
    )
    window = MainWindow(ctrl, templates=lambda: {})
    try:
        window.add_tab_widget("tab-1", "fake")
        tab = next(
            tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == "tab-1"
        )
        show_analysis_figures = MagicMock(wraps=tab.show_analysis_figures)
        monkeypatch.setattr(tab, "show_analysis_figures", show_analysis_figures)
        show_post_analysis_figures = MagicMock(wraps=tab.show_post_analysis_figures)
        monkeypatch.setattr(
            tab, "show_post_analysis_figures", show_post_analysis_figures
        )

        bus.emit(
            TabInteractionChangedPayload(
                "tab-1", TabInteractionFact.PRIMARY_ANALYZE_FAILED
            )
        )

        snapshot = ctrl.get_tab_snapshot.return_value
        assert snapshot.analysis is not None and snapshot.post_analysis is not None
        show_analysis_figures.assert_called_once_with(snapshot.analysis.figures)
        show_post_analysis_figures.assert_called_once_with(
            snapshot.post_analysis.figures
        )
    finally:
        window.remove_tab_widget("tab-1")


@pytest.mark.parametrize(
    "fact",
    [
        "primary_analyze_failed",
        "primary_analyze_cancelled",
        "primary_analyze_start_rejected",
        "post_analyze_failed",
        "post_analyze_start_rejected",
    ],
)
def test_analysis_terminal_restore_rebuilds_real_primary_then_post_canvas(
    control_factory, qapp, fact
):
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.measure.events.tab import (
        TabInteractionChangedPayload,
        TabInteractionFact,
    )
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget, MainWindow

    ctrl = _apply_window_defaults(_mock_ctrl(control_factory))
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.has_tab.return_value = True
    primary = Figure()
    post = Figure()
    snapshot = _snapshot(
        "tab-1",
        has_analyze_result=True,
        has_post_analyze_result=True,
        has_figure=True,
        supports_post_analysis=True,
        figure=primary,
        post_figure=post,
    )
    ctrl.get_tab_snapshot.return_value = snapshot
    window = MainWindow(ctrl, templates=lambda: {})
    try:
        window.add_tab_widget("tab-1", "fake")
        tab = next(
            tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == "tab-1"
        )
        assert snapshot.analysis is not None and snapshot.post_analysis is not None
        tab.show_analysis_figures(snapshot.analysis.figures)
        tab.show_post_analysis_figures(snapshot.post_analysis.figures)

        # Simulate operation start clearing only its pane; here clear both figures to emulate stale state
        tab.get_analysis_container().clear_dynamic_canvases()
        tab.get_post_container().clear_dynamic_canvases()
        assert tab.get_current_figure_for_pane("analysis") is None
        assert tab.get_current_figure_for_pane("post_analysis") is None

        bus.emit(TabInteractionChangedPayload("tab-1", TabInteractionFact(fact)))

        # Coordinator failure/cancel re-renders retained figures per pane
        assert tab.get_current_figure_for_pane("analysis") is primary
        assert tab.get_current_figure_for_pane("post_analysis") is post
    finally:
        window.remove_tab_widget("tab-1")


def test_loaded_content_clears_stale_real_canvas_when_state_has_no_figure(
    control_factory, qapp
):
    from dataclasses import dataclass

    from matplotlib.figure import Figure
    from zcu_tools.gui.app.measure.events.tab import (
        TabContentChangedPayload,
        TabContentFact,
    )
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget, MainWindow

    @dataclass
    class _Params:
        threshold: float = 0.5

    ctrl = _apply_window_defaults(_mock_ctrl(control_factory))
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.has_tab.return_value = True
    snapshot = _snapshot(
        "tab-1",
        has_run_result=True,
        has_analyze_result=False,
        has_figure=False,
        supports_post_analysis=True,
        analyze_params=_Params(),
        figure=None,
        post_figure=None,
    )
    ctrl.get_tab_snapshot.return_value = snapshot
    window = MainWindow(ctrl, templates=lambda: {})
    try:
        window.add_tab_widget("tab-1", "fake")
        tab = next(
            tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == "tab-1"
        )
        tab.show_analysis_figures(ready_figures(Figure()))
        assert tab.get_current_figure_for_pane("analysis") is not None

        bus.emit(
            TabContentChangedPayload("tab-1", TabContentFact.LOADED_RESULT_COMMITTED)
        )

        assert tab.get_current_figure_for_pane("analysis") is None
        assert tab.get_current_figure_for_pane("post_analysis") is None
    finally:
        window.remove_tab_widget("tab-1")


def _emit_run_finished(bus, tab_id: str, outcome: str) -> None:
    from zcu_tools.gui.app.measure.events.run import RunFinishedPayload

    bus.emit(RunFinishedPayload(tab_id=tab_id, outcome=outcome))


def test_finished_run_preserves_selected_subtab(control_factory, qapp, monkeypatch):
    """RUN_FINISHED refreshes state without choosing a result pane."""
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(_mock_ctrl(control_factory))
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_running_tab_id.return_value = None
    ctrl.get_tab_snapshot.return_value = _snapshot("tab-1", has_run_result=True)
    window = MainWindow(ctrl, templates=lambda: {})
    try:
        window.add_tab_widget("tab-1", "fake")
        tab = next(
            tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == "tab-1"
        )
        left_tabs = _left_tabs(tab)
        left_tabs.setCurrentIndex(2)
        assert left_tabs.currentIndex() == 2
        assert left_tabs.tabText(2) == "Data"
        focus_result_panel = MagicMock(wraps=tab.focus_result_panel)
        monkeypatch.setattr(tab, "focus_result_panel", focus_result_panel)

        _emit_run_finished(bus, "tab-1", outcome="finished")

        assert left_tabs.currentIndex() == 2
        focus_result_panel.assert_not_called()
    finally:
        window.remove_tab_widget("tab-1")


def test_stopped_run_does_not_auto_switch_to_analysis_tab(
    control_factory, qapp, monkeypatch
):
    """A stopped (cancelled) run may leave a partial result, but the user
    interrupted on purpose — must not yank them to the Analysis tab."""
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(_mock_ctrl(control_factory))
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_running_tab_id.return_value = None
    ctrl.get_tab_snapshot.return_value = _snapshot("tab-1", has_run_result=True)
    window = MainWindow(ctrl, templates=lambda: {})
    try:
        window.add_tab_widget("tab-1", "fake")
        tab = next(
            tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == "tab-1"
        )
        left_tabs = _left_tabs(tab)
        set_current_index = MagicMock(wraps=left_tabs.setCurrentIndex)
        monkeypatch.setattr(left_tabs, "setCurrentIndex", set_current_index)

        _emit_run_finished(bus, "tab-1", outcome="cancelled")

        set_current_index.assert_not_called()
    finally:
        window.remove_tab_widget("tab-1")


def test_non_analysis_adapter_run_preserves_selected_subtab(
    control_factory, qapp, monkeypatch
):
    """Non-analysis adapters also retain the user's selected pane after Run."""
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(_mock_ctrl(control_factory))
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_running_tab_id.return_value = None
    ctrl.get_tab_snapshot.return_value = _snapshot(
        "tab-1", has_run_result=True, supports_analysis=False, analyze_params=None
    )
    window = MainWindow(ctrl, templates=lambda: {})
    try:
        window.add_tab_widget("tab-1", "fake")
        tab = next(
            tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == "tab-1"
        )
        left_tabs = _left_tabs(tab)
        left_tabs.setCurrentIndex(1)
        assert left_tabs.currentIndex() == 1
        assert left_tabs.tabText(1) == "Data"
        focus_result_panel = MagicMock(wraps=tab.focus_result_panel)
        monkeypatch.setattr(tab, "focus_result_panel", focus_result_panel)

        _emit_run_finished(bus, "tab-1", outcome="finished")

        assert left_tabs.currentIndex() == 1
        focus_result_panel.assert_not_called()
    finally:
        window.remove_tab_widget("tab-1")


def test_refresh_analyze_form_skips_non_analysis_adapter_without_raising(
    control_factory, qapp, monkeypatch
):
    """A finished run on a non-analysis adapter has no analyze params; the
    content refresh must skip the analyze form rather than hit the Fast-Fail
    guard that demands initialized params (regression: it used to raise)."""
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(_mock_ctrl(control_factory))
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_running_tab_id.return_value = None
    ctrl.get_tab_snapshot.return_value = _snapshot(
        "tab-1", has_run_result=True, supports_analysis=False, analyze_params=None
    )
    window = MainWindow(ctrl, templates=lambda: {})
    try:
        window.add_tab_widget("tab-1", "fake")
        tab = next(
            tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == "tab-1"
        )
        populate_analyze_params = MagicMock(wraps=tab.populate_analyze_params)
        monkeypatch.setattr(tab, "populate_analyze_params", populate_analyze_params)

        # Must not raise "Run result has no initialized analyze parameters".
        window.refresh_tab_analyze_form("tab-1")

        populate_analyze_params.assert_not_called()
    finally:
        window.remove_tab_widget("tab-1")


def _editor_wiring_ctrl(control_factory: Callable[[], ControlIdentity]) -> MagicMock:
    """A view facade over real, caller-owned tab form resources."""
    ctrl = _mock_ctrl(control_factory)
    ctrl.active_operation_count.return_value = 0
    return ctrl


def test_exp_tab_attach_projects_existing_resource(control_factory, qapp):
    ctrl = _editor_wiring_ctrl(control_factory)
    cfg = ctrl.cfg_resources.lookup("tab-1")
    tab = ExpTabWidget("tab-1", ctrl, AdapterCapabilities(analysis=AnalysisMode.FIT))
    tab.attach(_snapshot("tab-1"), _RecordingTabActions())
    cfg.write({("gain",): 0.45})
    entry = tab.findChild(QLineEdit, "form.input.gain")
    assert entry is not None and float(entry.text()) == 0.45


def test_exp_tab_detach_preserves_resource_for_later_views(control_factory, qapp):
    ctrl = _editor_wiring_ctrl(control_factory)
    cfg = ctrl.cfg_resources.lookup("tab-1")
    tab = ExpTabWidget("tab-1", ctrl, AdapterCapabilities(analysis=AnalysisMode.FIT))
    tab.attach(_snapshot("tab-1"), _RecordingTabActions())
    tab.detach()
    cfg.write({("gain",): 0.45})
    other = ExpTabWidget("tab-1", ctrl, AdapterCapabilities(analysis=AnalysisMode.FIT))
    other.attach(_snapshot("tab-1"), _RecordingTabActions())
    entry = other.findChild(QLineEdit, "form.input.gain")
    assert entry is not None and float(entry.text()) == 0.45
    assert other.cfg_invalid_reason() is None


def test_exp_tab_buttons_dispatch_public_tab_actions(control_factory, qapp):
    import dataclasses

    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    ctrl = _editor_wiring_ctrl(control_factory)
    tab = ExpTabWidget(
        "tab-1",
        ctrl,
        AdapterCapabilities(
            analysis=AnalysisMode.FIT, post_analysis=True, load_data=True
        ),
    )
    actions = _RecordingTabActions()
    snapshot = dataclasses.replace(
        _snapshot(
            "tab-1",
            has_run_result=True,
            has_analyze_result=True,
            has_figure=True,
            has_post_analyze_result=True,
            supports_post_analysis=True,
            supports_load_data=True,
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


def test_exp_tab_reset_publishes_defaults_on_same_resource(control_factory, qapp):
    dialogs = RecordingDialogPresenter(confirm_answers=[True])
    ctrl = _editor_wiring_ctrl(control_factory)
    cfg = ctrl.cfg_resources.lookup("tab-1")
    defaults = cfg.input.data
    cfg.write({("gain",): 0.45})
    tab = ExpTabWidget(
        "tab-1",
        ctrl,
        AdapterCapabilities(analysis=AnalysisMode.FIT),
        dialog_presenter=dialogs,
    )
    actions = _RecordingTabActions()
    tab.attach(_snapshot("tab-1"), actions)
    tab.reset_btn.click()

    assert cfg.input.data == defaults
    assert actions.calls[-1] == ("refresh_interaction", "tab-1")


def test_exp_tab_reset_confirm_no_does_not_reset(control_factory, qapp):
    dialogs = RecordingDialogPresenter(confirm_answers=[False])
    ctrl = _editor_wiring_ctrl(control_factory)
    cfg = ctrl.cfg_resources.lookup("tab-1")
    before = cfg.input.data
    tab = ExpTabWidget(
        "tab-1",
        ctrl,
        AdapterCapabilities(analysis=AnalysisMode.FIT),
        dialog_presenter=dialogs,
    )
    actions = _RecordingTabActions()
    tab.attach(_snapshot("tab-1"), actions)
    actions.calls.clear()
    tab.reset_btn.click()
    assert cfg.input.data == before
    assert actions.calls == []


def test_exp_tab_reset_btn_idle_only_enable(control_factory, qapp):
    """reset_btn must be enabled when idle and disabled while the tab is busy."""
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        _mock_ctrl(control_factory),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )

    # Idle: reset_btn should be enabled.
    tab.update_interaction_state(_snapshot("tab-1", is_running=False))
    assert tab.reset_btn.isEnabled() is True

    # Busy (running): reset_btn must be disabled.
    tab.update_interaction_state(_snapshot("tab-1", is_running=True))
    assert tab.reset_btn.isEnabled() is False

    # Back to idle: reset_btn re-enabled.
    tab.update_interaction_state(_snapshot("tab-1", is_running=False))
    assert tab.reset_btn.isEnabled() is True


def test_main_window_confirms_and_begins_shutdown_when_operations_active(
    control_factory,
    qapp,
):
    """User close with work in progress: confirm, then begin_shutdown (which
    cancels-all and waits). The event is ignored now; begin_shutdown is deferred
    to the next event-loop turn and the coordinator drives the real close
    later."""
    from qtpy.QtCore import QCoreApplication
    from qtpy.QtGui import QCloseEvent
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    dialogs = RecordingDialogPresenter(confirm_answers=[True])
    ctrl = _mock_ctrl(control_factory)
    configure_form_lookup(ctrl)
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = 2
    window = MainWindow(ctrl, templates=lambda: {}, dialog_presenter=dialogs)
    event = QCloseEvent()

    window.closeEvent(event)

    assert dialogs.calls[-1].title == "Operations in progress"
    assert event.isAccepted() is False  # async wait — not closed yet
    QCoreApplication.processEvents()  # drain the deferred singleShot(0)
    _shutdown_callback(ctrl)


def test_main_window_declining_confirmation_keeps_window_open(control_factory, qapp):
    from qtpy.QtGui import QCloseEvent
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    dialogs = RecordingDialogPresenter(confirm_answers=[False])
    ctrl = _mock_ctrl(control_factory)
    configure_form_lookup(ctrl)
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = 1
    window = MainWindow(ctrl, templates=lambda: {}, dialog_presenter=dialogs)
    event = QCloseEvent()

    window.closeEvent(event)

    assert dialogs.calls[-1].title == "Operations in progress"
    assert event.isAccepted() is False
    ctrl.begin_shutdown.assert_not_called()


def test_main_window_persists_session_on_close_when_idle(control_factory, qapp):
    """Idle close: no confirmation; closeEvent ignores the event and *defers*
    begin_shutdown to the next event-loop turn (so it never re-enters
    self.close() within the closeEvent stack — the single-click bug). After the
    deferred turn fires, begin_shutdown runs _perform_close → persist."""
    from qtpy.QtCore import QCoreApplication
    from qtpy.QtGui import QCloseEvent
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _mock_ctrl(control_factory)
    configure_form_lookup(ctrl)
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = 0
    # The real coordinator runs on_closed once nothing is pending; here drive it
    # synchronously so the deferred turn exercises _perform_close.
    ctrl.begin_shutdown.side_effect = lambda on_closed: on_closed()
    window = MainWindow(ctrl, templates=lambda: {})
    event = QCloseEvent()

    window.closeEvent(event)

    # Deferred: closeEvent must NOT have begun shutdown synchronously.
    assert event.isAccepted() is False
    ctrl.begin_shutdown.assert_not_called()

    # Drain the singleShot(0) — the deferred turn runs begin_shutdown.
    QCoreApplication.processEvents()

    _shutdown_callback(ctrl)
    ctrl.persist_all.assert_called_once_with()


def test_main_window_close_removes_event_bus_subscriptions(
    control_factory, qapp, monkeypatch
):
    from qtpy.QtCore import QCoreApplication
    from qtpy.QtGui import QCloseEvent
    from zcu_tools.gui.app.measure.events.run import (
        RunFinishedPayload,
        RunStartedPayload,
    )
    from zcu_tools.gui.app.measure.events.tab import (
        TabAddedPayload,
        TabClosedPayload,
        TabContentChangedPayload,
        TabInteractionChangedPayload,
    )
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow
    from zcu_tools.gui.session.events import (
        ContextSwitchedPayload,
        SocChangedPayload,
    )

    ctrl = _mock_ctrl(control_factory)
    configure_form_lookup(ctrl)
    bus = EventBus()
    subscribe_spy = MagicMock(wraps=bus.subscribe)
    unsubscribe_spy = MagicMock(wraps=bus.unsubscribe)
    monkeypatch.setattr(bus, "subscribe", subscribe_spy)
    monkeypatch.setattr(bus, "unsubscribe", unsubscribe_spy)
    ctrl.get_bus.return_value = bus
    ctrl.active_operation_count.return_value = 0
    ctrl.begin_shutdown.side_effect = lambda on_closed: on_closed()
    window = MainWindow(ctrl, templates=lambda: {})

    payload_types = (
        TabInteractionChangedPayload,
        RunStartedPayload,
        RunFinishedPayload,
        ContextSwitchedPayload,
        TabAddedPayload,
        TabClosedPayload,
        TabContentChangedPayload,
        SocChangedPayload,
    )
    for payload_type in payload_types:
        subscription_calls = [
            call
            for call in subscribe_spy.call_args_list
            if call.args[0] is payload_type
        ]
        assert subscription_calls
    unsubscribe_spy.assert_not_called()

    window.closeEvent(QCloseEvent())
    QCoreApplication.processEvents()

    for payload_type in payload_types:
        subscription_calls = [
            call
            for call in subscribe_spy.call_args_list
            if call.args[0] is payload_type
        ]
        for subscription_call in subscription_calls:
            assert unsubscribe_spy.call_args_list.count(
                subscription_call
            ) == subscription_calls.count(subscription_call)


def test_new_tab_menu_supports_nested_paths(control_factory, qapp, monkeypatch):
    from qtpy.QtWidgets import QMenu
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    del qapp
    ctrl = _mock_ctrl(control_factory)
    configure_form_lookup(ctrl)
    ctrl.get_bus.return_value = EventBus()
    ctrl.get_adapter_names.return_value = [
        "fake",
        "twotone/rabi/length",
        "twotone/rabi/amp",
    ]
    window = MainWindow(ctrl, templates=lambda: {})

    def _find_action_by_data(menu: QMenu, target: str):
        for action in menu.actions():
            if action.data() == target:
                return action
            child = action.menu()
            if isinstance(child, QMenu):
                found = _find_action_by_data(child, target)
                if found is not None:
                    return found
        return None

    def _fake_exec(self, *_args, **_kwargs):
        action = _find_action_by_data(self, "twotone/rabi/length")
        assert action is not None
        return action

    monkeypatch.setattr(QMenu, "exec", _fake_exec)

    _new_tab_button(window).click()

    ctrl.new_tab.assert_called_once_with("twotone/rabi/length")


def test_show_analysis_figure_draws_canvas(control_factory, qapp, monkeypatch):
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    del qapp
    tab = ExpTabWidget(
        "tab-1",
        _mock_ctrl(control_factory),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )
    canvas = MagicMock()

    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.ui.exp_tab_widget.attach_existing_figure_to_container",
        lambda fig, container: canvas,
    )

    tab.show_analysis_figures(ready_figures(Figure()))

    canvas.draw.assert_called_once_with()


def _figure_pane_stack(tab: ExpTabWidget, accessible_name: str) -> QStackedWidget:
    """Find the stack beside the selector named by accessible_name within tab.

    Assert a unique selector, a non-null parent pane, and a unique stack in that
    pane; fail if any part of this public Qt lookup is missing or ambiguous.
    """
    selectors = [
        selector
        for selector in tab.findChildren(QComboBox)
        if selector.accessibleName() == accessible_name
    ]
    assert len(selectors) == 1
    (selector,) = selectors
    pane = selector.parentWidget()
    assert pane is not None
    stacks = pane.findChildren(QStackedWidget)
    assert len(stacks) == 1
    (stack,) = stacks
    return stack


def test_show_analysis_figure_keeps_two_figures_coexisting(control_factory, qapp):
    """Analysis and Post figures live in independent containers; each pane's history
    is distinct and showing one does not evict the other."""
    from matplotlib.figure import Figure
    from qtpy.QtWidgets import QApplication
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    del qapp
    tab = ExpTabWidget(
        "tab-1",
        _mock_ctrl(control_factory),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True),
    )
    tab.show()
    QApplication.processEvents()

    fig_a1 = Figure()
    fig_p1 = Figure()
    tab.show_analysis_figures(ready_figures(fig_a1))
    tab.show_post_analysis_figures(ready_figures(fig_p1))

    analysis_stack = _figure_pane_stack(tab, "Analysis figure")
    post_stack = _figure_pane_stack(tab, "Post-analysis figure")

    # Each pane has its own stack with placeholder + 1 canvas
    assert analysis_stack.count() == 2
    assert post_stack.count() == 2
    assert tab.get_current_figure_for_pane("analysis") is fig_a1
    assert tab.get_current_figure_for_pane("post_analysis") is fig_p1

    # Re-showing analysis does not affect post
    fig_a2 = Figure()
    tab.show_analysis_figures(ready_figures(fig_a2))
    assert tab.get_current_figure_for_pane("analysis") is fig_a2
    assert tab.get_current_figure_for_pane("post_analysis") is fig_p1
    assert post_stack.count() == 2


# ---------------------------------------------------------------------------
# Unsaved measurement data close guard tests (TKT-001)
# ---------------------------------------------------------------------------


def _setup_window_with_tabs(
    control_factory: Callable[[], ControlIdentity],
    dialog_presenter: RecordingDialogPresenter | None = None,
    active_operations: int = 0,
) -> tuple[MainWindow, MagicMock]:
    ctrl = _apply_window_defaults(_editor_wiring_ctrl(control_factory))
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = active_operations
    ctrl.has_tab.side_effect = lambda tid: True
    dialogs = dialog_presenter or RecordingDialogPresenter()
    window = MainWindow(ctrl, templates=lambda: {}, dialog_presenter=dialogs)
    return window, ctrl


def _add_tab(
    window: MainWindow,
    tab_id: str,
    *,
    ctrl: MagicMock,
    has_run: bool = False,
    saved: bool = False,
) -> ExpTabWidget:
    """Add a tab through the window while preserving the current selection."""
    tabs = _experiment_tabs(window)
    selected = tabs.currentWidget()
    snap = _snapshot(
        tab_id,
        has_run_result=has_run,
        data_status=SaveStatus.SAVED if saved else None,
    )
    original_snapshot = ctrl.get_tab_snapshot.return_value
    ctrl.get_tab_snapshot.return_value = snap
    try:
        window.add_tab_widget(tab_id, tab_id)
    finally:
        ctrl.get_tab_snapshot.return_value = original_snapshot
    tab = next(tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == tab_id)
    if selected is not None:
        tabs.setCurrentWidget(selected)
    return tab


def _shutdown_callback(ctrl: MagicMock) -> Callable[[], None]:
    ctrl.begin_shutdown.assert_called_once()
    callback = ctrl.begin_shutdown.call_args.args[0]
    assert callable(callback)
    return cast(Callable[[], None], callback)


class _DeferredClosePresenter(RecordingDialogPresenter):
    def __init__(self) -> None:
        super().__init__(confirm_answers=[True] * 4, destructive_answers=[True] * 4)
        self.confirm_decisions: list[Callable[[bool], None]] = []
        self.destructive_decisions: list[Callable[[bool], None]] = []

    def confirm_async(
        self,
        parent: Any,
        title: str,
        message: str,
        *,
        on_decision: Callable[[bool], None],
        default: bool = False,
    ) -> None:
        super().confirm_async(
            parent,
            title,
            message,
            on_decision=lambda _confirmed: None,
            default=default,
        )
        self.confirm_decisions.append(on_decision)

    def destructive_confirm(
        self,
        parent: Any,
        title: str,
        message: str,
        *,
        action_text: str,
        on_decision: Callable[[bool], None],
        default: bool = False,
        cancel_text: str = "Cancel",
    ) -> None:
        """Retain the decision callback while recording both button labels."""
        super().destructive_confirm(
            parent,
            title,
            message,
            action_text=action_text,
            on_decision=lambda _confirmed: None,
            default=default,
            cancel_text=cancel_text,
        )
        self.destructive_decisions.append(on_decision)


def test_window_drains_before_persisting_and_stopping_remote(control_factory, qapp):
    dialogs = RecordingDialogPresenter()
    window, ctrl = _setup_window_with_tabs(control_factory, dialogs)
    effects: list[str] = []
    ctrl.close.side_effect = lambda: effects.append("drained")
    ctrl.persist_all.side_effect = lambda: effects.append("persisted")
    remote = MagicMock()
    remote.stop.side_effect = lambda: effects.append("remote stopped")
    window.remote_control_service = remote
    ctrl.begin_shutdown.side_effect = lambda on_closed: on_closed()
    window.show()

    window.close()
    qapp.processEvents()

    assert effects == ["drained", "persisted", "remote stopped"]
    assert not window.isVisible()
    assert dialogs.calls == []


@pytest.mark.parametrize("programmatic", [False, True])
def test_window_timeout_waits_for_decision_and_retries(
    control_factory, qapp, programmatic
):
    dialogs = _DeferredClosePresenter()
    window, ctrl = _setup_window_with_tabs(control_factory, dialogs)
    ctrl.begin_shutdown.side_effect = lambda on_closed: on_closed()
    ctrl.close.side_effect = [
        TimeoutError("still running"),
        TimeoutError("still running"),
        None,
    ]
    remote = MagicMock()
    window.remote_control_service = remote
    window.show()

    if programmatic:
        window.request_shutdown()
    else:
        window.close()
    qapp.processEvents()

    assert window.isVisible()
    ctrl.persist_all.assert_not_called()
    remote.stop.assert_not_called()
    assert len(dialogs.destructive_decisions) == 1
    assert "背景工作仍在執行" in dialogs.calls[-1].message
    assert dialogs.calls[-1].action_text == "強制關閉"
    assert dialogs.calls[-1].cancel_text == "再等一下"

    # Reentrant GUI/RPC requests cannot turn a pending decision into force.
    window.close()
    window.request_shutdown()
    window.request_shutdown()
    qapp.processEvents()
    assert window.isVisible()
    assert len(dialogs.destructive_decisions) == 1
    assert ctrl.close.call_count == 1

    dialogs.destructive_decisions.pop(0)(False)
    qapp.processEvents()
    assert window.isVisible()
    assert len(dialogs.destructive_decisions) == 1
    assert ctrl.close.call_count == 2
    ctrl.persist_all.assert_not_called()
    remote.stop.assert_not_called()

    dialogs.destructive_decisions.pop(0)(False)
    qapp.processEvents()
    assert not window.isVisible()
    ctrl.persist_all.assert_called_once()
    remote.stop.assert_called_once()
    assert ctrl.close.call_count == 3


@pytest.mark.parametrize("programmatic", [False, True])
def test_window_timeout_requires_explicit_force(control_factory, qapp, programmatic):
    dialogs = _DeferredClosePresenter()
    window, ctrl = _setup_window_with_tabs(control_factory, dialogs)
    ctrl.begin_shutdown.side_effect = lambda on_closed: on_closed()
    ctrl.close.side_effect = TimeoutError("still running")
    remote = MagicMock()
    window.remote_control_service = remote
    window.show()

    if programmatic:
        window.request_shutdown()
    else:
        window.close()
    qapp.processEvents()

    assert window.isVisible()
    ctrl.persist_all.assert_not_called()
    remote.stop.assert_not_called()
    assert len(dialogs.destructive_decisions) == 1
    dialogs.destructive_decisions.pop(0)(True)
    qapp.processEvents()

    assert not window.isVisible()
    ctrl.persist_all.assert_called_once()
    remote.stop.assert_called_once()
    assert ctrl.close.call_count == 1


def test_tab_close_without_result_closes_immediately(control_factory, qapp):
    """Scenario 1: Tab with NO_RESULT closes without prompting."""
    dialogs = RecordingDialogPresenter()
    window, ctrl = _setup_window_with_tabs(control_factory, dialogs)
    _add_tab(window, "tab-1", has_run=False, ctrl=ctrl)

    _experiment_tabs(window).tabCloseRequested.emit(0)

    assert len(dialogs.calls) == 0
    ctrl.close_tab.assert_called_once_with("tab-1")


def test_tab_close_with_saved_data_closes_immediately(control_factory, qapp):
    """Scenario 1: Tab with SAVED data closes without prompting."""
    dialogs = RecordingDialogPresenter()
    window, ctrl = _setup_window_with_tabs(control_factory, dialogs)
    tab = _add_tab(window, "tab-1", has_run=True, saved=True, ctrl=ctrl)
    assert tab.has_unsaved_data() is False

    _experiment_tabs(window).tabCloseRequested.emit(0)

    assert len(dialogs.calls) == 0
    ctrl.close_tab.assert_called_once_with("tab-1")


@pytest.mark.parametrize(
    "data_status", [SaveStatus.NOT_SAVED, SaveStatus.UNSAVED_CHANGES]
)
def test_tab_close_with_unsaved_data_declined_keeps_tab(
    control_factory, qapp, data_status
):
    """Cancelling tab-close leaves either unsaved data state untouched."""
    dialogs = RecordingDialogPresenter(destructive_answers=[False])
    window, ctrl = _setup_window_with_tabs(control_factory, dialogs)
    tab = _add_tab(window, "tab-1", has_run=True, saved=False, ctrl=ctrl)
    tab.update_interaction_state(_snapshot("tab-1", data_status=data_status))
    assert tab.has_unsaved_data() is True

    _experiment_tabs(window).tabCloseRequested.emit(0)

    assert len(dialogs.calls) == 1
    call = dialogs.calls[0]
    assert call.kind == "destructive_confirm"
    assert call.title == "Unsaved measurement data"
    assert call.action_text == "Discard and Close"
    ctrl.close_tab.assert_not_called()
    assert window.has_tab_widget("tab-1") is True


def test_tab_close_with_unsaved_data_confirmed_closes_tab(control_factory, qapp):
    """Scenario 2: Confirming tab-close closes the unsaved tab."""
    dialogs = RecordingDialogPresenter(destructive_answers=[True])
    window, ctrl = _setup_window_with_tabs(control_factory, dialogs)
    tab = _add_tab(window, "tab-1", has_run=True, saved=False, ctrl=ctrl)
    assert tab.has_unsaved_data() is True

    _experiment_tabs(window).tabCloseRequested.emit(0)

    assert len(dialogs.calls) == 1
    call = dialogs.calls[0]
    assert call.kind == "destructive_confirm"
    assert call.title == "Unsaved measurement data"
    assert call.action_text == "Discard and Close"
    ctrl.close_tab.assert_called_once_with("tab-1")


def test_moved_tab_with_unsaved_data_closes_by_captured_identity(control_factory, qapp):
    """Scenario 3: A tab moved before close request closes by captured widget identity."""
    dialogs = RecordingDialogPresenter(destructive_answers=[True])
    window, ctrl = _setup_window_with_tabs(control_factory, dialogs)
    tab_a = _add_tab(window, "tab-a", has_run=True, saved=False, ctrl=ctrl)
    _add_tab(window, "tab-b", has_run=False, ctrl=ctrl)
    assert tab_a.has_unsaved_data() is True

    tab_bar = _experiment_tabs(window).tabBar()
    assert tab_bar is not None
    # Move tab-a from index 0 to index 1
    tab_bar.moveTab(0, 1)
    assert _experiment_tabs(window).widget(1) is tab_a

    # Request close on moved tab (now at index 1)
    _experiment_tabs(window).tabCloseRequested.emit(1)

    assert len(dialogs.calls) == 1
    assert dialogs.calls[0].kind == "destructive_confirm"
    assert dialogs.calls[0].title == "Unsaved measurement data"
    ctrl.close_tab.assert_called_once_with("tab-a")


def test_tab_moved_while_confirmation_pending_closes_captured_widget(
    control_factory, qapp
):
    """Scenario 3: Tab moved while confirmation prompt is open closes captured identity."""
    decision_cb: list[Callable[[bool], None]] = []

    class _DeferredPresenter(RecordingDialogPresenter):
        def destructive_confirm(
            self,
            parent: Any,
            title: str,
            message: str,
            *,
            action_text: str,
            on_decision: Callable[[bool], None],
            default: bool = False,
            cancel_text: str = "Cancel",
        ) -> None:
            """Retain the decision and preserve the alternative button label."""
            super().destructive_confirm(
                parent,
                title,
                message,
                action_text=action_text,
                on_decision=lambda _: None,
                default=default,
                cancel_text=cancel_text,
            )
            decision_cb.append(on_decision)

    deferred_presenter = _DeferredPresenter(destructive_answers=[True])
    window, ctrl = _setup_window_with_tabs(control_factory, deferred_presenter)
    tab_a = _add_tab(window, "tab-a", has_run=True, saved=False, ctrl=ctrl)
    _add_tab(window, "tab-b", has_run=False, ctrl=ctrl)

    tabs = window.findChild(QTabWidget)
    assert tabs is not None
    # Request close on tab-a at index 0 via signal
    tabs.tabCloseRequested.emit(0)
    assert len(decision_cb) == 1

    # While dialog is pending, move tab-a to index 1
    tab_bar = tabs.tabBar()
    assert tab_bar is not None
    tab_bar.moveTab(0, 1)
    assert tabs.widget(1) is tab_a

    # User confirms the dialog
    decision_cb[0](True)

    ctrl.close_tab.assert_called_once_with("tab-a")


def test_tab_close_suppresses_duplicate_prompt_while_decision_pending(
    control_factory, qapp
):
    """Scenario 2/3: Duplicate tab close requests for same tab are suppressed while prompt is open."""
    decision_cb: list[Callable[[bool], None]] = []

    class _DeferredPresenter(RecordingDialogPresenter):
        def destructive_confirm(
            self,
            parent: Any,
            title: str,
            message: str,
            *,
            action_text: str,
            on_decision: Callable[[bool], None],
            default: bool = False,
            cancel_text: str = "Cancel",
        ) -> None:
            """Retain the decision and preserve the alternative button label."""
            super().destructive_confirm(
                parent,
                title,
                message,
                action_text=action_text,
                on_decision=lambda _: None,
                default=default,
                cancel_text=cancel_text,
            )
            decision_cb.append(on_decision)

    deferred_presenter = _DeferredPresenter(destructive_answers=[True])
    window, ctrl = _setup_window_with_tabs(control_factory, deferred_presenter)
    _add_tab(window, "tab-1", has_run=True, saved=False, ctrl=ctrl)

    tabs = window.findChild(QTabWidget)
    assert tabs is not None
    # First request
    tabs.tabCloseRequested.emit(0)
    assert len(deferred_presenter.calls) == 1

    # Second request while prompt is pending
    tabs.tabCloseRequested.emit(0)
    assert len(deferred_presenter.calls) == 1

    # Confirm
    decision_cb[0](True)
    ctrl.close_tab.assert_called_once_with("tab-1")


def test_app_close_idle_with_no_unsaved_tabs_shuts_down_without_prompt(
    control_factory, qapp
):
    """Scenario 6: App close with neither unsaved data nor active operations closes cleanly."""
    dialogs = RecordingDialogPresenter()
    window, ctrl = _setup_window_with_tabs(
        control_factory, dialogs, active_operations=0
    )
    _add_tab(window, "tab-1", has_run=True, saved=True, ctrl=ctrl)

    window.close()

    assert len(dialogs.calls) == 0
    QCoreApplication.processEvents()
    _shutdown_callback(ctrl)


def test_closing_window_leaves_plot_host_available_until_app_quits(
    control_factory, qapp
):
    window, _ctrl = _setup_window_with_tabs(
        control_factory, RecordingDialogPresenter(), active_operations=0
    )
    window.close()
    QCoreApplication.processEvents()

    stack = QStackedWidget()
    placeholder = QLabel("empty")
    stack.addWidget(placeholder)
    container = FigureContainer(stack, placeholder)
    owner = QtOwnerScheduler()
    try:
        plots = Plots(QtPlotHost(container, owner))
        plots.subplots("fresh")
        plots.finish()
        assert stack.count() == 2
        plots.release()
        assert stack.count() == 1
    finally:
        container.clear_dynamic_canvases()
        owner.deleteLater()
        stack.deleteLater()
        window.deleteLater()
        qapp.processEvents()


def test_app_close_with_unsaved_tabs_prompts_and_declining_cancels(
    control_factory, qapp
):
    """Scenario 4: App close with unsaved data warns; cancelling leaves app open."""
    dialogs = RecordingDialogPresenter(destructive_answers=[False])
    window, ctrl = _setup_window_with_tabs(
        control_factory, dialogs, active_operations=0
    )
    _add_tab(window, "tab-1", has_run=True, saved=False, ctrl=ctrl)

    window.close()

    assert len(dialogs.calls) == 1
    call = dialogs.calls[0]
    assert call.kind == "destructive_confirm"
    assert call.title == "Unsaved measurement data"
    assert call.action_text == "Discard and Close"
    QCoreApplication.processEvents()
    ctrl.begin_shutdown.assert_not_called()


def test_app_close_with_unsaved_tabs_prompts_and_confirming_shuts_down(
    control_factory, qapp
):
    """Scenario 4: App close with unsaved data warns; confirming begins shutdown."""
    dialogs = RecordingDialogPresenter(destructive_answers=[True])
    window, ctrl = _setup_window_with_tabs(
        control_factory, dialogs, active_operations=0
    )
    _add_tab(window, "tab-1", has_run=True, saved=False, ctrl=ctrl)
    _add_tab(window, "tab-2", has_run=False, ctrl=ctrl)

    window.close()

    assert len(dialogs.calls) == 1
    call = dialogs.calls[0]
    assert call.kind == "destructive_confirm"
    assert call.title == "Unsaved measurement data"
    assert call.action_text == "Discard and Close"
    QCoreApplication.processEvents()
    _shutdown_callback(ctrl)


def test_app_close_with_both_unsaved_tabs_and_active_operations(control_factory, qapp):
    """Scenario 5: Both risks present; prompt communicates both, confirm shuts down."""
    dialogs = RecordingDialogPresenter(destructive_answers=[True])
    window, ctrl = _setup_window_with_tabs(
        control_factory, dialogs, active_operations=3
    )
    _add_tab(window, "tab-1", has_run=True, saved=False, ctrl=ctrl)

    window.close()

    assert len(dialogs.calls) == 1
    call = dialogs.calls[0]
    assert call.kind == "destructive_confirm"
    assert call.title == "Unsaved data and operations in progress"
    assert "unsaved measurement data" in call.message.lower()
    assert "3 operation(s)" in call.message
    assert call.action_text == "Discard and Close"
    QCoreApplication.processEvents()
    _shutdown_callback(ctrl)


def test_app_close_with_both_unsaved_tabs_and_active_operations_declined(
    control_factory, qapp
):
    """Scenario 5: Both risks present; decline preserves app state."""
    dialogs = RecordingDialogPresenter(destructive_answers=[False])
    window, ctrl = _setup_window_with_tabs(
        control_factory, dialogs, active_operations=2
    )
    _add_tab(window, "tab-1", has_run=True, saved=False, ctrl=ctrl)

    window.close()

    assert len(dialogs.calls) == 1
    call = dialogs.calls[0]
    assert call.kind == "destructive_confirm"
    assert call.title == "Unsaved data and operations in progress"
    assert call.action_text == "Discard and Close"
    QCoreApplication.processEvents()
    ctrl.begin_shutdown.assert_not_called()


def test_app_close_with_active_operations_only_uses_confirm_async(
    control_factory, qapp
):
    """Active-operation-only prompt uses non-blocking confirm_async."""
    dialogs = RecordingDialogPresenter(confirm_answers=[True])
    window, ctrl = _setup_window_with_tabs(
        control_factory, dialogs, active_operations=2
    )
    _add_tab(window, "tab-1", has_run=True, saved=True, ctrl=ctrl)

    window.close()

    assert len(dialogs.calls) == 1
    call = dialogs.calls[0]
    assert call.kind == "confirm"
    assert call.title == "Operations in progress"
    assert "2 operation(s)" in call.message
    QCoreApplication.processEvents()
    _shutdown_callback(ctrl)


def test_app_close_suppresses_duplicate_prompt_while_decision_pending(
    control_factory, qapp
):
    """Scenario 4: Duplicate app close requests while prompt is open are suppressed."""
    decision_cb: list[Callable[[bool], None]] = []

    class _DeferredPresenter(RecordingDialogPresenter):
        def destructive_confirm(
            self,
            parent: Any,
            title: str,
            message: str,
            *,
            action_text: str,
            on_decision: Callable[[bool], None],
            default: bool = False,
            cancel_text: str = "Cancel",
        ) -> None:
            """Retain the decision and preserve the alternative button label."""
            super().destructive_confirm(
                parent,
                title,
                message,
                action_text=action_text,
                on_decision=lambda _: None,
                default=default,
                cancel_text=cancel_text,
            )
            decision_cb.append(on_decision)

    deferred_presenter = _DeferredPresenter(destructive_answers=[True])
    window, ctrl = _setup_window_with_tabs(
        control_factory, deferred_presenter, active_operations=0
    )
    _add_tab(window, "tab-1", has_run=True, saved=False, ctrl=ctrl)

    # First close
    window.close()
    assert len(deferred_presenter.calls) == 1

    # Second close while dialog is pending
    window.close()
    assert len(deferred_presenter.calls) == 1

    decision_cb[0](True)
    QCoreApplication.processEvents()
    _shutdown_callback(ctrl)


def test_app_close_reprompts_if_unsaved_data_appears_before_decision(
    control_factory, qapp
):
    """A newly completed result upgrades an active-only prompt before shutdown."""
    dialogs = _DeferredClosePresenter()
    window, ctrl = _setup_window_with_tabs(
        control_factory, dialogs, active_operations=1
    )
    tab = _add_tab(window, "tab-1", has_run=True, saved=True, ctrl=ctrl)

    window.close()

    assert [call.kind for call in dialogs.calls] == ["confirm"]
    tab.update_interaction_state(_snapshot("tab-1", has_run_result=True))
    ctrl.active_operation_count.return_value = 0
    dialogs.confirm_decisions.pop()(True)

    assert [call.kind for call in dialogs.calls] == [
        "confirm",
        "destructive_confirm",
    ]
    assert dialogs.calls[-1].title == "Unsaved measurement data"
    ctrl.begin_shutdown.assert_not_called()

    dialogs.destructive_decisions.pop()(False)
    ctrl.begin_shutdown.assert_not_called()


def test_app_close_suppresses_reentry_while_shutdown_is_coordinating(
    control_factory, qapp
):
    """Accepted close owns the lifecycle until the coordinator callback runs."""
    dialogs = RecordingDialogPresenter(confirm_answers=[True])
    window, ctrl = _setup_window_with_tabs(
        control_factory, dialogs, active_operations=1
    )
    _add_tab(window, "tab-1", has_run=True, saved=True, ctrl=ctrl)

    window.close()
    window.close()
    QCoreApplication.processEvents()
    _shutdown_callback(ctrl)

    window.close()

    assert len(dialogs.calls) == 1
    ctrl.begin_shutdown.assert_called_once()


def test_app_close_suppresses_duplicate_idle_shutdown_requests(control_factory, qapp):
    """Repeated idle close events schedule only one shutdown coordinator run."""
    dialogs = RecordingDialogPresenter()
    window, ctrl = _setup_window_with_tabs(
        control_factory, dialogs, active_operations=0
    )
    _add_tab(window, "tab-1", has_run=True, saved=True, ctrl=ctrl)

    window.close()
    window.close()
    QCoreApplication.processEvents()
    _shutdown_callback(ctrl)

    window.close()

    assert dialogs.calls == []
    ctrl.begin_shutdown.assert_called_once()


def test_app_close_rechecks_unsaved_data_after_operations_settle(control_factory, qapp):
    """A result completed during shutdown gets a final data-loss decision."""
    dialogs = RecordingDialogPresenter(
        confirm_answers=[True], destructive_answers=[False]
    )
    window, ctrl = _setup_window_with_tabs(
        control_factory, dialogs, active_operations=1
    )
    tab = _add_tab(window, "tab-1", has_run=True, saved=True, ctrl=ctrl)

    window.close()
    QCoreApplication.processEvents()
    on_closed = _shutdown_callback(ctrl)

    tab.update_interaction_state(_snapshot("tab-1", has_run_result=True))
    ctrl.active_operation_count.return_value = 0
    on_closed()

    assert [call.kind for call in dialogs.calls] == [
        "confirm",
        "destructive_confirm",
    ]
    assert "completed with unsaved measurement data" in dialogs.calls[-1].message
    ctrl.persist_all.assert_not_called()

    dialogs.queue_destructive_confirm(False)
    window.close()
    assert [call.kind for call in dialogs.calls] == [
        "confirm",
        "destructive_confirm",
        "destructive_confirm",
    ]


def test_app_close_waits_for_pending_tab_close_decision(control_factory, qapp):
    """Tab and app close confirmations never overlap."""
    dialogs = _DeferredClosePresenter()
    window, ctrl = _setup_window_with_tabs(
        control_factory, dialogs, active_operations=0
    )
    _add_tab(window, "tab-1", has_run=True, saved=False, ctrl=ctrl)

    _experiment_tabs(window).tabCloseRequested.emit(0)
    window.close()

    assert [call.kind for call in dialogs.calls] == ["destructive_confirm"]
    ctrl.begin_shutdown.assert_not_called()

    dialogs.destructive_decisions.pop()(False)


def test_repeated_programmatic_shutdown_waits_for_pre_prompt_coordination(
    control_factory, qapp
):
    """Repeated RPC takeover cannot bypass the captured shutdown coordinator."""
    dialogs = _DeferredClosePresenter()
    window, ctrl = _setup_window_with_tabs(
        control_factory, dialogs, active_operations=1
    )
    _add_tab(window, "tab-1", has_run=True, saved=True, ctrl=ctrl)

    window.close()
    assert [call.kind for call in dialogs.calls] == ["confirm"]

    window.request_shutdown()
    QCoreApplication.processEvents()
    on_closed = _shutdown_callback(ctrl)

    window.request_shutdown()
    QCoreApplication.processEvents()
    ctrl.persist_all.assert_not_called()

    dialogs.confirm_decisions.pop()(True)
    ctrl.persist_all.assert_not_called()

    on_closed()
    ctrl.persist_all.assert_called_once_with()


def test_programmatic_shutdown_supersedes_user_shutdown_coordination(
    control_factory, qapp
):
    """RPC intent forces close when user-started coordination later settles."""
    dialogs = RecordingDialogPresenter(confirm_answers=[True])
    window, ctrl = _setup_window_with_tabs(
        control_factory, dialogs, active_operations=1
    )
    tab = _add_tab(window, "tab-1", has_run=True, saved=True, ctrl=ctrl)

    window.close()
    QCoreApplication.processEvents()
    on_closed = _shutdown_callback(ctrl)

    window.request_shutdown()
    tab.update_interaction_state(_snapshot("tab-1", has_run_result=True))
    ctrl.active_operation_count.return_value = 0
    on_closed()

    assert [call.kind for call in dialogs.calls] == ["confirm"]
    ctrl.begin_shutdown.assert_called_once()
    ctrl.persist_all.assert_called_once_with()


def test_programmatic_shutdown_supersedes_final_unsaved_prompt(control_factory, qapp):
    """RPC intent tears down without waiting for a post-settlement dialog."""
    dialogs = _DeferredClosePresenter()
    window, ctrl = _setup_window_with_tabs(
        control_factory, dialogs, active_operations=1
    )
    tab = _add_tab(window, "tab-1", has_run=True, saved=True, ctrl=ctrl)

    window.close()
    dialogs.confirm_decisions.pop()(True)
    QCoreApplication.processEvents()
    on_closed = _shutdown_callback(ctrl)

    tab.update_interaction_state(_snapshot("tab-1", has_run_result=True))
    ctrl.active_operation_count.return_value = 0
    on_closed()
    assert [call.kind for call in dialogs.calls] == [
        "confirm",
        "destructive_confirm",
    ]

    window.request_shutdown()
    ctrl.persist_all.assert_not_called()
    QCoreApplication.processEvents()
    ctrl.persist_all.assert_called_once_with()

    dialogs.destructive_decisions.pop()(False)
    ctrl.persist_all.assert_called_once_with()


def test_programmatic_request_shutdown_bypasses_unsaved_guard(control_factory, qapp):
    """Programmatic request_shutdown() remains non-interactive even with unsaved data."""
    dialogs = RecordingDialogPresenter()
    window, ctrl = _setup_window_with_tabs(
        control_factory, dialogs, active_operations=1
    )
    _add_tab(window, "tab-1", has_run=True, saved=False, ctrl=ctrl)

    window.request_shutdown()

    assert len(dialogs.calls) == 0
    ctrl.begin_shutdown.assert_not_called()
    QCoreApplication.processEvents()
    ctrl.close.assert_not_called()
    ctrl.persist_all.assert_not_called()
    _shutdown_callback(ctrl)()
    ctrl.close.assert_called_once_with()
    ctrl.persist_all.assert_called_once_with()
    assert len(dialogs.calls) == 0


class ValueCfg(ExpCfgModel):
    """One float the Run panel edits and Run accepts."""

    value: float = 2.0


@dataclass
class RunFormFixture:
    window: MainWindow
    ctrl: MagicMock
    owner: TabFormResource
    tab: ExpTabWidget
    line: QLineEdit
    runs: list[AcceptedFormConfig]


@pytest.fixture
def run_form(
    control_factory, qapp: QApplication, request: pytest.FixtureRequest
) -> Iterator[RunFormFixture]:
    ctrl = _apply_window_defaults(_mock_ctrl(control_factory))
    ctrl.bus = EventBus()
    snapshot = _snapshot(
        "pending-tab",
        supports_analysis=False,
        has_run_result=False,
        has_analyze_result=False,
        has_figure=False,
    )
    ctrl.get_tab_snapshot.return_value = snapshot
    sources = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    owner = make_form(Form(ValueCfg, {}), state=sources)
    ctrl.cfg_resources.lookup.side_effect = lambda tab_id: owner
    runs: list[AcceptedFormConfig] = []

    def start(tab_id: str) -> None:
        runs.append(owner.accept())

    ctrl.start_run.side_effect = start
    window = MainWindow(
        ctrl, templates=lambda: {}, dialog_presenter=RecordingDialogPresenter()
    )
    window.add_tab_widget("pending-tab", "fake")
    window.resize(1100, 700)
    window.show()
    qapp.processEvents()
    tabs = window.findChildren(ExpTabWidget)
    assert len(tabs) == 1
    tab = tabs[0]
    line = tab.findChild(QLineEdit, "form.input.value")
    assert line is not None
    try:
        yield RunFormFixture(window, ctrl, owner, tab, line, runs)
    finally:
        window.remove_tab_widget("pending-tab")
        window.deleteLater()
        qapp.processEvents()


def edit_run_input(fx: RunFormFixture, text: str) -> None:
    """Replace the Value text and press Enter; the editor parses each commit."""
    fx.line.setFocus()
    commit_text(fx.line, text)


def test_run_uses_the_input_published_on_commit(
    run_form: RunFormFixture,
) -> None:
    fx = run_form
    edit_run_input(fx, "3.5")
    assert fx.tab.run_btn.isEnabled()
    fx.tab.run_btn.click()
    assert len(fx.runs) == 1
    assert fx.runs[0].values == {"value": 3.5}


@pytest.mark.parametrize("text", ["-", "nan"], ids=["incomplete", "nonfinite"])
def test_run_publishes_invalid_input_but_never_runs_old_values(
    run_form: RunFormFixture,
    text: str,
) -> None:
    fx = run_form
    edit_run_input(fx, text)
    fx.window.run_or_stop_tab("pending-tab")
    assert fx.runs == []
    assert fx.tab.cfg_invalid_reason() is not None
    assert fx.line.text() == text


def test_run_uses_the_external_update_shown_in_the_form(
    run_form: RunFormFixture,
) -> None:
    fx = run_form
    edit_run_input(fx, "3.51")
    fx.owner.write({("value",): 8.0})
    assert fx.line.text() == "8.0"
    fx.window.run_or_stop_tab("pending-tab")
    assert len(fx.runs) == 1
    assert fx.runs[0].values == {"value": 8.0}


def test_run_can_submit_a_repair_to_invalid_published_config(
    run_form: RunFormFixture,
) -> None:
    fx = run_form
    fx.owner.write({("value",): {"expr": "unfinished +"}})
    assert not fx.tab.run_btn.isEnabled()
    edit_run_input(fx, "4.5")
    assert fx.tab.run_btn.isEnabled()
    fx.tab.run_btn.click()
    assert len(fx.runs) == 1
    assert fx.runs[0].values == {"value": 4.5}


@pytest.mark.parametrize("block", ["busy", "context", "soc", "global-run"])
def test_valid_input_does_not_bypass_other_run_gates(
    run_form: RunFormFixture,
    block: str,
) -> None:
    fx = run_form
    fx.ctrl.get_tab_snapshot.return_value = _snapshot(
        "pending-tab",
        supports_analysis=False,
        has_run_result=False,
        has_analyze_result=False,
        has_figure=False,
        is_analyzing=block == "busy",
        has_active_context=block != "context",
        has_soc=block != "soc",
        global_run_active=block == "global-run",
    )
    edit_run_input(fx, "3.5")
    assert not fx.tab.run_btn.isEnabled()
    assert fx.runs == []


def test_stop_does_not_publish_or_run(run_form: RunFormFixture) -> None:
    fx = run_form
    edit_run_input(fx, "3.5")
    fx.ctrl.get_tab_snapshot.return_value = _snapshot(
        "pending-tab",
        supports_analysis=False,
        is_running=True,
    )
    fx.window.run_or_stop_tab("pending-tab")
    assert fx.runs == []
    fx.ctrl.cancel_run.assert_called_once_with()


def test_reset_replaces_published_input_with_defaults(
    run_form: RunFormFixture,
) -> None:
    fx = run_form
    dialogs = RecordingDialogPresenter(confirm_answers=[True])
    # Reopen through the public view lifecycle using the same cfg owner.
    fx.window.remove_tab_widget("pending-tab")
    fx.window.deleteLater()
    fx.window = MainWindow(fx.ctrl, templates=lambda: {}, dialog_presenter=dialogs)
    fx.window.add_tab_widget("pending-tab", "fake")
    tab = fx.window.findChildren(ExpTabWidget)[0]
    line = tab.findChild(QLineEdit, "form.input.value")
    assert line is not None
    commit_text(line, "3.5")
    assert fx.owner.accept().values == {"value": 3.5}
    tab.reset_btn.click()
    line = tab.findChild(QLineEdit, "form.input.value")
    assert line is not None and line.text() == "2.0"
    assert fx.owner.accept().values == {"value": 2.0}
