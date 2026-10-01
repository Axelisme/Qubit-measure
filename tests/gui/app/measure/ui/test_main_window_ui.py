"""UI structure tests for ExpTabWidget layout decisions."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QCoreApplication, QEvent, Qt
from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import QApplication, QLabel, QLineEdit, QStackedWidget, QWidget
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    MetaDictWriteback,
)
from zcu_tools.gui.app.measure.artifact_tracker import ArtifactKind, SaveStatus
from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.app.measure.services import TabSnapshot
from zcu_tools.gui.app.measure.state import State, TabInteractionState
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)
from zcu_tools.gui.cfg.resource import (
    AcceptedConfig,
    CfgEdit,
    CfgRef,
    CfgResolution,
    CfgResource,
)
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.plotting import FigureContainer
from zcu_tools.gui.plotting.explicit import QtPlotHost
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.events import SocChangedPayload
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.plotting.plots import Plots

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui.app.measure.ui._artifact_snapshots import ready_figures, with_artifacts


def _mock_ctrl() -> MagicMock:
    ctrl = MagicMock()
    configure_cfg_lookup(ctrl)
    ctrl.get_left_panel_width.return_value = 500
    return ctrl


def _apply_window_defaults(ctrl: MagicMock) -> MagicMock:
    """Set the minimal return values required by MainWindow.__init__ on a mock ctrl.

    MainWindow calls active_operation_count() and has_agent_connected() during
    bus-event handlers (FeedbackPanel docking, ADR-0066); tests
    that emit bus events must stub both to deterministic values.
    """
    ctrl.active_operation_count.return_value = 0
    ctrl.has_agent_connected.return_value = False
    ctrl.get_session_env.return_value = SessionEnv(
        md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None
    )
    return ctrl


# Distinguishes "caller did not specify analyze_params" (default to a MagicMock)
# from "caller explicitly wants None" (a non-analysis adapter's snapshot).
_DEFAULT_PARAMS = object()


class _RecordingTabActions:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str] | tuple[str, str, ArtifactKey]] = []

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

    def save_data(self, tab_id: str) -> None:
        self.calls.append(("save_data", tab_id))

    def save_image(self, tab_id: str, key: ArtifactKey) -> None:
        self.calls.append(("save_image", tab_id, key))

    def save_all(self, tab_id: str) -> None:
        self.calls.append(("save_all", tab_id))


def _snapshot(
    tab_id: str,
    *,
    global_run_active: bool = False,
    is_running: bool = False,
    is_analyzing: bool = False,
    is_saving_data: bool = False,
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
    writeback_items: tuple = (),
    figure: object = _DEFAULT_PARAMS,
    post_figure: object = _DEFAULT_PARAMS,
    data_status: SaveStatus | None = None,
) -> TabSnapshot:
    from zcu_tools.gui.app.measure.services.ports import (
        AnalysisPaneSnapshot,
        PathResourceSnapshot,
        PostAnalysisPaneSnapshot,
        RunPaneSnapshot,
        SavePaneSnapshot,
        TabPathsSnapshot,
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
        figure_obj = figure  # type: ignore[assignment]
    if post_figure is _DEFAULT_PARAMS:
        from matplotlib.figure import Figure as _Fig2

        post_figure_obj: Figure | None = _Fig2() if has_post_analyze_result else None
    else:
        post_figure_obj = post_figure  # type: ignore[assignment]

    # Capabilities include load_data
    caps = AdapterCapabilities(
        analysis=AnalysisMode.FIT if supports_analysis else AnalysisMode.NONE,
        post_analysis=supports_post_analysis,
        load_data=supports_load_data,
    )

    # Path resources per pane
    data_path_snap = PathResourceSnapshot(
        override=None, path="/tmp/data.hdf5" if has_run_result else None
    )
    analysis_image_snap = PathResourceSnapshot(
        override=None,
        path="/tmp/image.png" if has_figure and has_analyze_result else None,
    )
    post_image_snap = PathResourceSnapshot(
        override=None,
        path="/tmp/post.png" if has_post_analyze_result else None,
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
        image_paths={"fit": analysis_image_snap} if figure_obj is not None else {},
    )
    post_snap = PostAnalysisPaneSnapshot(
        params=post_analyze_params,
        result=object() if has_post_analyze_result else None,
        figures=ready_figures(post_figure_obj),
        writeback_items=(),
        image_paths={"fit": post_image_snap} if post_figure_obj is not None else {},
    )
    save_snap = SavePaneSnapshot(data_path=data_path_snap)
    paths_snap = TabPathsSnapshot(
        data=data_path_snap,
        analysis_images=analysis_snap.image_paths,
        post_analysis_images=post_snap.image_paths,
    )

    snapshot = TabSnapshot(
        adapter_name="fake",
        tab_id=tab_id,
        interaction=TabInteractionState(
            global_run_active=global_run_active,
            is_running=is_running,
            is_analyzing=is_analyzing,
            is_saving_data=is_saving_data,
            has_context=has_context,
            has_active_context=has_active_context,
            has_soc=has_soc,
            has_run_result=has_run_result,
            has_analyze_result=has_analyze_result,
            has_figure=has_figure,
            has_post_analyze_result=has_post_analyze_result,
        ),
        cfg_schema=MagicMock(),
        capabilities=caps,
        run=run_snap,
        analysis=analysis_snap,
        post_analysis=post_snap,
        save=save_snap,
        paths=paths_snap,
    )
    statuses = {ArtifactKind.DATA: data_status} if data_status is not None else None
    return with_artifacts(snapshot, statuses)


def test_left_panel_toggle_is_attached_to_tab_bar(qapp):
    from qtpy.QtWidgets import QApplication
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        _mock_ctrl(),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )
    tab.show()
    QApplication.processEvents()

    corner = tab._left_tabs.cornerWidget(Qt.TopLeftCorner)  # type: ignore[attr-defined]
    assert corner is None
    assert tab._left_edge_handle.isVisible() is True


def test_left_panel_toggle_uses_collapsed_boundary_handle(qapp):
    from qtpy.QtWidgets import QApplication
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        _mock_ctrl(),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )
    tab.resize(1000, 700)
    tab.show()
    QApplication.processEvents()

    expanded_left = tab._splitter.sizes()[0]
    expanded_handle_x = tab._left_edge_handle.x()
    assert expanded_handle_x > 0
    assert tab._left_panel_collapsed is False

    tab._left_edge_handle.click()
    QApplication.processEvents()

    assert tab._splitter.sizes()[0] == 0
    assert tab._left_panel_collapsed is True
    assert tab._left_edge_handle.x() == 0

    tab._left_edge_handle.click()
    QApplication.processEvents()

    assert tab._left_panel_collapsed is False
    assert tab._left_edge_handle.x() > 0
    assert tab._splitter.sizes()[0] >= expanded_left


def test_left_panel_handle_tracks_splitter_boundary(qapp):
    from qtpy.QtWidgets import QApplication
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        _mock_ctrl(),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )
    tab.resize(1000, 700)
    tab.show()
    QApplication.processEvents()

    initial_x = tab._left_edge_handle.x()
    splitter_x = tab._splitter.geometry().x()
    left_boundary = splitter_x + tab._left_tabs.geometry().right() + 1
    assert abs(initial_x - (left_boundary - tab._left_edge_handle.width() // 2)) <= 2

    tab._splitter.setSizes([260, 740])
    tab._schedule_handle_layout()
    QApplication.processEvents()

    moved_x = tab._left_edge_handle.x()
    splitter_x = tab._splitter.geometry().x()
    left_boundary = splitter_x + tab._left_tabs.geometry().right() + 1
    assert abs(moved_x - (left_boundary - tab._left_edge_handle.width() // 2)) <= 2


def test_exp_tab_disables_local_buttons_while_analyzing(qapp):
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        _mock_ctrl(),
        AdapterCapabilities(
            analysis=AnalysisMode.FIT, post_analysis=False, load_data=True
        ),
    )
    tab.update_writeback_items(
        [MetaDictWriteback("r_f", "freq", proposed_value=6100.0)]
    )
    tab.update_interaction_state(
        _snapshot(
            "tab-1",
            global_run_active=False,
            is_running=False,
            is_analyzing=True,
            is_saving_data=False,
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
    assert tab._save_center.is_load_visible()
    assert tab._save_center.is_load_enabled() is False
    assert tab.writeback_widget.isEnabled() is False
    assert (
        tab._save_center.is_save_enabled(ArtifactKey(ArtifactKind.ANALYSIS, "fit"))
        is False
    )  # disabled because is_analyzing


def test_exp_tab_keeps_analyze_enabled_while_other_tab_running(qapp):
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        _mock_ctrl(),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )
    tab.update_interaction_state(
        _snapshot(
            "tab-1",
            global_run_active=True,
            is_running=False,
            is_analyzing=False,
            is_saving_data=False,
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
    assert tab._save_center.is_save_enabled(ArtifactKey(ArtifactKind.DATA)) is True


def test_exp_tab_disables_save_buttons_while_saving_data(qapp):
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        _mock_ctrl(),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )
    tab.update_interaction_state(
        _snapshot(
            "tab-1",
            global_run_active=False,
            is_running=False,
            is_analyzing=False,
            is_saving_data=True,
            has_context=True,
            has_active_context=True,
            has_soc=True,
            has_run_result=True,
            has_analyze_result=True,
            has_figure=True,
        )
    )

    assert tab._save_center.is_save_enabled(ArtifactKey(ArtifactKind.DATA)) is False
    assert (
        tab._save_center.is_save_enabled(ArtifactKey(ArtifactKind.ANALYSIS, "fit"))
        is False
    )
    assert tab.run_btn.text() == "Run"
    assert tab.run_btn.toolTip() == "Tab is busy"


def test_exp_tab_run_tooltip_shows_no_soc_reason(qapp):
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        _mock_ctrl(),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )
    tab.update_interaction_state(
        _snapshot(
            "tab-1",
            global_run_active=False,
            is_running=False,
            is_analyzing=False,
            is_saving_data=False,
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


def test_exp_tab_run_tooltip_shows_cfg_invalid_reason(qapp):
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        _mock_ctrl(),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )
    tab.cfg_form.first_invalid_reason = MagicMock(
        return_value="modules.readout: invalid"
    )
    tab.cfg_form.is_valid = MagicMock(return_value=False)
    tab.update_interaction_state(
        _snapshot(
            "tab-1",
            global_run_active=False,
            is_running=False,
            is_analyzing=False,
            is_saving_data=False,
            has_context=True,
            has_active_context=True,
            has_soc=True,
            has_run_result=False,
            has_analyze_result=False,
            has_figure=False,
        )
    )

    assert tab.run_btn.isEnabled() is False
    assert tab.run_btn.toolTip() == "Config invalid: modules.readout: invalid"


def test_exp_tab_draft_context_allows_analysis_but_disables_run_and_save(qapp):
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    snap = _snapshot(
        "tab-1",
        supports_load_data=True,
        global_run_active=False,
        is_running=False,
        is_analyzing=False,
        is_saving_data=False,
        has_context=True,
        has_active_context=False,
        has_soc=True,
        has_run_result=True,
        has_analyze_result=True,
        has_figure=True,
    )
    assert snap.capabilities is not None
    tab = ExpTabWidget("tab-1", _mock_ctrl(), snap.capabilities)
    tab.update_writeback_items(
        [MetaDictWriteback("r_f", "freq", proposed_value=6100.0)]
    )
    tab.update_interaction_state(snap)

    assert tab.run_btn.isEnabled() is False
    assert tab.run_btn.toolTip() == "Select or create a file-backed context"
    assert tab._save_center.is_load_visible()
    assert tab._save_center.is_load_enabled() is True
    assert tab.analyze_btn.isEnabled() is True
    assert tab.writeback_widget.isEnabled() is True
    assert tab._save_center.is_save_enabled(ArtifactKey(ArtifactKind.DATA)) is False
    assert (
        tab._save_center.is_save_enabled(ArtifactKey(ArtifactKind.ANALYSIS, "fit"))
        is False
    )


def test_non_analysis_adapter_hides_analysis_widgets_but_keeps_save(qapp):
    """flux_dep / power_dep adapters (analysis=NONE) hide only the
    analysis widgets, never the Save section. Regression: the whole second tab
    used to be hidden, so the user could not save a 2D-sweep run at all."""
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        _mock_ctrl(),
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
    visible = [tab._left_tabs.tabText(i) for i in range(tab._left_tabs.count())]
    assert visible == ["Run", "Data", "Guide"]
    # Prove Analysis page and controls were never constructed, not only hidden
    assert not hasattr(tab, "_analysis_panel")
    assert not hasattr(tab, "analyze_form")
    assert not hasattr(tab, "_analyze_section")
    # No Load Data capability means no control is constructed.
    assert not tab._save_center.is_load_visible()
    # ... but Save stays reachable and usable (run result + active context).
    assert tab._save_center.has_artifact(ArtifactKey(ArtifactKind.DATA))
    assert tab._save_center.is_save_enabled(ArtifactKey(ArtifactKind.DATA)) is True


def test_analysis_adapter_shows_analysis_widgets_and_labels_tab(qapp):
    """An analysis adapter keeps the analysis widgets visible and the second tab
    labelled 'Analysis' — the counterpart to the non-analysis case."""
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        _mock_ctrl(),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )
    tab.update_interaction_state(
        _snapshot("tab-1", has_run_result=True, supports_analysis=True)
    )

    assert tab._left_tabs.tabText(1) == "Analysis"
    assert tab._analyze_section.isHidden() is False
    assert tab.analyze_btn.isHidden() is False
    assert tab._save_center.has_artifact(ArtifactKey(ArtifactKind.DATA))


def test_exp_tab_load_button_requires_context_but_not_soc(qapp):
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
    tab = ExpTabWidget("tab-1", _mock_ctrl(), snap1.capabilities)
    tab.update_interaction_state(snap1)
    assert tab._save_center.is_load_visible()
    assert tab._save_center.is_load_enabled() is True

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
    assert tab._save_center.is_load_enabled() is False


@pytest.mark.parametrize("analysis_error", [None, "analysis defaults unavailable"])
def test_main_window_load_data_dialog_calls_controller(
    qapp, monkeypatch, tmp_path, analysis_error
):
    from qtpy.QtWidgets import QFileDialog
    from zcu_tools.gui.app.measure.services.load import LoadTabResultOutcome
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    ctrl.get_bus.return_value = EventBus()
    ctrl.has_tab.return_value = True
    database_root = tmp_path / "Database" / "Q3_2D" / "Q1"
    database_root.mkdir(parents=True)
    ctrl.get_session_env.return_value = SessionEnv(
        md=MagicMock(),
        ml=MagicMock(),
        soc=None,
        soccfg=None,
        database_path=str(database_root / "2026" / "06" / "Data_0625"),
    )
    ctrl.load_tab_result.return_value = LoadTabResultOutcome(
        tab_id="tab-1",
        data_path="/tmp/result.hdf5",
        result_type="Result",
        has_cfg_snapshot=False,
        has_analyze_params=analysis_error is None,
        analysis_error=analysis_error,
    )
    dialogs = RecordingDialogPresenter()
    window = MainWindow(ctrl, dialog_presenter=dialogs)
    window._tab_widgets["tab-1"] = MagicMock()
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


def test_main_window_successful_writeback_relies_on_event_owned_projection(qapp):
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    ctrl.get_bus.return_value = EventBus()
    ctrl.has_tab.return_value = True
    ctrl.apply_writeback_for_pane.return_value = {
        "applied_ids": ["md-1"],
        "written": {"md": ["r_f"], "ml_modules": [], "ml_waveforms": []},
    }
    window = MainWindow(ctrl)
    window._tab_widgets["tab-1"] = MagicMock()
    window.refresh_tab_writeback = MagicMock()
    window.show_status_message = MagicMock()

    window.apply_tab_writeback("tab-1", pane="analysis")

    ctrl.apply_writeback_for_pane.assert_called_once_with("tab-1", "analysis")
    window.refresh_tab_writeback.assert_not_called()
    window.show_status_message.assert_called_once_with("Writeback applied: md-1")


def test_main_window_toolbar_does_not_show_arb_waveforms(qapp):
    from qtpy.QtWidgets import QPushButton
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    ctrl.get_bus.return_value = EventBus()

    window = MainWindow(ctrl)
    texts = {button.text() for button in window.findChildren(QPushButton)}

    assert "Inspect…" in texts
    assert "Agent…" not in texts
    assert "Arb Waveforms…" not in texts


def test_main_window_named_dialog_facade_delegates_to_registry(qapp):
    from zcu_tools.gui.app.measure.remote.dialogs import DialogName
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    ctrl.get_bus.return_value = EventBus()
    window = MainWindow(ctrl)
    registry = MagicMock()
    registry.visible_names.return_value = [DialogName.PREDICTOR]
    registry.take_screenshot.return_value = b"png"
    window._dialog_registry = registry

    window.close_dialog(DialogName.PREDICTOR)
    window.open_dialog(DialogName.PREDICTOR)
    assert window.list_open_dialogs() == [DialogName.PREDICTOR]

    assert window.take_dialog_screenshot(DialogName.SETUP) == b"png"

    registry.close.assert_called_once_with(DialogName.PREDICTOR)
    registry.open.assert_called_once_with(DialogName.PREDICTOR)
    registry.visible_names.assert_called_once_with()
    registry.take_screenshot.assert_called_once_with(DialogName.SETUP)


def test_show_error_dialog_retains_until_close(qapp):
    from qtpy.QtWidgets import QMessageBox
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    ctrl.get_bus.return_value = EventBus()
    window = MainWindow(ctrl)

    window.show_error_dialog("Broken", "Something failed")
    qapp.processEvents()

    dialogs = [
        dialog
        for dialog in window.findChildren(QMessageBox)
        if dialog.windowTitle() == "Broken"
    ]
    assert len(dialogs) == 1
    assert len(window._dialog_refs) == 1
    assert window.list_open_dialogs() == []

    dialogs[0].reject()
    qapp.processEvents()

    assert len(window._dialog_refs) == 0


def test_open_notify_prompt_retains_until_close(qapp):
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow
    from zcu_tools.gui.app.measure.ui.notify_dialog import NotifyUserDialog

    ctrl = _apply_window_defaults(MagicMock())
    ctrl.get_bus.return_value = EventBus()
    window = MainWindow(ctrl)

    window.open_notify_prompt(17, "Need input", timeout=60.0)
    qapp.processEvents()

    dialogs = window.findChildren(NotifyUserDialog)
    assert len(dialogs) == 1
    assert dialogs[0]._token == 17
    assert len(window._dialog_refs) == 1
    assert window.list_open_dialogs() == []

    dialogs[0].reject()
    qapp.processEvents()

    assert len(window._dialog_refs) == 0


def test_main_window_load_data_dialog_cancel_is_noop(qapp, monkeypatch):
    from qtpy.QtWidgets import QFileDialog
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    ctrl.get_bus.return_value = EventBus()
    ctrl.has_tab.return_value = True
    window = MainWindow(ctrl)
    window._tab_widgets["tab-1"] = MagicMock()
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *args, **kwargs: ("", "")
    )

    window.load_tab_data_dialog("tab-1")

    ctrl.load_tab_result.assert_not_called()


def test_main_window_load_data_dialog_shows_user_facing_error(qapp, monkeypatch):
    from qtpy.QtWidgets import QFileDialog
    from zcu_tools.gui.app.measure.services.load import LoadDataError
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    ctrl.get_bus.return_value = EventBus()
    ctrl.has_tab.return_value = True
    ctrl.load_tab_result.side_effect = LoadDataError(
        "Cannot load this data file into the current tab.\n\nDetails: bad axes",
        reason_code="invalid_data_file",
    )
    window = MainWindow(ctrl)
    window._tab_widgets["tab-1"] = MagicMock()
    window.show_error_dialog = MagicMock()
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: ("/tmp/bad.hdf5", ""),
    )

    window.load_tab_data_dialog("tab-1")

    window.show_error_dialog.assert_called_once()
    title, message = window.show_error_dialog.call_args.args
    assert title == "Load data failed"
    assert "Cannot load this data file" in message
    assert "bad axes" in message


def test_main_window_run_lock_keeps_new_tab_available(qapp):
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = MagicMock()
    configure_cfg_lookup(ctrl)
    ctrl.get_bus.return_value = EventBus()
    ctrl.has_tab.return_value = True
    ctrl.get_tab_snapshot.side_effect = lambda tab_id: _snapshot(
        tab_id,
        global_run_active=tab_id != "tab-1",
        is_running=tab_id == "tab-1",
    )

    window = MainWindow(ctrl)
    tab_one = MagicMock()
    tab_two = MagicMock()
    window._tab_widgets["tab-1"] = tab_one
    window._tab_widgets["tab-2"] = tab_two

    window.refresh_run_lock("tab-1")

    assert window._toolbar.new_tab_button.isEnabled() is True
    tab_one.update_interaction_state.assert_called_once()
    tab_two.update_interaction_state.assert_called_once()


def test_main_window_tabs_are_movable_and_close_uses_moved_widget(qapp):
    from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(_editor_wiring_ctrl())
    ctrl.get_bus.return_value = EventBus()
    ctrl.has_tab.side_effect = lambda tab_id: tab_id in {"tab-a", "tab-b"}
    window = MainWindow(ctrl)
    tab_a = ExpTabWidget(
        "tab-a",
        ctrl,
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )
    tab_b = ExpTabWidget(
        "tab-b",
        ctrl,
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )
    tab_a.attach(_snapshot("tab-a", has_run_result=False), _RecordingTabActions())
    tab_b.attach(_snapshot("tab-b", has_run_result=False), _RecordingTabActions())
    window._tab_widgets["tab-a"] = tab_a
    window._tab_widgets["tab-b"] = tab_b
    window._tabs.addTab(tab_a, "A")
    window._tabs.addTab(tab_b, "B")

    assert window._tabs.isMovable() is True

    tab_bar = window._tabs.tabBar()
    assert tab_bar is not None
    tab_bar.moveTab(0, 1)
    assert window._tabs.widget(1) is tab_a
    ctrl.reorder_tabs.assert_called_once_with(["tab-b", "tab-a"])

    window._on_tab_close_requested(1)

    ctrl.close_tab.assert_called_once_with("tab-a")


def test_main_window_soc_changed_refreshes_run_lock(qapp):
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_running_tab_id.return_value = None
    ctrl.has_tab.return_value = True
    ctrl.get_tab_snapshot.return_value = _snapshot("tab-1", has_soc=False)

    window = MainWindow(ctrl)
    tab = MagicMock()
    window._tab_widgets["tab-1"] = tab

    bus.emit(SocChangedPayload(soc=None, soccfg=None))

    tab.update_interaction_state.assert_called()


def test_main_window_content_event_queries_single_tab_snapshot(qapp):
    from zcu_tools.gui.app.measure.events.tab import (
        TabContentChangedPayload,
        TabContentFact,
    )
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_tab_snapshot.return_value = _snapshot("tab-1")
    window = MainWindow(ctrl)
    window._tab_widgets["tab-1"] = MagicMock()

    bus.emit(TabContentChangedPayload("tab-1", TabContentFact.RUN_RESULT_COMMITTED))

    ctrl.get_tab_snapshot.assert_called_once_with("tab-1")


def test_main_window_interaction_event_refreshes_finished_analysis_figure(qapp):
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.measure.events.tab import (
        TabInteractionChangedPayload,
        TabInteractionFact,
    )
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
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
    window = MainWindow(ctrl)
    tab = MagicMock()
    window._tab_widgets["tab-1"] = tab

    bus.emit(
        TabInteractionChangedPayload("tab-1", TabInteractionFact.PRIMARY_ANALYZE_FAILED)
    )

    ctrl.get_tab_snapshot.assert_called_once_with("tab-1")
    tab.update_writeback_items.assert_not_called()
    tab.show_analysis_figures.assert_called_once_with(
        ctrl.get_tab_snapshot.return_value.analysis.figures
    )


def test_main_window_interaction_event_does_not_restore_old_figure_on_analyze_start(
    qapp,
):
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.measure.events.tab import (
        TabInteractionChangedPayload,
        TabInteractionFact,
    )
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
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
    window = MainWindow(ctrl)
    tab = MagicMock()
    window._tab_widgets["tab-1"] = tab

    bus.emit(
        TabInteractionChangedPayload(
            "tab-1", TabInteractionFact.PRIMARY_ANALYZE_STARTED
        )
    )

    ctrl.get_tab_snapshot.assert_called_once_with("tab-1")
    tab.show_analysis_figures.assert_not_called()


def test_main_window_interaction_event_does_not_restore_old_figure_on_run_start(
    qapp,
):
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.measure.events.tab import (
        TabInteractionChangedPayload,
        TabInteractionFact,
    )
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.has_tab.return_value = True
    figure = Figure()
    ctrl.get_tab_snapshot.return_value = _snapshot(
        "tab-1",
        is_running=True,
        is_analyzing=False,
        has_analyze_result=True,
        has_figure=True,
        figure=figure,
    )
    window = MainWindow(ctrl)
    tab = MagicMock()
    window._tab_widgets["tab-1"] = tab

    bus.emit(
        TabInteractionChangedPayload("tab-1", TabInteractionFact.RUN_START_REJECTED)
    )

    ctrl.get_tab_snapshot.assert_called_once_with("tab-1")
    tab.show_analysis_figures.assert_not_called()


def test_main_window_interaction_event_shows_post_figure_after_primary(qapp):
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.measure.events.tab import (
        TabInteractionChangedPayload,
        TabInteractionFact,
    )
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
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
    window = MainWindow(ctrl)
    tab = MagicMock()
    window._tab_widgets["tab-1"] = tab

    bus.emit(
        TabInteractionChangedPayload("tab-1", TabInteractionFact.PRIMARY_ANALYZE_FAILED)
    )

    snapshot = ctrl.get_tab_snapshot.return_value
    tab.show_analysis_figures.assert_called_once_with(snapshot.analysis.figures)
    tab.show_post_analysis_figures.assert_called_once_with(
        snapshot.post_analysis.figures
    )


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
def test_analysis_terminal_restore_rebuilds_real_primary_then_post_canvas(qapp, fact):
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.measure.events.tab import (
        TabInteractionChangedPayload,
        TabInteractionFact,
    )
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget, MainWindow

    ctrl = _apply_window_defaults(MagicMock())
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
    window = MainWindow(ctrl)
    tab = ExpTabWidget(
        "tab-1",
        ctrl,
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True),
    )
    window._tab_widgets["tab-1"] = tab
    tab.show_analysis_figures(snapshot.analysis.figures)
    tab.show_post_analysis_figures(snapshot.post_analysis.figures)

    # Simulate operation start clearing only its pane; here clear both figures to emulate stale state
    tab._analysis_container.clear_dynamic_canvases()
    tab._post_container.clear_dynamic_canvases()
    assert tab.get_current_figure_for_pane("analysis") is None
    assert tab.get_current_figure_for_pane("post_analysis") is None

    bus.emit(TabInteractionChangedPayload("tab-1", TabInteractionFact(fact)))

    # Coordinator failure/cancel re-renders retained figures per pane
    assert tab.get_current_figure_for_pane("analysis") is primary
    assert tab.get_current_figure_for_pane("post_analysis") is post


def test_loaded_content_clears_stale_real_canvas_when_state_has_no_figure(qapp):
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

    ctrl = _apply_window_defaults(MagicMock())
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
    window = MainWindow(ctrl)
    tab = ExpTabWidget(
        "tab-1",
        ctrl,
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True),
    )
    window._tab_widgets["tab-1"] = tab
    tab.show_analysis_figures(ready_figures(Figure()))
    assert tab.get_current_figure_for_pane("analysis") is not None

    bus.emit(TabContentChangedPayload("tab-1", TabContentFact.LOADED_RESULT_COMMITTED))

    assert tab.get_current_figure_for_pane("analysis") is None
    assert tab.get_current_figure_for_pane("post_analysis") is None


def _emit_run_finished(bus, tab_id: str, outcome: str) -> None:
    from zcu_tools.gui.app.measure.events.run import RunFinishedPayload

    bus.emit(RunFinishedPayload(tab_id=tab_id, outcome=outcome))


def test_finished_run_preserves_selected_subtab(qapp):
    """RUN_FINISHED refreshes state without choosing a result pane."""
    from qtpy.QtWidgets import QTabWidget, QWidget
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_running_tab_id.return_value = None
    ctrl.get_tab_snapshot.return_value = _snapshot("tab-1", has_run_result=True)
    window = MainWindow(ctrl)
    tab = MagicMock()
    tab._left_tabs = QTabWidget()
    for label in ("Run", "Analysis", "Data"):
        tab._left_tabs.addTab(QWidget(), label)
    tab._left_tabs.setCurrentIndex(2)
    window._tab_widgets["tab-1"] = tab

    _emit_run_finished(bus, "tab-1", outcome="finished")

    assert tab._left_tabs.currentIndex() == 2
    tab.focus_result_panel.assert_not_called()


def test_stopped_run_does_not_auto_switch_to_analysis_tab(qapp):
    """A stopped (cancelled) run may leave a partial result, but the user
    interrupted on purpose — must not yank them to the Analysis tab."""
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_running_tab_id.return_value = None
    ctrl.get_tab_snapshot.return_value = _snapshot("tab-1", has_run_result=True)
    window = MainWindow(ctrl)
    tab = MagicMock()
    window._tab_widgets["tab-1"] = tab

    _emit_run_finished(bus, "tab-1", outcome="cancelled")

    tab._left_tabs.setCurrentIndex.assert_not_called()


def test_non_analysis_adapter_run_preserves_selected_subtab(qapp):
    """Non-analysis adapters also retain the user's selected pane after Run."""
    from qtpy.QtWidgets import QTabWidget, QWidget
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_running_tab_id.return_value = None
    ctrl.get_tab_snapshot.return_value = _snapshot(
        "tab-1", has_run_result=True, supports_analysis=False, analyze_params=None
    )
    window = MainWindow(ctrl)
    tab = MagicMock()
    tab._left_tabs = QTabWidget()
    for label in ("Run", "Data", "Guide"):
        tab._left_tabs.addTab(QWidget(), label)
    tab._left_tabs.setCurrentIndex(1)
    window._tab_widgets["tab-1"] = tab

    _emit_run_finished(bus, "tab-1", outcome="finished")

    assert tab._left_tabs.currentIndex() == 1
    tab.focus_result_panel.assert_not_called()


def test_refresh_analyze_form_skips_non_analysis_adapter_without_raising(qapp):
    """A finished run on a non-analysis adapter has no analyze params; the
    content refresh must skip the analyze form rather than hit the Fast-Fail
    guard that demands initialized params (regression: it used to raise)."""
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_running_tab_id.return_value = None
    ctrl.get_tab_snapshot.return_value = _snapshot(
        "tab-1", has_run_result=True, supports_analysis=False, analyze_params=None
    )
    window = MainWindow(ctrl)
    tab = MagicMock()
    window._tab_widgets["tab-1"] = tab

    # Must not raise "Run result has no initialized analyze parameters".
    window.refresh_tab_analyze_form("tab-1")

    tab.populate_analyze_params.assert_not_called()


def _editor_wiring_ctrl() -> MagicMock:
    """A view facade over a real, caller-owned cfg resource."""
    from tests.gui.app.measure._cfg_fakes import make_cfg

    ctrl = _mock_ctrl()
    cfg = make_cfg(_pulse_schema())
    ctrl.cfg_resources.lookup.side_effect = lambda _tab: cfg
    ctrl.active_operation_count.return_value = 0
    ctrl.has_agent_connected.return_value = False
    return ctrl


def _pulse_schema():
    from zcu_tools.gui.app.measure.specs import make_pulse_spec
    from zcu_tools.gui.cfg import (
        CfgSchema,
        make_default_value,
    )

    spec = make_pulse_spec()
    return CfgSchema(spec=spec, value=make_default_value(spec))


def test_exp_tab_attach_projects_existing_resource(qapp):
    from qtpy.QtWidgets import QLineEdit, QWidget
    from zcu_tools.gui.cfg.resource import CfgEdit

    ctrl = _editor_wiring_ctrl()
    cfg = ctrl.cfg_resources.lookup("tab-1")
    before = cfg.observe().ref
    tab = ExpTabWidget("tab-1", ctrl, AdapterCapabilities(analysis=AnalysisMode.FIT))
    tab.attach(_snapshot("tab-1"), _RecordingTabActions())
    assert cfg.observe().ref == before
    cfg.edit(before.revision, (CfgEdit(("gain",), 0.45),))
    widget = tab.cfg_form.findChild(QWidget, "cfgInput:gain")
    assert widget is not None
    entry = widget.findChild(QLineEdit)
    assert entry is not None and float(entry.text()) == 0.45


def test_exp_tab_detach_preserves_resource_for_later_views(qapp):
    from zcu_tools.gui.cfg.resource import CfgEdit

    ctrl = _editor_wiring_ctrl()
    cfg = ctrl.cfg_resources.lookup("tab-1")
    tab = ExpTabWidget("tab-1", ctrl, AdapterCapabilities(analysis=AnalysisMode.FIT))
    tab.attach(_snapshot("tab-1"), _RecordingTabActions())
    tab.detach()
    after = cfg.edit(cfg.observe().ref.revision, (CfgEdit(("gain",), 0.45),))
    other = ExpTabWidget("tab-1", ctrl, AdapterCapabilities(analysis=AnalysisMode.FIT))
    other.attach(_snapshot("tab-1"), _RecordingTabActions())
    assert cfg.observe().ref == after.ref
    assert other.cfg_form.is_valid() == (after.status.value == "Valid")


def test_ml_change_refreshes_resource_form_and_run_gate_without_main_loop(qapp):
    from dataclasses import replace
    from typing import Any, cast

    from qtpy.QtWidgets import QComboBox, QWidget
    from zcu_tools.experiment.v2_gui.measure.adapters.fake import FakeAdapter
    from zcu_tools.gui.app.measure.cfg_schemas import module_cfg_to_value
    from zcu_tools.gui.app.measure.specs import make_pulse_spec
    from zcu_tools.gui.cfg import (
        CfgSchema,
        CfgSectionSpec,
        CfgSectionValue,
        ReferenceSpec,
        ReferenceValue,
    )
    from zcu_tools.gui.session.events import MlChangedPayload
    from zcu_tools.resources.context import MetaDict, ModuleLibrary

    from tests.gui.app.measure.remote._helpers import Fixture

    fx = Fixture(headless=True)
    ml = ModuleLibrary(None)
    fx.state.set_context(replace(fx.state.session_env, md=MetaDict(None), ml=ml))
    raw = {
        "type": "pulse",
        "waveform": {"style": "const", "length": 1.0},
        "ch": 0,
        "nqz": 1,
        "freq": 5000.0,
        "gain": 0.2,
        "phase": 0.0,
        "pre_delay": 0.0,
        "post_delay": 0.0,
        "mixer_freq": None,
    }
    _, value = module_cfg_to_value(raw)
    spec = make_pulse_spec()
    schema = CfgSchema(
        CfgSectionSpec(
            fields={"drive": ReferenceSpec("module", [spec], label="Drive")}
        ),
        CfgSectionValue({"drive": ReferenceValue("drive-pulse", value)}),
    )
    cfg = fx.prepare_tab("tab-1", FakeAdapter(), schema)
    tab = ExpTabWidget("tab-1", fx.ctrl, AdapterCapabilities(analysis=AnalysisMode.FIT))
    snapshot = _snapshot("tab-1", has_run_result=False)

    class GateActions(_RecordingTabActions):
        def refresh_interaction(self, tab_id):
            super().refresh_interaction(tab_id)
            tab.update_interaction_state(snapshot)

    tab.attach(snapshot, GateActions())
    assert not tab.cfg_form.is_valid()
    assert not tab.run_btn.isEnabled()
    ml.modules["drive-pulse"] = cast(Any, raw)
    fx.bus.emit(MlChangedPayload(ml))
    widget = tab.cfg_form.findChild(QWidget, "cfgInput:drive")
    assert widget is not None
    combo = widget.findChild(QComboBox)
    assert combo is not None and combo.currentText() == "Lib: drive-pulse"
    assert tab.cfg_form.is_valid()
    assert tab.run_btn.isEnabled()
    assert cfg.observe().status.value == "Valid"


def test_exp_tab_buttons_dispatch_public_tab_actions(qapp):
    import dataclasses

    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    ctrl = _editor_wiring_ctrl()
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
        cfg_schema=_pulse_schema(),
    )
    tab.attach(snapshot, actions)

    assert tab.run_btn.isEnabled() is True
    assert tab._save_center.is_load_visible()
    assert tab._save_center.is_load_enabled() is True
    assert tab.analyze_btn.isEnabled() is True
    assert tab.post_analyze_btn.isEnabled() is True
    assert tab._save_center.is_save_enabled(ArtifactKey(ArtifactKind.DATA)) is True
    assert (
        tab._save_center.is_save_enabled(ArtifactKey(ArtifactKind.ANALYSIS, "fit"))
        is True
    )
    assert (
        tab._save_center.is_save_enabled(ArtifactKey(ArtifactKind.POST_ANALYSIS, "fit"))
        is True
    )

    actions.calls.clear()
    tab.run_btn.click()
    tab._save_center.load_button.click()
    tab.analyze_btn.click()
    tab.post_analyze_btn.click()
    tab.writeback_widget.apply_requested.emit()
    tab.post_writeback_widget.apply_requested.emit()
    tab._save_center.save_button(ArtifactKey(ArtifactKind.DATA)).click()
    tab._save_center.save_button(ArtifactKey(ArtifactKind.ANALYSIS, "fit")).click()
    tab._save_center.save_button(ArtifactKey(ArtifactKind.POST_ANALYSIS, "fit")).click()

    assert actions.calls == [
        ("run_or_stop", "tab-1"),
        ("load_data", "tab-1"),
        ("analyze", "tab-1"),
        ("post_analyze", "tab-1"),
        ("apply_writeback", "tab-1"),
        ("apply_post_writeback", "tab-1"),
        ("save_data", "tab-1"),
        ("save_image", "tab-1", ArtifactKey(ArtifactKind.ANALYSIS, "fit")),
        ("save_image", "tab-1", ArtifactKey(ArtifactKind.POST_ANALYSIS, "fit")),
    ]


def test_exp_tab_reset_publishes_defaults_on_same_resource(qapp):
    from zcu_tools.gui.cfg.resource import CfgEdit

    dialogs = RecordingDialogPresenter(confirm_answers=[True])
    ctrl = _editor_wiring_ctrl()
    cfg = ctrl.cfg_resources.lookup("tab-1")
    defaults = cfg.snapshot_inputs()
    cfg.edit(cfg.observe().ref.revision, (CfgEdit(("gain",), 0.45),))
    before = cfg.observe().ref
    tab = ExpTabWidget(
        "tab-1",
        ctrl,
        AdapterCapabilities(analysis=AnalysisMode.FIT),
        dialog_presenter=dialogs,
    )
    actions = _RecordingTabActions()
    tab.attach(_snapshot("tab-1"), actions)
    tab.reset_btn.click()
    after = cfg.observe().ref
    assert after.cfg_id == before.cfg_id
    assert after.revision == before.revision + 1
    assert cfg.snapshot_inputs() == defaults
    assert actions.calls[-1] == ("refresh_interaction", "tab-1")


def test_exp_tab_reset_confirm_no_does_not_reset(qapp):
    dialogs = RecordingDialogPresenter(confirm_answers=[False])
    ctrl = _editor_wiring_ctrl()
    cfg = ctrl.cfg_resources.lookup("tab-1")
    before = cfg.observe()
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
    assert cfg.observe() == before
    assert actions.calls == []


def test_exp_tab_reset_btn_idle_only_enable(qapp):
    """reset_btn must be enabled when idle and disabled while the tab is busy."""
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        _mock_ctrl(),
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
    ctrl = MagicMock()
    configure_cfg_lookup(ctrl)
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = 2
    window = MainWindow(ctrl, dialog_presenter=dialogs)
    event = QCloseEvent()

    window.closeEvent(event)

    assert dialogs.calls[-1].title == "Operations in progress"
    assert event.isAccepted() is False  # async wait — not closed yet
    QCoreApplication.processEvents()  # drain the deferred singleShot(0)
    _shutdown_callback(ctrl)


def test_main_window_declining_confirmation_keeps_window_open(qapp):
    from qtpy.QtGui import QCloseEvent
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    dialogs = RecordingDialogPresenter(confirm_answers=[False])
    ctrl = MagicMock()
    configure_cfg_lookup(ctrl)
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = 1
    window = MainWindow(ctrl, dialog_presenter=dialogs)
    event = QCloseEvent()

    window.closeEvent(event)

    assert dialogs.calls[-1].title == "Operations in progress"
    assert event.isAccepted() is False
    ctrl.begin_shutdown.assert_not_called()


def test_main_window_persists_session_on_close_when_idle(qapp):
    """Idle close: no confirmation; closeEvent ignores the event and *defers*
    begin_shutdown to the next event-loop turn (so it never re-enters
    self.close() within the closeEvent stack — the single-click bug). After the
    deferred turn fires, begin_shutdown runs _perform_close → persist."""
    from qtpy.QtCore import QCoreApplication
    from qtpy.QtGui import QCloseEvent
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = MagicMock()
    configure_cfg_lookup(ctrl)
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = 0
    # The real coordinator runs on_closed once nothing is pending; here drive it
    # synchronously so the deferred turn exercises _perform_close.
    ctrl.begin_shutdown.side_effect = lambda on_closed: on_closed()
    window = MainWindow(ctrl)
    event = QCloseEvent()

    window.closeEvent(event)

    # Deferred: closeEvent must NOT have begun shutdown synchronously.
    assert event.isAccepted() is False
    ctrl.begin_shutdown.assert_not_called()

    # Drain the singleShot(0) — the deferred turn runs begin_shutdown.
    QCoreApplication.processEvents()

    _shutdown_callback(ctrl)
    ctrl.persist_all.assert_called_once_with()


def test_main_window_close_removes_event_bus_subscriptions(qapp):
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
        DeviceChangedPayload,
        DeviceSetupFinishedPayload,
        DeviceSetupStartedPayload,
        PredictorChangedPayload,
        SocChangedPayload,
    )

    ctrl = MagicMock()
    configure_cfg_lookup(ctrl)
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.active_operation_count.return_value = 0
    ctrl.begin_shutdown.side_effect = lambda on_closed: on_closed()
    window = MainWindow(ctrl)

    payload_types = (
        TabInteractionChangedPayload,
        RunStartedPayload,
        RunFinishedPayload,
        ContextSwitchedPayload,
        TabAddedPayload,
        TabClosedPayload,
        TabContentChangedPayload,
        PredictorChangedPayload,
        SocChangedPayload,
        DeviceSetupStartedPayload,
        DeviceSetupFinishedPayload,
        DeviceChangedPayload,
    )
    for payload_type in payload_types:
        assert bus._subs.get(payload_type)

    window.closeEvent(QCloseEvent())
    QCoreApplication.processEvents()

    for payload_type in payload_types:
        assert bus._subs.get(payload_type, []) == []


def test_new_tab_menu_supports_nested_paths(qapp, monkeypatch):
    from qtpy.QtWidgets import QMenu
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    del qapp
    ctrl = MagicMock()
    configure_cfg_lookup(ctrl)
    ctrl.get_bus.return_value = EventBus()
    ctrl.get_adapter_names.return_value = [
        "fake",
        "twotone/rabi/length",
        "twotone/rabi/amp",
    ]
    window = MainWindow(ctrl)

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

    window._toolbar.show_new_tab_menu()

    ctrl.new_tab.assert_called_once_with("twotone/rabi/length")


def test_show_analysis_figure_draws_canvas(qapp, monkeypatch):
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    del qapp
    tab = ExpTabWidget(
        "tab-1",
        _mock_ctrl(),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )
    canvas = MagicMock()

    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.ui.exp_tab_widget.attach_existing_figure_to_container",
        lambda fig, container: canvas,
    )

    tab.show_analysis_figures(ready_figures(Figure()))

    canvas.draw.assert_called_once_with()


def test_show_analysis_figure_keeps_two_figures_coexisting(qapp):
    """Analysis and Post figures live in independent containers; each pane's history
    is distinct and showing one does not evict the other."""
    from matplotlib.figure import Figure
    from qtpy.QtWidgets import QApplication
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    del qapp
    tab = ExpTabWidget(
        "tab-1",
        _mock_ctrl(),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True),
    )
    tab.show()
    QApplication.processEvents()

    fig_a1 = Figure()
    fig_p1 = Figure()
    tab.show_analysis_figures(ready_figures(fig_a1))
    tab.show_post_analysis_figures(ready_figures(fig_p1))

    # Each pane has its own stack with placeholder + 1 canvas
    assert tab._analysis_stack.count() == 2
    assert tab._post_stack.count() == 2
    assert tab.get_current_figure_for_pane("analysis") is fig_a1
    assert tab.get_current_figure_for_pane("post_analysis") is fig_p1

    # Re-showing analysis does not affect post
    fig_a2 = Figure()
    tab.show_analysis_figures(ready_figures(fig_a2))
    assert tab.get_current_figure_for_pane("analysis") is fig_a2
    assert tab.get_current_figure_for_pane("post_analysis") is fig_p1
    assert tab._post_stack.count() == 2


# ---------------------------------------------------------------------------
# FeedbackPanel docking gate (ADR-0066): op-count AND agent-connected
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unsaved measurement data close guard tests (TKT-001)
# ---------------------------------------------------------------------------


def _setup_window_with_tabs(
    dialog_presenter: RecordingDialogPresenter | None = None,
    active_operations: int = 0,
) -> tuple[MainWindow, MagicMock]:
    ctrl = _apply_window_defaults(_editor_wiring_ctrl())
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = active_operations
    ctrl.has_tab.side_effect = lambda tid: True
    dialogs = dialog_presenter or RecordingDialogPresenter()
    window = MainWindow(ctrl, dialog_presenter=dialogs)
    return window, ctrl


def _add_tab(
    window: MainWindow,
    tab_id: str,
    *,
    has_run: bool = False,
    saved: bool = False,
) -> ExpTabWidget:
    tab = ExpTabWidget(
        tab_id,
        window._ctrl,
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )
    snap = _snapshot(
        tab_id,
        has_run_result=has_run,
        data_status=SaveStatus.SAVED if saved else None,
    )
    tab.attach(snap, _RecordingTabActions())
    window._tab_widgets[tab_id] = tab
    window._tabs.addTab(tab, tab_id)
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
    ) -> None:
        super().destructive_confirm(
            parent,
            title,
            message,
            action_text=action_text,
            on_decision=lambda _confirmed: None,
            default=default,
        )
        self.destructive_decisions.append(on_decision)


def test_tab_close_without_result_closes_immediately(qapp):
    """Scenario 1: Tab with NO_RESULT closes without prompting."""
    dialogs = RecordingDialogPresenter()
    window, ctrl = _setup_window_with_tabs(dialogs)
    _add_tab(window, "tab-1", has_run=False)

    window._tabs.tabCloseRequested.emit(0)

    assert len(dialogs.calls) == 0
    ctrl.close_tab.assert_called_once_with("tab-1")


def test_tab_close_with_saved_data_closes_immediately(qapp):
    """Scenario 1: Tab with SAVED data closes without prompting."""
    dialogs = RecordingDialogPresenter()
    window, ctrl = _setup_window_with_tabs(dialogs)
    tab = _add_tab(window, "tab-1", has_run=True, saved=True)
    assert tab.has_unsaved_data() is False

    window._tabs.tabCloseRequested.emit(0)

    assert len(dialogs.calls) == 0
    ctrl.close_tab.assert_called_once_with("tab-1")


@pytest.mark.parametrize(
    "data_status", [SaveStatus.NOT_SAVED, SaveStatus.UNSAVED_CHANGES]
)
def test_tab_close_with_unsaved_data_declined_keeps_tab(qapp, data_status):
    """Cancelling tab-close leaves either unsaved data state untouched."""
    dialogs = RecordingDialogPresenter(destructive_answers=[False])
    window, ctrl = _setup_window_with_tabs(dialogs)
    tab = _add_tab(window, "tab-1", has_run=True, saved=False)
    tab.update_interaction_state(_snapshot("tab-1", data_status=data_status))
    assert tab.has_unsaved_data() is True

    window._tabs.tabCloseRequested.emit(0)

    assert len(dialogs.calls) == 1
    call = dialogs.calls[0]
    assert call.kind == "destructive_confirm"
    assert call.title == "Unsaved measurement data"
    assert call.action_text == "Discard and Close"
    ctrl.close_tab.assert_not_called()
    assert window.has_tab_widget("tab-1") is True


def test_tab_close_with_unsaved_data_confirmed_closes_tab(qapp):
    """Scenario 2: Confirming tab-close closes the unsaved tab."""
    dialogs = RecordingDialogPresenter(destructive_answers=[True])
    window, ctrl = _setup_window_with_tabs(dialogs)
    tab = _add_tab(window, "tab-1", has_run=True, saved=False)
    assert tab.has_unsaved_data() is True

    window._tabs.tabCloseRequested.emit(0)

    assert len(dialogs.calls) == 1
    call = dialogs.calls[0]
    assert call.kind == "destructive_confirm"
    assert call.title == "Unsaved measurement data"
    assert call.action_text == "Discard and Close"
    ctrl.close_tab.assert_called_once_with("tab-1")


def test_moved_tab_with_unsaved_data_closes_by_captured_identity(qapp):
    """Scenario 3: A tab moved before close request closes by captured widget identity."""
    dialogs = RecordingDialogPresenter(destructive_answers=[True])
    window, ctrl = _setup_window_with_tabs(dialogs)
    tab_a = _add_tab(window, "tab-a", has_run=True, saved=False)
    _add_tab(window, "tab-b", has_run=False)
    assert tab_a.has_unsaved_data() is True

    tab_bar = window._tabs.tabBar()
    assert tab_bar is not None
    # Move tab-a from index 0 to index 1
    tab_bar.moveTab(0, 1)
    assert window._tabs.widget(1) is tab_a

    # Request close on moved tab (now at index 1)
    window._tabs.tabCloseRequested.emit(1)

    assert len(dialogs.calls) == 1
    assert dialogs.calls[0].kind == "destructive_confirm"
    assert dialogs.calls[0].title == "Unsaved measurement data"
    ctrl.close_tab.assert_called_once_with("tab-a")


def test_tab_moved_while_confirmation_pending_closes_captured_widget(qapp):
    """Scenario 3: Tab moved while confirmation prompt is open closes captured identity."""
    window, ctrl = _setup_window_with_tabs()
    tab_a = _add_tab(window, "tab-a", has_run=True, saved=False)
    _add_tab(window, "tab-b", has_run=False)

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
        ) -> None:
            super().destructive_confirm(
                parent,
                title,
                message,
                action_text=action_text,
                on_decision=lambda _: None,
                default=default,
            )
            decision_cb.append(on_decision)

    deferred_presenter = _DeferredPresenter(destructive_answers=[True])
    window._dialog_presenter = deferred_presenter

    # Request close on tab-a at index 0 via signal
    window._tabs.tabCloseRequested.emit(0)
    assert len(decision_cb) == 1

    # While dialog is pending, move tab-a to index 1
    tab_bar = window._tabs.tabBar()
    assert tab_bar is not None
    tab_bar.moveTab(0, 1)
    assert window._tabs.widget(1) is tab_a

    # User confirms the dialog
    decision_cb[0](True)

    ctrl.close_tab.assert_called_once_with("tab-a")


def test_tab_close_suppresses_duplicate_prompt_while_decision_pending(qapp):
    """Scenario 2/3: Duplicate tab close requests for same tab are suppressed while prompt is open."""
    window, ctrl = _setup_window_with_tabs()
    _add_tab(window, "tab-1", has_run=True, saved=False)

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
        ) -> None:
            super().destructive_confirm(
                parent,
                title,
                message,
                action_text=action_text,
                on_decision=lambda _: None,
                default=default,
            )
            decision_cb.append(on_decision)

    deferred_presenter = _DeferredPresenter(destructive_answers=[True])
    window._dialog_presenter = deferred_presenter

    # First request
    window._tabs.tabCloseRequested.emit(0)
    assert len(deferred_presenter.calls) == 1

    # Second request while prompt is pending
    window._tabs.tabCloseRequested.emit(0)
    assert len(deferred_presenter.calls) == 1

    # Confirm
    decision_cb[0](True)
    ctrl.close_tab.assert_called_once_with("tab-1")


def test_app_close_idle_with_no_unsaved_tabs_shuts_down_without_prompt(qapp):
    """Scenario 6: App close with neither unsaved data nor active operations closes cleanly."""
    dialogs = RecordingDialogPresenter()
    window, ctrl = _setup_window_with_tabs(dialogs, active_operations=0)
    _add_tab(window, "tab-1", has_run=True, saved=True)

    window.close()

    assert len(dialogs.calls) == 0
    QCoreApplication.processEvents()
    _shutdown_callback(ctrl)


def test_closing_window_leaves_plot_host_available_until_app_quits(qapp):
    window, _ctrl = _setup_window_with_tabs(
        RecordingDialogPresenter(), active_operations=0
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


def test_app_close_with_unsaved_tabs_prompts_and_declining_cancels(qapp):
    """Scenario 4: App close with unsaved data warns; cancelling leaves app open."""
    dialogs = RecordingDialogPresenter(destructive_answers=[False])
    window, ctrl = _setup_window_with_tabs(dialogs, active_operations=0)
    _add_tab(window, "tab-1", has_run=True, saved=False)

    window.close()

    assert len(dialogs.calls) == 1
    call = dialogs.calls[0]
    assert call.kind == "destructive_confirm"
    assert call.title == "Unsaved measurement data"
    assert call.action_text == "Discard and Close"
    QCoreApplication.processEvents()
    ctrl.begin_shutdown.assert_not_called()


def test_app_close_with_unsaved_tabs_prompts_and_confirming_shuts_down(qapp):
    """Scenario 4: App close with unsaved data warns; confirming begins shutdown."""
    dialogs = RecordingDialogPresenter(destructive_answers=[True])
    window, ctrl = _setup_window_with_tabs(dialogs, active_operations=0)
    _add_tab(window, "tab-1", has_run=True, saved=False)
    _add_tab(window, "tab-2", has_run=False)

    window.close()

    assert len(dialogs.calls) == 1
    call = dialogs.calls[0]
    assert call.kind == "destructive_confirm"
    assert call.title == "Unsaved measurement data"
    assert call.action_text == "Discard and Close"
    QCoreApplication.processEvents()
    _shutdown_callback(ctrl)


def test_app_close_with_both_unsaved_tabs_and_active_operations(qapp):
    """Scenario 5: Both risks present; prompt communicates both, confirm shuts down."""
    dialogs = RecordingDialogPresenter(destructive_answers=[True])
    window, ctrl = _setup_window_with_tabs(dialogs, active_operations=3)
    _add_tab(window, "tab-1", has_run=True, saved=False)

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


def test_app_close_with_both_unsaved_tabs_and_active_operations_declined(qapp):
    """Scenario 5: Both risks present; decline preserves app state."""
    dialogs = RecordingDialogPresenter(destructive_answers=[False])
    window, ctrl = _setup_window_with_tabs(dialogs, active_operations=2)
    _add_tab(window, "tab-1", has_run=True, saved=False)

    window.close()

    assert len(dialogs.calls) == 1
    call = dialogs.calls[0]
    assert call.kind == "destructive_confirm"
    assert call.title == "Unsaved data and operations in progress"
    assert call.action_text == "Discard and Close"
    QCoreApplication.processEvents()
    ctrl.begin_shutdown.assert_not_called()


def test_app_close_with_active_operations_only_uses_confirm_async(qapp):
    """Active-operation-only prompt uses non-blocking confirm_async."""
    dialogs = RecordingDialogPresenter(confirm_answers=[True])
    window, ctrl = _setup_window_with_tabs(dialogs, active_operations=2)
    _add_tab(window, "tab-1", has_run=True, saved=True)

    window.close()

    assert len(dialogs.calls) == 1
    call = dialogs.calls[0]
    assert call.kind == "confirm"
    assert call.title == "Operations in progress"
    assert "2 operation(s)" in call.message
    QCoreApplication.processEvents()
    _shutdown_callback(ctrl)


def test_app_close_suppresses_duplicate_prompt_while_decision_pending(qapp):
    """Scenario 4: Duplicate app close requests while prompt is open are suppressed."""
    window, ctrl = _setup_window_with_tabs(active_operations=0)
    _add_tab(window, "tab-1", has_run=True, saved=False)

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
        ) -> None:
            super().destructive_confirm(
                parent,
                title,
                message,
                action_text=action_text,
                on_decision=lambda _: None,
                default=default,
            )
            decision_cb.append(on_decision)

    deferred_presenter = _DeferredPresenter(destructive_answers=[True])
    window._dialog_presenter = deferred_presenter

    # First close
    window.close()
    assert len(deferred_presenter.calls) == 1

    # Second close while dialog is pending
    window.close()
    assert len(deferred_presenter.calls) == 1

    decision_cb[0](True)
    QCoreApplication.processEvents()
    _shutdown_callback(ctrl)


def test_app_close_reprompts_if_unsaved_data_appears_before_decision(qapp):
    """A newly completed result upgrades an active-only prompt before shutdown."""
    dialogs = _DeferredClosePresenter()
    window, ctrl = _setup_window_with_tabs(dialogs, active_operations=1)
    tab = _add_tab(window, "tab-1", has_run=True, saved=True)

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


def test_app_close_suppresses_reentry_while_shutdown_is_coordinating(qapp):
    """Accepted close owns the lifecycle until the coordinator callback runs."""
    dialogs = RecordingDialogPresenter(confirm_answers=[True])
    window, ctrl = _setup_window_with_tabs(dialogs, active_operations=1)
    _add_tab(window, "tab-1", has_run=True, saved=True)

    window.close()
    window.close()
    QCoreApplication.processEvents()
    _shutdown_callback(ctrl)

    window.close()

    assert len(dialogs.calls) == 1
    ctrl.begin_shutdown.assert_called_once()


def test_app_close_suppresses_duplicate_idle_shutdown_requests(qapp):
    """Repeated idle close events schedule only one shutdown coordinator run."""
    dialogs = RecordingDialogPresenter()
    window, ctrl = _setup_window_with_tabs(dialogs, active_operations=0)
    _add_tab(window, "tab-1", has_run=True, saved=True)

    window.close()
    window.close()
    QCoreApplication.processEvents()
    _shutdown_callback(ctrl)

    window.close()

    assert dialogs.calls == []
    ctrl.begin_shutdown.assert_called_once()


def test_app_close_rechecks_unsaved_data_after_operations_settle(qapp):
    """A result completed during shutdown gets a final data-loss decision."""
    dialogs = RecordingDialogPresenter(
        confirm_answers=[True], destructive_answers=[False]
    )
    window, ctrl = _setup_window_with_tabs(dialogs, active_operations=1)
    tab = _add_tab(window, "tab-1", has_run=True, saved=True)

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


def test_app_close_waits_for_pending_tab_close_decision(qapp):
    """Tab and app close confirmations never overlap."""
    dialogs = _DeferredClosePresenter()
    window, ctrl = _setup_window_with_tabs(dialogs, active_operations=0)
    _add_tab(window, "tab-1", has_run=True, saved=False)

    window._tabs.tabCloseRequested.emit(0)
    window.close()

    assert [call.kind for call in dialogs.calls] == ["destructive_confirm"]
    ctrl.begin_shutdown.assert_not_called()

    dialogs.destructive_decisions.pop()(False)


def test_repeated_programmatic_shutdown_waits_for_pre_prompt_coordination(qapp):
    """Repeated RPC takeover cannot bypass the captured shutdown coordinator."""
    dialogs = _DeferredClosePresenter()
    window, ctrl = _setup_window_with_tabs(dialogs, active_operations=1)
    _add_tab(window, "tab-1", has_run=True, saved=True)

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


def test_programmatic_shutdown_supersedes_user_shutdown_coordination(qapp):
    """RPC intent forces close when user-started coordination later settles."""
    dialogs = RecordingDialogPresenter(confirm_answers=[True])
    window, ctrl = _setup_window_with_tabs(dialogs, active_operations=1)
    tab = _add_tab(window, "tab-1", has_run=True, saved=True)

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


def test_programmatic_shutdown_supersedes_final_unsaved_prompt(qapp):
    """RPC intent tears down without waiting for a post-settlement dialog."""
    dialogs = _DeferredClosePresenter()
    window, ctrl = _setup_window_with_tabs(dialogs, active_operations=1)
    tab = _add_tab(window, "tab-1", has_run=True, saved=True)

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


def test_programmatic_request_shutdown_bypasses_unsaved_guard(qapp):
    """Programmatic request_shutdown() remains non-interactive even with unsaved data."""
    dialogs = RecordingDialogPresenter()
    window, ctrl = _setup_window_with_tabs(dialogs, active_operations=1)
    _add_tab(window, "tab-1", has_run=True, saved=False)

    window.request_shutdown()

    assert len(dialogs.calls) == 0
    QCoreApplication.processEvents()
    ctrl.begin_shutdown.assert_called_once_with(window._perform_close)


@dataclass
class RunFormFixture:
    window: MainWindow
    ctrl: MagicMock
    owner: CfgResource
    tab: ExpTabWidget
    line: QLineEdit
    runs: list[tuple[CfgRef, AcceptedConfig]]
    source_fault: list[bool]


@pytest.fixture
def run_form(
    qapp: QApplication, request: pytest.FixtureRequest
) -> Iterator[RunFormFixture]:
    ctrl = _apply_window_defaults(_mock_ctrl())
    ctrl.bus = EventBus()
    snapshot = _snapshot(
        "pending-tab",
        supports_analysis=False,
        has_run_result=False,
        has_analyze_result=False,
        has_figure=False,
    )
    ctrl.get_tab_snapshot.return_value = snapshot
    sources = State(
        SessionEnv(md=MetaDict(None), ml=ModuleLibrary(None), soc=None, soccfg=None)
    )
    bindings = MeasureCfgBindings(PublishedHost(sources))
    source_fault = [False]

    def resolution() -> CfgResolution:
        if source_fault[0]:
            raise RuntimeError("Source snapshot failed")
        return bindings.snapshot_from_state(sources, captured_values={})

    schema = CfgSchema(
        CfgSectionSpec(fields={"value": ScalarSpec("Value", float)}),
        CfgSectionValue({"value": DirectValue(2.0)}),
    )
    range_mode = getattr(request, "param", None)
    if range_mode is not None:
        centered = range_mode == "center-span"
        schema = CfgSchema(
            CfgSectionSpec(
                fields={
                    "value": ScalarSpec("Value", float),
                    "range": CenteredSweepSpec() if centered else SweepSpec(),
                }
            ),
            CfgSectionValue(
                {
                    "value": DirectValue(2.0),
                    "range": CenteredSweepValue(1.0, 2.0, 3)
                    if centered
                    else SweepValue(0.0, 2.0, 3),
                }
            ),
        )
    owner = CfgResource(
        lambda: schema,
        resolution=resolution,
        make_range=lambda start, stop, *, expts: (start, stop, expts),
    )
    ctrl.cfg_resources.lookup.side_effect = lambda tab_id: owner
    runs: list[tuple[CfgRef, AcceptedConfig]] = []

    def start(tab_id: str, ref: CfgRef) -> None:
        runs.append((ref, owner.accept(ref.revision)))

    ctrl.start_run.side_effect = start
    window = MainWindow(ctrl, dialog_presenter=RecordingDialogPresenter())
    window.add_tab_widget("pending-tab", "fake")
    window.resize(1100, 700)
    window.show()
    qapp.processEvents()
    tabs = window.findChildren(ExpTabWidget)
    assert len(tabs) == 1
    tab = tabs[0]
    widget = tab.cfg_form.findChild(QWidget, "cfgInput:value")
    assert widget is not None
    line = widget.findChild(QLineEdit)
    assert line is not None
    try:
        yield RunFormFixture(window, ctrl, owner, tab, line, runs, source_fault)
    finally:
        window.remove_tab_widget("pending-tab")
        window.deleteLater()
        qapp.processEvents()


def edit_run_input(fx: RunFormFixture, text: str) -> None:
    fx.line.setFocus()
    fx.line.selectAll()
    for char in text:
        for event_type in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            QApplication.sendEvent(
                fx.line, QKeyEvent(event_type, 0, Qt.KeyboardModifier.NoModifier, char)
            )


def test_run_submits_local_input_and_uses_returned_ref(
    run_form: RunFormFixture,
) -> None:
    fx = run_form
    base = fx.owner.observe().ref
    edit_run_input(fx, "3.5")
    assert fx.owner.observe().ref == base
    assert fx.tab.run_btn.isEnabled()
    fx.tab.run_btn.click()
    assert len(fx.runs) == 1
    ref, accepted = fx.runs[0]
    assert ref == fx.owner.observe().ref == fx.tab.cfg_form.current_ref()
    assert ref.revision == base.revision + 1
    assert accepted.values == {"value": 3.5}
    assert not fx.tab.cfg_form.has_pending()


@pytest.mark.parametrize("run_form", ["endpoints", "center-span"], indirect=True)
def test_run_uses_the_last_pending_sampling_operation(
    run_form: RunFormFixture,
) -> None:
    fx = run_form
    widget = fx.tab.cfg_form.findChild(QWidget, "cfgInput:range")
    assert widget is not None
    points = widget.findChild(QLineEdit, "expts")
    step = widget.findChild(QLineEdit, "step")
    assert points is not None and step is not None
    for line, text in ((points, "9"), (step, "0.5"), (points, "7")):
        line.setFocus()
        line.selectAll()
        for char in text:
            for event_type in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
                QApplication.sendEvent(
                    line, QKeyEvent(event_type, 0, Qt.KeyboardModifier.NoModifier, char)
                )
    fx.tab.run_btn.click()
    assert len(fx.runs) == 1
    ref, accepted = fx.runs[0]
    assert ref == fx.owner.observe().ref == fx.tab.cfg_form.current_ref()
    assert accepted.values == {"value": 2.0, "range": (0, 2, 7)}
    assert not fx.tab.cfg_form.has_pending()


@pytest.mark.parametrize("text", ["-", "nan"], ids=["incomplete", "nonfinite"])
def test_run_publishes_invalid_input_but_never_runs_old_values(
    run_form: RunFormFixture,
    text: str,
) -> None:
    fx = run_form
    edit_run_input(fx, text)
    fx.window.run_or_stop_tab("pending-tab")
    assert fx.runs == []
    assert not fx.tab.cfg_form.is_valid()
    assert fx.line.text() == text


def test_run_stale_keeps_input_focus_selection_and_does_not_retry(
    run_form: RunFormFixture,
) -> None:
    fx = run_form
    edit_run_input(fx, "3.51")
    fx.line.setSelection(1, 2)
    current = fx.owner.edit(
        fx.owner.observe().ref.revision, (CfgEdit(("value",), DirectValue(8.0)),)
    )
    fx.window.run_or_stop_tab("pending-tab")
    assert fx.runs == []
    assert fx.owner.observe().ref == current.ref
    assert fx.tab.cfg_form.has_pending()
    assert fx.line.text() == "3.51"
    assert fx.line.hasFocus() and fx.line.selectedText() == ".5"


def test_run_unavailable_stops_without_discarding_input(
    run_form: RunFormFixture,
) -> None:
    fx = run_form
    edit_run_input(fx, "3.5")
    fx.owner.revoke()
    fx.window.run_or_stop_tab("pending-tab")
    assert fx.runs == []
    assert fx.tab.cfg_form.has_pending()
    assert fx.line.text() == "3.5"


def test_run_submission_fault_propagates_without_using_old_values(
    run_form: RunFormFixture,
) -> None:
    fx = run_form
    edit_run_input(fx, "3.5")
    base = fx.owner.observe().ref
    fx.source_fault[0] = True
    with pytest.raises(RuntimeError, match="Source snapshot failed"):
        fx.window.run_or_stop_tab("pending-tab")
    assert fx.owner.observe().ref == base
    assert fx.runs == []
    assert fx.tab.cfg_form.has_pending()
    assert fx.line.text() == "3.5"


def test_run_can_submit_a_repair_to_invalid_published_config(
    run_form: RunFormFixture,
) -> None:
    fx = run_form
    fx.owner.edit(
        fx.owner.observe().ref.revision,
        (CfgEdit(("value",), DirectValue(None, raw="-")),),
    )
    assert not fx.tab.run_btn.isEnabled()
    edit_run_input(fx, "4.5")
    assert fx.tab.run_btn.isEnabled()
    fx.tab.run_btn.click()
    assert len(fx.runs) == 1
    assert fx.runs[0][1].values == {"value": 4.5}


@pytest.mark.parametrize("block", ["busy", "context", "soc", "global-run"])
def test_pending_input_does_not_bypass_other_run_gates(
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


def test_stop_does_not_submit_local_input(run_form: RunFormFixture) -> None:
    fx = run_form
    edit_run_input(fx, "3.5")
    base = fx.owner.observe().ref
    fx.ctrl.get_tab_snapshot.return_value = _snapshot(
        "pending-tab",
        supports_analysis=False,
        is_running=True,
    )
    fx.window.run_or_stop_tab("pending-tab")
    assert fx.owner.observe().ref == base
    assert fx.tab.cfg_form.has_pending()
    assert fx.runs == []
    fx.ctrl.cancel_run.assert_called_once_with()


def test_reset_explicitly_discards_pending_input(run_form: RunFormFixture) -> None:
    fx = run_form
    # Reset's confirmation is the explicit authorization to discard local input.
    dialogs = RecordingDialogPresenter(confirm_answers=[True])
    # Reopen through the public view lifecycle using the same cfg owner.
    fx.window.remove_tab_widget("pending-tab")
    fx.window.deleteLater()
    fx.window = MainWindow(fx.ctrl, dialog_presenter=dialogs)
    fx.window.add_tab_widget("pending-tab", "fake")
    tab = fx.window.findChildren(ExpTabWidget)[0]
    line = tab.cfg_form.findChild(QLineEdit)
    assert line is not None
    line.selectAll()
    for char in "3.5":
        QApplication.sendEvent(
            line,
            QKeyEvent(QEvent.Type.KeyPress, 0, Qt.KeyboardModifier.NoModifier, char),
        )
        QApplication.sendEvent(
            line,
            QKeyEvent(QEvent.Type.KeyRelease, 0, Qt.KeyboardModifier.NoModifier, char),
        )
    assert tab.cfg_form.has_pending()
    tab.reset_btn.click()
    assert not tab.cfg_form.has_pending()
    assert fx.owner.accept(tab.cfg_form.current_ref().revision).values == {"value": 2.0}
