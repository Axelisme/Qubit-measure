"""Focused Data-center behavior tests for TKT-001 save-subtab-redesign.

Validates S1-S3 acceptance via production ExpTabWidget / MainWindow seams.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from qtpy.QtWidgets import QApplication, QLabel, QLineEdit, QPushButton, QTextEdit
from zcu_tools.experiment.v2_gui.measure.adapters.fake import FakeAdapter
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    ContextReadiness,
    SessionEnv,
)
from zcu_tools.gui.app.measure.artifact_tracker import ArtifactKind, SaveStatus
from zcu_tools.gui.app.measure.events.completion import SaveDataFinishedPayload
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote.handlers.run_save import h_tab_save_image
from zcu_tools.gui.app.measure.services import TabSnapshot
from zcu_tools.gui.app.measure.services.save_control import SaveControlFacet
from zcu_tools.gui.app.measure.services.tab import TabService
from zcu_tools.gui.app.measure.state import State, TabInteractionState
from zcu_tools.gui.app.measure.ui.artifact_save_center import ArtifactSaveCenter
from zcu_tools.gui.event_bus import BaseEventBus as EventBus

from tests.gui.app.measure.ui._artifact_snapshots import with_artifacts


@dataclass
class _DummyParams:
    x: int = 1


def _require_qapp() -> QApplication:
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    return app


def _mock_ctrl() -> MagicMock:
    ctrl = MagicMock()
    ctrl.get_left_panel_width.return_value = 500
    ctrl.get_tab_adapter_name.return_value = "fake"
    ctrl.get_adapter_guide.return_value = {}
    ctrl.progress_control.attach_progress.return_value = lambda: None
    ctrl.progress_control.progress_bars.return_value = []
    ctrl.active_operation_count.return_value = 0
    ctrl.has_agent_connected.return_value = False
    from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
    from zcu_tools.gui.app.measure.specs import make_pulse_spec
    from zcu_tools.gui.cfg import CfgSchema, make_default_value

    spec = make_pulse_spec()
    draft = MeasureCfgBindings(ctrl).new_draft(
        CfgSchema(spec, make_default_value(spec))
    )
    ctrl.open_seeded_cfg_editor.return_value = ("editor-tab", ())
    ctrl.get_cfg_editor_draft.return_value = draft
    return ctrl


def _snapshot(
    tab_id: str,
    *,
    has_run: bool = False,
    has_analysis: bool = False,
    has_post: bool = False,
    analysis_mode=AnalysisMode.FIT,
    post_cap: bool = False,
    load_cap: bool = False,
    has_active_context: bool = True,
    has_context: bool = True,
    is_running: bool = False,
    is_analyzing: bool = False,
    is_saving: bool = False,
    data_path: str | None = None,
    analysis_path: str | None = None,
    post_path: str | None = None,
    analysis_has_figure: bool | None = None,
    post_has_figure: bool | None = None,
    data_status: SaveStatus | None = None,
    analysis_status: SaveStatus | None = None,
    post_status: SaveStatus | None = None,
) -> TabSnapshot:
    from zcu_tools.gui.app.measure.services.ports import (
        AnalysisPaneSnapshot,
        PathResourceSnapshot,
        PostAnalysisPaneSnapshot,
        RunPaneSnapshot,
        SavePaneSnapshot,
        TabPathsSnapshot,
    )

    caps = AdapterCapabilities(
        analysis=analysis_mode, post_analysis=post_cap, load_data=load_cap
    )
    run_result = object() if has_run else None
    ana_result = object() if has_analysis else None
    post_result = object() if has_post else None
    if analysis_has_figure is None:
        fig = Figure() if has_analysis else None
    elif analysis_has_figure:
        fig = Figure()
    else:
        fig = None
    if post_has_figure is None:
        post_fig = Figure() if has_post else None
    elif post_has_figure:
        post_fig = Figure()
    else:
        post_fig = None

    data_ps = PathResourceSnapshot(
        override=data_path, path=data_path or ("/tmp/data.h5" if has_run else None)
    )
    ana_ps = PathResourceSnapshot(
        override=analysis_path,
        path=analysis_path or ("/tmp/a.png" if has_analysis else None),
    )
    post_ps = PathResourceSnapshot(
        override=post_path, path=post_path or ("/tmp/p.png" if has_post else None)
    )

    snapshot = TabSnapshot(
        adapter_name="fake",
        cfg_schema=MagicMock(),
        tab_id=tab_id,
        interaction=TabInteractionState(
            global_run_active=False,
            is_running=is_running,
            is_analyzing=is_analyzing,
            is_saving_data=is_saving,
            has_context=has_context,
            has_active_context=has_active_context,
            has_soc=True,
            has_run_result=has_run,
            has_analyze_result=has_analysis,
            has_figure=bool(fig is not None),
            has_post_analyze_result=has_post,
        ),
        capabilities=caps,
        run=RunPaneSnapshot(result=run_result, source_path=None),
        analysis=AnalysisPaneSnapshot(
            params=_DummyParams() if has_analysis else None,
            result=ana_result,
            figure=fig,
            writeback_items=(),
            image_path=ana_ps,
        ),
        post_analysis=PostAnalysisPaneSnapshot(
            params=_DummyParams() if has_post else None,
            result=post_result,
            figure=post_fig,
            writeback_items=(),
            image_path=post_ps,
        ),
        save=SavePaneSnapshot(data_path=data_ps),
        paths=TabPathsSnapshot(
            data=data_ps, analysis_image=ana_ps, post_analysis_image=post_ps
        ),
    )
    statuses = {
        kind: status
        for kind, status in (
            (ArtifactKind.DATA, data_status),
            (ArtifactKind.ANALYSIS, analysis_status),
            (ArtifactKind.POST_ANALYSIS, post_status),
        )
        if status is not None
    }
    return with_artifacts(snapshot, statuses)


@pytest.fixture
def exp_tab_factory(qapp, monkeypatch):
    import zcu_tools.gui.app.measure.ui.exp_tab_widget as mod

    orig = mod.ExpTabWidget._populate_cfg

    def stub(self, schema, ctrl):
        self._cfg_editor_id = "probe-editor"
        self.cfg_form.is_valid = lambda: True  # type: ignore[method-assign]
        self.cfg_form.first_invalid_reason = lambda: None  # type: ignore[method-assign]

    monkeypatch.setattr(mod.ExpTabWidget, "_populate_cfg", stub)
    orig_attach = mod.attach_existing_figure_to_container

    def mock_attach(fig, container):
        from qtpy.QtWidgets import QWidget

        w = QWidget()
        w.figure = fig  # type: ignore[attr-defined]
        container.attach_canvas(w)
        w.draw = lambda: None  # type: ignore[attr-defined]
        return w

    monkeypatch.setattr(mod, "attach_existing_figure_to_container", mock_attach)
    yield mod.ExpTabWidget
    monkeypatch.setattr(mod.ExpTabWidget, "_populate_cfg", orig)
    monkeypatch.setattr(mod, "attach_existing_figure_to_container", orig_attach)


# ---------------------------------------------------------------------------
# A1 composition
# ---------------------------------------------------------------------------


def test_data_subtab_contains_save_center_and_order(exp_tab_factory):
    ctrl = _mock_ctrl()
    caps = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=True, load_data=True
    )
    snap = _snapshot(
        "tab-1",
        has_run=True,
        has_analysis=True,
        has_post=True,
        analysis_mode=AnalysisMode.FIT,
        post_cap=True,
        load_cap=True,
    )
    tab = exp_tab_factory("tab-1", ctrl, caps)
    tab.attach(snap, MagicMock())
    labels = [tab._left_tabs.tabText(i) for i in range(tab._left_tabs.count())]
    assert labels == ["Run", "Analysis", "Post-Analysis", "Data", "Guide"]
    heading = tab._save_center.findChildren(QLabel)
    assert any(lbl.text() == "Save results" for lbl in heading)
    assert tab._save_center.artifact_kinds == [
        ArtifactKind.DATA,
        ArtifactKind.ANALYSIS,
        ArtifactKind.POST_ANALYSIS,
    ]
    # Analysis/Post panels no longer own image save controls
    assert not any(
        isinstance(c, QLineEdit) and c.placeholderText() == "/tmp/image.png"
        for c in tab._analysis_panel.findChildren(QLineEdit)
    )
    assert not any(
        isinstance(c, QPushButton) and c.text() == "Save Image"
        for c in tab._analysis_panel.findChildren(QPushButton)
    )
    # Data center hosts them via its narrow interface
    assert tab._save_center.has_artifact(ArtifactKind.DATA)
    assert tab._save_center.has_artifact(ArtifactKind.ANALYSIS)
    assert tab._save_center.has_artifact(ArtifactKind.POST_ANALYSIS)
    # Verify placeholder via findChildren on center
    placeholders = {
        c.placeholderText() for c in tab._save_center.findChildren(QLineEdit)
    }
    assert "/tmp/data.hdf5" in placeholders
    assert "/tmp/image.png" in placeholders
    assert "/tmp/post_image.png" in placeholders
    tab.deleteLater()
    _require_qapp().processEvents()


def test_measurement_always_post_conditional(exp_tab_factory):
    ctrl = _mock_ctrl()
    caps_none = AdapterCapabilities(
        analysis=AnalysisMode.NONE, post_analysis=False, load_data=False
    )
    tab_none = exp_tab_factory("tab-1", ctrl, caps_none)
    snap_none = _snapshot(
        "tab-1",
        has_run=False,
        analysis_mode=AnalysisMode.NONE,
        post_cap=False,
        load_cap=False,
    )
    tab_none.attach(snap_none, MagicMock())
    assert tab_none._save_center.artifact_kinds == [ArtifactKind.DATA]
    assert not tab_none._save_center.has_artifact(ArtifactKind.ANALYSIS)
    caps_a = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=False, load_data=False
    )
    tab_a = exp_tab_factory("tab-2", ctrl, caps_a)
    snap_a = _snapshot(
        "tab-2",
        has_run=True,
        has_analysis=False,
        analysis_mode=AnalysisMode.FIT,
        post_cap=False,
        load_cap=False,
    )
    tab_a.attach(snap_a, MagicMock())
    assert tab_a._save_center.artifact_kinds == [
        ArtifactKind.DATA,
        ArtifactKind.ANALYSIS,
    ]
    caps_both = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=True, load_data=False
    )
    tab_both = exp_tab_factory("tab-3", ctrl, caps_both)
    snap_both = _snapshot(
        "tab-3",
        has_run=True,
        has_analysis=True,
        has_post=True,
        analysis_mode=AnalysisMode.FIT,
        post_cap=True,
        load_cap=False,
    )
    tab_both.attach(snap_both, MagicMock())
    assert tab_both._save_center.artifact_kinds == [
        ArtifactKind.DATA,
        ArtifactKind.ANALYSIS,
        ArtifactKind.POST_ANALYSIS,
    ]

    # ---------------------------------------------------------------------------
    # A2 row composition and bottom layout
    # ---------------------------------------------------------------------------
    tab_none.deleteLater()
    tab_a.deleteLater()
    tab_both.deleteLater()
    _require_qapp().processEvents()


def test_artifact_rows_have_status_path_browse_save_and_comment(exp_tab_factory):
    ctrl = _mock_ctrl()
    caps = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=True, load_data=True
    )
    snap = _snapshot(
        "tab-1",
        has_run=True,
        has_analysis=True,
        has_post=True,
        analysis_mode=AnalysisMode.FIT,
        post_cap=True,
        load_cap=True,
    )
    tab = exp_tab_factory("tab-1", ctrl, caps)
    tab.attach(snap, MagicMock())
    center = tab._save_center
    for kind in center.artifact_kinds:
        text = center.status_text(kind)
        assert any(sym in text for sym in ["—", "○", "●", "✓"])
        ss = center.status_color(kind)
        assert ss  # high contrast color
        assert center.has_artifact(kind)
        assert center.is_path_enabled(kind) is True
    # Measurement comment
    assert center.get_comment() == ""
    # Set comment and verify
    center.set_comment_text("hello")
    assert center.get_comment() == "hello"
    # Bottom Load/Save All
    assert center.load_button.text() == "Load Data"
    assert center.save_all_button.text() == "Save All"
    assert center.load_button.height() == center.save_all_button.height() == 36
    assert center.is_load_visible()
    caps_no_load = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=False, load_data=False
    )
    tab2 = exp_tab_factory("tab-2", ctrl, caps_no_load)
    snap2 = _snapshot(
        "tab-2",
        has_run=True,
        analysis_mode=AnalysisMode.FIT,
        post_cap=False,
        load_cap=False,
    )
    tab2.attach(snap2, MagicMock())
    assert not tab2._save_center.is_load_visible()
    assert not tab2._save_center.save_all_button.isHidden()
    assert tab2._save_center.is_save_all_enabled() is True

    # ---------------------------------------------------------------------------
    # A3 status lifecycle
    # ---------------------------------------------------------------------------
    tab.deleteLater()
    tab2.deleteLater()
    _require_qapp().processEvents()


def test_data_actions_appear_before_measurement_data_card(
    exp_tab_factory,
):
    ctrl = _mock_ctrl()
    caps = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=False, load_data=True
    )
    snap = _snapshot(
        "tab-1",
        analysis_mode=AnalysisMode.FIT,
        post_cap=False,
        load_cap=True,
    )
    tab = exp_tab_factory("tab-1", ctrl, caps)
    tab.attach(snap, MagicMock())
    tab.resize(1000, 900)
    tab._left_tabs.setCurrentWidget(tab._save_panel)
    tab.show()
    _require_qapp().processEvents()

    center = tab._save_center
    title = next(
        label
        for label in center.findChildren(QLabel)
        if label.text() == "Measurement data"
    )
    path_edit = center._path_edits[ArtifactKind.DATA]
    title_top = title.mapTo(center, title.rect().topLeft()).y()
    title_bottom = title.mapTo(center, title.rect().bottomLeft()).y()
    path_top = path_edit.mapTo(center, path_edit.rect().topLeft()).y()
    actions_top = center.save_all_button.mapTo(
        center, center.save_all_button.rect().topLeft()
    ).y()

    assert title.height() <= title.sizeHint().height() + 4
    assert path_top - title_bottom <= 16
    assert actions_top < title_top

    tab.close()
    tab.deleteLater()
    _require_qapp().processEvents()


def test_status_no_result_and_not_saved(exp_tab_factory):
    ctrl = _mock_ctrl()
    caps = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=True, load_data=False
    )
    tab = exp_tab_factory("tab-1", ctrl, caps)
    snap_none = _snapshot(
        "tab-1",
        has_run=False,
        has_analysis=False,
        has_post=False,
        analysis_mode=AnalysisMode.FIT,
        post_cap=True,
        load_cap=False,
    )
    tab.attach(snap_none, MagicMock())
    center = tab._save_center
    for kind in center.artifact_kinds:
        assert center.status_text(kind) == "— NO RESULT"
        assert center.is_save_enabled(kind) is False
        assert center.is_path_enabled(kind) is True
    snap_run = _snapshot(
        "tab-1",
        has_run=True,
        has_analysis=False,
        has_post=False,
        analysis_mode=AnalysisMode.FIT,
        post_cap=True,
        load_cap=False,
    )
    tab.update_interaction_state(snap_run)
    assert center.status_text(ArtifactKind.DATA) == "○ NOT SAVED"
    assert center.is_save_enabled(ArtifactKind.DATA) is True
    assert center.status_text(ArtifactKind.ANALYSIS) == "— NO RESULT"
    assert center.is_save_enabled(ArtifactKind.ANALYSIS) is False
    snap_ana = _snapshot(
        "tab-1",
        has_run=True,
        has_analysis=True,
        has_post=False,
        analysis_mode=AnalysisMode.FIT,
        post_cap=True,
        load_cap=False,
    )
    tab.update_interaction_state(snap_ana)
    assert center.status_text(ArtifactKind.ANALYSIS) == "○ NOT SAVED"
    assert center.is_save_enabled(ArtifactKind.ANALYSIS) is True
    assert center.status_text(ArtifactKind.POST_ANALYSIS) == "— NO RESULT"
    tab.deleteLater()
    _require_qapp().processEvents()


def test_status_text_renders_shared_snapshot(exp_tab_factory):
    tab = exp_tab_factory(
        "tab-1",
        _mock_ctrl(),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )
    tab.attach(_snapshot("tab-1", has_run=True), MagicMock())
    center = tab._save_center
    assert center.status_text(ArtifactKind.DATA) == "○ NOT SAVED"
    for status, label in (
        (SaveStatus.SAVED, "✓ SAVED"),
        (SaveStatus.UNSAVED_CHANGES, "● UNSAVED CHANGES"),
        (SaveStatus.NO_RESULT, "— NO RESULT"),
    ):
        tab.update_interaction_state(
            _snapshot(
                "tab-1", has_run=status is not SaveStatus.NO_RESULT, data_status=status
            )
        )
        assert center.status_text(ArtifactKind.DATA) == label
    tab.deleteLater()
    _require_qapp().processEvents()


def test_save_all_preserves_data_pane_editor_state(exp_tab_factory, qapp):
    from qtpy.QtCore import Qt
    from qtpy.QtTest import QTest
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = 0
    ctrl.has_agent_connected.return_value = False
    ctrl.has_tab.return_value = True
    ctrl.save_data = MagicMock(return_value="/tmp/data.h5")
    ctrl.save_image = MagicMock(return_value="/tmp/a.png")
    ctrl.save_post_image = MagicMock(return_value="/tmp/p.png")
    window = MainWindow(ctrl)

    caps = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=True, load_data=True
    )
    snap = _snapshot(
        "tab-1",
        has_run=True,
        has_analysis=True,
        has_post=True,
        analysis_mode=AnalysisMode.FIT,
        post_cap=True,
        load_cap=True,
        has_active_context=True,
    )
    tab_ctrl = _mock_ctrl()
    tab = exp_tab_factory("tab-1", tab_ctrl, caps)
    tab.attach(snap, window._tab_actions)
    ctrl.get_tab_snapshot.return_value = snap
    window._tab_widgets["tab-1"] = tab
    tab.update_interaction_state(snap)

    tab._left_tabs.setCurrentWidget(tab._save_panel)
    tab.show()
    qapp.processEvents()
    center = tab._save_center
    data_edit = center._path_edits[ArtifactKind.DATA]
    data_edit.setFocus()
    qapp.processEvents()
    assert qapp.focusWidget() is data_edit
    data_edit.setCursorPosition(4)
    data_edit.cursorBackward(True, 3)
    before = (
        data_edit.hasFocus(),
        data_edit.cursorPosition(),
        data_edit.selectionStart(),
        data_edit.selectedText(),
    )

    # A state refresh with identical text must not clear the user's selection.
    center.set_data_path(data_edit.text())
    assert (
        data_edit.hasFocus(),
        data_edit.cursorPosition(),
        data_edit.selectionStart(),
        data_edit.selectedText(),
    ) == before

    selected_pane = tab._left_tabs.currentWidget()
    data_edit_identity = center._path_edits[ArtifactKind.DATA]
    cast(Any, QTest).mouseClick(
        cast(QPushButton, center.save_all_button), Qt.MouseButton.LeftButton
    )
    qapp.processEvents()

    # This fake controller does not publish terminal State snapshots; dispatch
    # alone must not mark any artifact as saved.
    assert all(
        center.status_text(kind) == "○ NOT SAVED" for kind in center.artifact_kinds
    )

    window.handle_save_data_finished(
        SaveDataFinishedPayload(tab_id="tab-1", data_path="/tmp/data.h5")
    )
    qapp.processEvents()

    assert tab._left_tabs.currentWidget() is selected_pane
    assert tab._save_center is center
    assert center._path_edits[ArtifactKind.DATA] is data_edit_identity
    assert (
        data_edit.hasFocus(),
        data_edit.cursorPosition(),
        data_edit.selectionStart(),
        data_edit.selectedText(),
    ) == before
    assert center.status_text(ArtifactKind.DATA) == "○ NOT SAVED"
    window.deleteLater()
    tab.deleteLater()
    qapp.processEvents()


def test_changed_path_refresh_preserves_reverse_data_editor_state(
    exp_tab_factory, qapp
):
    ctrl = _mock_ctrl()
    caps = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=False, load_data=False
    )
    snap = _snapshot(
        "tab-1",
        has_run=True,
        analysis_mode=AnalysisMode.FIT,
        post_cap=False,
        load_cap=False,
    )
    tab = exp_tab_factory("tab-1", ctrl, caps)
    tab.attach(snap, MagicMock())
    tab.show()
    qapp.processEvents()

    center = tab._save_center
    data_edit = center._path_edits[ArtifactKind.DATA]
    data_edit.setFocus()
    qapp.processEvents()
    data_edit.setCursorPosition(4)
    data_edit.cursorBackward(True, 3)
    assert data_edit.cursorPosition() == 1
    assert data_edit.selectionStart() == 1
    assert data_edit.selectionLength() == 3
    assert data_edit.selectedText() == "tmp"

    center.set_data_path("/tmp/changed-data.h5")

    assert data_edit.hasFocus()
    assert data_edit.cursorPosition() == 1
    assert data_edit.selectionStart() == 1
    assert data_edit.selectionLength() == 3
    assert data_edit.selectedText() == "tmp"
    tab.deleteLater()
    qapp.processEvents()


def test_save_all_disabled_when_no_result(exp_tab_factory):
    ctrl = _mock_ctrl()
    caps = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=True, load_data=True
    )
    tab = exp_tab_factory("tab-1", ctrl, caps)
    snap_none = _snapshot(
        "tab-1",
        has_run=False,
        has_analysis=False,
        has_post=False,
        analysis_mode=AnalysisMode.FIT,
        post_cap=True,
        load_cap=True,
    )
    tab.attach(snap_none, MagicMock())
    assert tab._save_center.is_save_all_enabled() is False
    snap_some = _snapshot(
        "tab-1",
        has_run=True,
        has_analysis=False,
        has_post=False,
        analysis_mode=AnalysisMode.FIT,
        post_cap=True,
        load_cap=True,
        has_active_context=True,
    )
    tab.update_interaction_state(snap_some)
    assert tab._save_center.is_save_all_enabled() is True

    saved = _snapshot(
        "tab-1",
        has_run=True,
        has_analysis=True,
        post_cap=True,
        load_cap=True,
        data_status=SaveStatus.SAVED,
        analysis_status=SaveStatus.SAVED,
    )
    tab.update_interaction_state(saved)
    assert tab._save_center.is_save_all_enabled() is False
    assert tab._save_center.is_save_enabled(ArtifactKind.DATA) is True
    assert tab._save_center.is_save_enabled(ArtifactKind.ANALYSIS) is True
    changed = _snapshot(
        "tab-1",
        has_run=True,
        has_analysis=True,
        post_cap=True,
        load_cap=True,
        data_status=SaveStatus.UNSAVED_CHANGES,
        analysis_status=SaveStatus.SAVED,
    )
    tab.update_interaction_state(changed)
    assert tab._save_center.is_save_all_enabled() is True
    tab.deleteLater()
    _require_qapp().processEvents()


def test_load_data_gates(exp_tab_factory):
    ctrl = _mock_ctrl()
    caps_load = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=False, load_data=True
    )
    tab = exp_tab_factory("tab-1", ctrl, caps_load)
    snap_idle = _snapshot(
        "tab-1",
        has_run=False,
        analysis_mode=AnalysisMode.FIT,
        post_cap=False,
        load_cap=True,
        has_context=True,
        has_active_context=False,
    )
    tab.attach(snap_idle, MagicMock())
    assert tab._save_center.is_load_enabled() is True
    snap_no_ctx = _snapshot(
        "tab-1",
        has_run=False,
        analysis_mode=AnalysisMode.FIT,
        post_cap=False,
        load_cap=True,
        has_context=False,
        has_active_context=False,
    )
    tab.update_interaction_state(snap_no_ctx)
    assert tab._save_center.is_load_enabled() is False
    snap_busy = _snapshot(
        "tab-1",
        has_run=False,
        analysis_mode=AnalysisMode.FIT,
        post_cap=False,
        load_cap=True,
        has_context=True,
        is_analyzing=True,
    )
    tab.update_interaction_state(snap_busy)
    assert tab._save_center.is_load_enabled() is False
    caps_no = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=False, load_data=False
    )
    tab2 = exp_tab_factory("tab-2", ctrl, caps_no)
    snap2 = _snapshot(
        "tab-2",
        has_run=True,
        analysis_mode=AnalysisMode.FIT,
        post_cap=False,
        load_cap=False,
    )
    tab2.attach(snap2, MagicMock())
    assert not tab2._save_center.is_load_visible()
    assert not tab2._save_center.save_all_button.isHidden()
    tab.deleteLater()
    tab2.deleteLater()
    _require_qapp().processEvents()


def test_remote_save_completion_refreshes_state_owned_status(exp_tab_factory, qapp):
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = 0
    ctrl.has_agent_connected.return_value = False
    ctrl.has_tab.return_value = True
    window = MainWindow(ctrl)
    snap = _snapshot("tab-1", has_run=True, data_path="/gui.h5")
    assert snap.capabilities is not None
    tab = exp_tab_factory("tab-1", _mock_ctrl(), snap.capabilities)
    tab.attach(snap, MagicMock())
    window._tab_widgets["tab-1"] = tab
    center = tab._save_center
    assert center.status_text(ArtifactKind.DATA) == "○ NOT SAVED"

    ctrl.get_tab_snapshot.return_value = _snapshot(
        "tab-1", has_run=True, data_path="/gui.h5", data_status=SaveStatus.SAVED
    )
    window.handle_save_data_finished(
        SaveDataFinishedPayload(tab_id="tab-1", data_path="/gui_1.h5")
    )

    assert center.status_text(ArtifactKind.DATA) == "✓ SAVED"
    assert center.get_data_path() == "/gui.h5"
    window.deleteLater()
    tab.deleteLater()
    qapp.processEvents()


def test_comment_edit_updates_shared_draft_without_touching_paths(
    exp_tab_factory, qapp
):
    ctrl = _mock_ctrl()
    ctrl.save_control.set_comment = MagicMock()
    ctrl.update_tab_data_path = MagicMock()
    ctrl.update_tab_analysis_image_path = MagicMock()
    ctrl.update_tab_post_analysis_image_path = MagicMock()
    caps = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=False, load_data=False
    )
    tab = exp_tab_factory("tab-1", ctrl, caps)
    snap = _snapshot("tab-1", has_run=True)
    save = snap.save
    assert save is not None
    ctrl.get_tab_snapshot.return_value = snap

    def publish_comment(_tab_id: str, text: str) -> None:
        ctrl.get_tab_snapshot.return_value = replace(
            snap, save=replace(save, comment=text)
        )

    ctrl.save_control.set_comment.side_effect = publish_comment
    tab.attach(snap, MagicMock())
    ctrl.update_tab_data_path.reset_mock()
    center = tab._save_center
    center.set_comment_text("programmatic")
    ctrl.save_control.set_comment.assert_not_called()
    center._comment_edit.setPlainText("typed by user")
    _require_qapp().processEvents()
    ctrl.save_control.set_comment.assert_called_once_with("tab-1", "typed by user")
    ctrl.update_tab_data_path.assert_not_called()
    ctrl.update_tab_analysis_image_path.assert_not_called()
    ctrl.update_tab_post_analysis_image_path.assert_not_called()
    tab.deleteLater()
    _require_qapp().processEvents()


def _live_save_ui(tmp_path: Path):
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    state = State(
        SessionEnv(
            md=MagicMock(),
            ml=MagicMock(),
            soc=MagicMock(),
            soccfg=MagicMock(),
            result_dir=str(tmp_path / "result"),
            database_path=str(tmp_path / "database"),
            active_label="ctx001",
            readiness=ContextReadiness.ACTIVE,
        )
    )
    registry = Registry()
    registry.register("fake", FakeAdapter)
    tabs = TabService(state, registry, MagicMock())
    tab_id = tabs.new_tab("fake")
    state.update_tab_result(tab_id, object())
    save = MagicMock()
    save.start_save_data.return_value = "saved"
    ctrl = _mock_ctrl()
    ctrl.get_bus.return_value = EventBus()
    ctrl.has_tab.side_effect = state.has_tab
    ctrl.get_tab_snapshot.side_effect = tabs.get_snapshot
    ctrl.update_tab_data_path.side_effect = tabs.update_tab_data_path_override
    ctrl.save_control = SaveControlFacet(
        state=state,
        bus=ctrl.get_bus(),
        guard=MagicMock(),
        tab=tabs,
        save=save,
        notify_info=MagicMock(),
    )
    ctrl.save_data.side_effect = ctrl.save_control.save_data
    ctrl.save_artifacts.side_effect = ctrl.save_control.save_artifacts
    ctrl.save_image.side_effect = ctrl.save_control.save_image
    ctrl.save_post_image.side_effect = ctrl.save_control.save_post_image
    window = MainWindow(ctrl)
    window.add_tab_widget(tab_id, "fake")
    return window, state, tabs, save, tab_id, ctrl


@pytest.mark.parametrize("kind", [ArtifactKind.ANALYSIS, ArtifactKind.POST_ANALYSIS])
@pytest.mark.parametrize("fails", [False, True])
def test_remote_image_draft_updates_open_gui_and_next_save(
    exp_tab_factory, qapp, tmp_path: Path, monkeypatch, kind: ArtifactKind, fails: bool
) -> None:
    monkeypatch.setattr(
        FakeAdapter,
        "capabilities",
        replace(FakeAdapter.capabilities, post_analysis=True),
    )
    window, state, tabs, save, tab_id, ctrl = _live_save_ui(tmp_path)
    try:
        state.update_tab_analyze(tab_id, object(), Figure())
        state.update_tab_post_analyze(tab_id, object(), Figure())
        window.refresh_tab_interaction(tab_id)
        center = window.findChild(ArtifactSaveCenter)
        assert center is not None
        before = (
            center.get_analysis_path()
            if kind is ArtifactKind.ANALYSIS
            else center.get_post_analysis_path()
        )
        path = str(tmp_path / "remote.png")
        assert before != path
        params = {"tab_id": tab_id, "subtab_id": kind.value, "image_path": path}
        export = (
            save.save_image_sync
            if kind is ArtifactKind.ANALYSIS
            else save.save_post_image_sync
        )
        if fails:
            export.side_effect = OSError("export failed")
            with pytest.raises(OSError, match="export failed"):
                h_tab_save_image(ctrl, params)
            export.side_effect = None
        else:
            assert h_tab_save_image(ctrl, params) == {"image_path": path}
        # No manual UI refresh between the remote command and the GUI action.
        assert (
            center.get_analysis_path()
            if kind is ArtifactKind.ANALYSIS
            else center.get_post_analysis_path()
        ) == path
        button = center.save_button(kind)
        assert button.isEnabled()
        button.click()
        assert export.call_count == 2
        assert [call.args[1] for call in export.call_args_list] == [path, path]
        projected = next(
            a for a in tabs.get_snapshot(tab_id).artifacts if a.kind is kind
        )
        assert projected.default_path == path
    finally:
        window.deleteLater()
        qapp.processEvents()


def test_live_comment_typing_preserves_undo_through_state_projection(
    exp_tab_factory, qapp, tmp_path: Path
) -> None:
    from qtpy.QtCore import QEvent, Qt
    from qtpy.QtGui import QKeyEvent

    window, state, tabs, _save, tab_id, _ctrl = _live_save_ui(tmp_path)
    try:
        state.update_tab_comment(tab_id, "original")
        assert tabs.get_snapshot(tab_id).artifacts[0].status is SaveStatus.NOT_SAVED
        state.get_tab(tab_id).artifacts.started(ArtifactKind.DATA)
        default = tabs.get_tab_data_path(tab_id)
        assert default is not None
        state.get_tab(tab_id).artifacts.succeeded(ArtifactKind.DATA, default)
        assert tabs.get_snapshot(tab_id).artifacts[0].status is SaveStatus.SAVED
        window.refresh_tab_interaction(tab_id)
        center = window.findChild(ArtifactSaveCenter)
        assert center is not None
        editor = center.findChild(QTextEdit)
        assert editor is not None
        editor.setFocus()
        cursor = editor.textCursor()
        cursor.setPosition(4)
        editor.setTextCursor(cursor)
        for event_type in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            qapp.sendEvent(
                editor,
                QKeyEvent(
                    event_type,
                    Qt.Key.Key_Exclam,
                    Qt.KeyboardModifier.NoModifier,
                    "!",
                ),
            )
        qapp.processEvents()
        assert editor.toPlainText() == "orig!inal"
        assert state.get_tab(tab_id).save.comment == "orig!inal"
        assert (
            tabs.get_snapshot(tab_id).artifacts[0].status is SaveStatus.UNSAVED_CHANGES
        )
        assert editor.textCursor().position() == 5
        document = editor.document()
        assert document is not None and document.isUndoAvailable()
        window.refresh_tab_interaction(tab_id)
        assert editor.textCursor().position() == 5
        editor.undo()
        qapp.processEvents()
        assert state.get_tab(tab_id).save.comment == "original"
        assert tabs.get_snapshot(tab_id).artifacts[0].status is SaveStatus.SAVED
        state.update_tab_comment(tab_id, "external edit")
        window.refresh_tab_interaction(tab_id)
        assert editor.toPlainText() == "external edit"
    finally:
        window.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("button", ["single", "all"])
def test_cleared_gui_data_path_saves_to_state_default(
    exp_tab_factory, qapp, tmp_path: Path, button: str
) -> None:
    window, state, tabs, save, tab_id, _ctrl = _live_save_ui(tmp_path)
    try:
        default = tabs.get_tab_data_path(tab_id)
        assert default is not None and default.startswith(str(tmp_path / "database"))
        center = window.findChild(ArtifactSaveCenter)
        assert center is not None
        data_edit = next(
            edit for edit in center.findChildren(QLineEdit) if edit.text() == default
        )
        data_edit.clear()
        qapp.processEvents()
        assert center.get_data_path() == ""
        assert state.get_tab(tab_id).save.data_path_override is None
        target = (
            center.save_button(ArtifactKind.DATA)
            if button == "single"
            else center.save_all_button
        )
        assert target.isEnabled()
        target.click()
        if button == "single":
            assert save.start_save_data.call_args.args[1] == default
        else:
            save.start_save_artifacts.assert_called_once()
            assert save.start_save_artifacts.call_args.args[1][0].path == default
        assert state.get_tab(tab_id).save.data_path_override is None
        state.get_tab(tab_id).artifacts.started(ArtifactKind.DATA)
        state.get_tab(tab_id).artifacts.succeeded(ArtifactKind.DATA, default)
        window.refresh_tab_interaction(tab_id)
        assert center.status_text(ArtifactKind.DATA) == "✓ SAVED"
    finally:
        window.deleteLater()
        qapp.processEvents()


# ---------------------------------------------------------------------------
# Figure gating (correction 2)
# ---------------------------------------------------------------------------


def test_analysis_save_requires_figure(exp_tab_factory):
    ctrl = _mock_ctrl()
    caps = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=False, load_data=False
    )
    # Result present but figure absent -> NOT saveable; data not present to isolate
    snap_no_fig = _snapshot(
        "tab-1",
        has_run=False,
        has_analysis=True,
        analysis_mode=AnalysisMode.FIT,
        post_cap=False,
        load_cap=False,
        analysis_has_figure=False,
    )
    tab = exp_tab_factory("tab-1", ctrl, caps)
    tab.attach(snap_no_fig, MagicMock())
    center = tab._save_center
    # Status still reflects result lifecycle (NOT SAVED), but save disabled
    assert center.status_text(ArtifactKind.ANALYSIS) == "○ NOT SAVED"
    assert center.is_save_enabled(ArtifactKind.ANALYSIS) is False
    assert center.is_save_all_enabled() is False
    # With figure, enabled
    snap_with_fig = _snapshot(
        "tab-1",
        has_run=False,
        has_analysis=True,
        analysis_mode=AnalysisMode.FIT,
        post_cap=False,
        load_cap=False,
        analysis_has_figure=True,
    )
    tab.update_interaction_state(snap_with_fig)
    assert center.is_save_enabled(ArtifactKind.ANALYSIS) is True
    assert center.is_save_all_enabled() is True
    # When data also present, Save All remains enabled even if analysis figure missing, but analysis save stays disabled
    snap_mixed = _snapshot(
        "tab-1",
        has_run=True,
        has_analysis=True,
        analysis_mode=AnalysisMode.FIT,
        post_cap=False,
        load_cap=False,
        analysis_has_figure=False,
    )
    tab.update_interaction_state(snap_mixed)
    assert center.is_save_enabled(ArtifactKind.ANALYSIS) is False
    assert center.is_save_enabled(ArtifactKind.DATA) is True
    assert center.is_save_all_enabled() is True
    tab.deleteLater()
    _require_qapp().processEvents()


def test_individual_image_save_dispatch_requires_figure(qapp):
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = 0
    ctrl.has_agent_connected.return_value = False
    window = MainWindow(ctrl)
    caps = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=False, load_data=False
    )
    snap_no_fig = _snapshot(
        "tab-1",
        has_run=True,
        has_analysis=True,
        analysis_mode=AnalysisMode.FIT,
        post_cap=False,
        load_cap=False,
        has_active_context=True,
        analysis_has_figure=False,
    )
    from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget

    tab_ctrl = _mock_ctrl()
    tab = ExpTabWidget("tab-1", tab_ctrl, caps)
    tab.attach(snap_no_fig, MagicMock())
    ctrl.get_tab_snapshot.return_value = snap_no_fig
    ctrl.has_tab.return_value = True
    window._tab_widgets["tab-1"] = tab
    tab.update_interaction_state(snap_no_fig)
    assert tab._save_center.is_save_enabled(ArtifactKind.ANALYSIS) is False
    btn = tab._save_center.save_button(ArtifactKind.ANALYSIS)
    assert not btn.isEnabled()
    btn.click()
    ctrl.save_image.assert_not_called()
    ctrl.save_data.assert_not_called()

    # ---------------------------------------------------------------------------
    # Monotonic revision regression (correction 1)
    # ---------------------------------------------------------------------------
    window.deleteLater()
    tab.deleteLater()
    qapp.processEvents()


def test_tab_close_query_reads_only_data_artifact(exp_tab_factory):
    tab = exp_tab_factory(
        "tab-1",
        _mock_ctrl(),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )
    tab.attach(_snapshot("tab-1", has_run=False), MagicMock())
    assert tab.has_unsaved_data() is False

    for status, expected in (
        (SaveStatus.NOT_SAVED, True),
        (SaveStatus.UNSAVED_CHANGES, True),
        (SaveStatus.SAVED, False),
    ):
        tab.update_interaction_state(
            _snapshot(
                "tab-1",
                has_run=True,
                has_analysis=True,
                data_status=status,
                analysis_status=SaveStatus.NOT_SAVED,
            )
        )
        assert tab.has_unsaved_data() is expected

    tab.deleteLater()
    _require_qapp().processEvents()
