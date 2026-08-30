"""Focused Data-center behavior tests for TKT-001 save-subtab-redesign.

Validates S1-S3 acceptance via production ExpTabWidget / MainWindow seams.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from qtpy.QtWidgets import QApplication, QLabel, QLineEdit, QPushButton, QTextEdit
from zcu_tools.gui.app.main.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.main.events.completion import SaveDataFinishedPayload
from zcu_tools.gui.app.main.services import PersistedStartup, TabSnapshot
from zcu_tools.gui.app.main.state import TabInteractionState
from zcu_tools.gui.app.main.ui.artifact_save_center import ArtifactKind
from zcu_tools.gui.event_bus import BaseEventBus as EventBus


@dataclass
class _DummyParams:
    x: int = 1


def _require_qapp() -> QApplication:
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    return app


def _mock_ctrl() -> MagicMock:
    ctrl = MagicMock()
    ctrl.get_persisted_startup.return_value = PersistedStartup(left_panel_width=500)
    ctrl.get_tab_adapter_name.return_value = "fake"
    ctrl.get_adapter_guide.return_value = {}
    ctrl.progress_control.attach_progress.return_value = lambda: None
    ctrl.progress_control.progress_bars.return_value = []
    ctrl.active_operation_count.return_value = 0
    ctrl.has_agent_connected.return_value = False
    from zcu_tools.gui.app.main.cfg_binding import MeasureCfgBindings
    from zcu_tools.gui.app.main.specs import make_pulse_spec
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
) -> TabSnapshot:
    from zcu_tools.gui.app.main.services.ports import (
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

    return TabSnapshot(
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


@pytest.fixture
def exp_tab_factory(qapp, monkeypatch):
    import zcu_tools.gui.app.main.ui.exp_tab_widget as mod

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


def test_tall_data_pane_keeps_artifacts_compact_and_actions_at_bottom(
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
    title_bottom = title.mapTo(center, title.rect().bottomLeft()).y()
    path_top = path_edit.mapTo(center, path_edit.rect().topLeft()).y()

    assert title.height() <= title.sizeHint().height() + 4
    assert path_top - title_bottom <= 16

    save_all_bottom = center.save_all_button.mapTo(
        center, center.save_all_button.rect().bottomLeft()
    ).y()
    assert center.rect().bottom() - save_all_bottom <= 16

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


def test_status_transitions_path_comment_and_result(exp_tab_factory):
    ctrl = _mock_ctrl()
    caps = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=False, load_data=False
    )
    tab = exp_tab_factory("tab-1", ctrl, caps)
    snap = _snapshot(
        "tab-1",
        has_run=True,
        has_analysis=True,
        analysis_mode=AnalysisMode.FIT,
        post_cap=False,
        load_cap=False,
    )
    tab.attach(snap, MagicMock())
    center = tab._save_center
    assert center.status_text(ArtifactKind.DATA) == "○ NOT SAVED"
    assert center.status_text(ArtifactKind.ANALYSIS) == "○ NOT SAVED"
    center.notify_save_succeeded(ArtifactKind.ANALYSIS)
    assert center.status_text(ArtifactKind.ANALYSIS) == "✓ SAVED"
    center.set_analysis_path("/tmp/new.png")
    assert center.status_text(ArtifactKind.ANALYSIS) == "● UNSAVED CHANGES"
    center.notify_save_succeeded(ArtifactKind.ANALYSIS)
    assert center.status_text(ArtifactKind.ANALYSIS) == "✓ SAVED"
    center.notify_save_succeeded(ArtifactKind.DATA)
    assert center.status_text(ArtifactKind.DATA) == "✓ SAVED"
    center.set_data_path("/tmp/other.h5")
    assert center.status_text(ArtifactKind.DATA) == "● UNSAVED CHANGES"
    center.notify_save_succeeded(ArtifactKind.DATA)
    assert center.status_text(ArtifactKind.DATA) == "✓ SAVED"
    center.set_comment_text("new comment")
    assert center.status_text(ArtifactKind.DATA) == "● UNSAVED CHANGES"
    snap2 = _snapshot(
        "tab-1",
        has_run=True,
        has_analysis=True,
        analysis_mode=AnalysisMode.FIT,
        post_cap=False,
        load_cap=False,
    )
    tab.update_interaction_state(snap2)
    assert center.status_text(ArtifactKind.DATA) == "● UNSAVED CHANGES"
    tab.deleteLater()
    _require_qapp().processEvents()


def test_status_not_only_color(exp_tab_factory):
    ctrl = _mock_ctrl()
    caps = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=False, load_data=False
    )
    tab = exp_tab_factory("tab-1", ctrl, caps)
    snap = _snapshot(
        "tab-1",
        has_run=True,
        has_analysis=True,
        analysis_mode=AnalysisMode.FIT,
        post_cap=False,
        load_cap=False,
    )
    tab.attach(snap, MagicMock())
    center = tab._save_center
    assert "○" in center.status_text(ArtifactKind.DATA)
    center.notify_save_succeeded(ArtifactKind.DATA)
    assert "✓" in center.status_text(ArtifactKind.DATA)
    center.set_data_path("/tmp/changed.h5")
    assert "●" in center.status_text(ArtifactKind.DATA)
    snap_none = _snapshot(
        "tab-1",
        has_run=False,
        has_analysis=False,
        analysis_mode=AnalysisMode.FIT,
        post_cap=False,
        load_cap=False,
    )
    tab.update_interaction_state(snap_none)
    assert "—" in center.status_text(ArtifactKind.DATA)
    tab.deleteLater()
    _require_qapp().processEvents()


def test_individual_image_save_success_and_failure(exp_tab_factory):
    ctrl = _mock_ctrl()
    caps = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=False, load_data=False
    )
    tab = exp_tab_factory("tab-1", ctrl, caps)
    snap = _snapshot(
        "tab-1",
        has_run=True,
        has_analysis=True,
        analysis_mode=AnalysisMode.FIT,
        post_cap=False,
        load_cap=False,
    )
    tab.attach(snap, MagicMock())
    center = tab._save_center
    center.notify_save_succeeded(ArtifactKind.ANALYSIS)
    assert center.status_text(ArtifactKind.ANALYSIS) == "✓ SAVED"
    center.notify_save_failed(ArtifactKind.ANALYSIS)
    assert center.status_text(ArtifactKind.ANALYSIS) == "✓ SAVED"
    center.set_analysis_path("/tmp/unsaved.png")
    assert center.status_text(ArtifactKind.ANALYSIS) == "● UNSAVED CHANGES"
    center.notify_save_failed(ArtifactKind.ANALYSIS)
    assert center.status_text(ArtifactKind.ANALYSIS) == "● UNSAVED CHANGES"
    tab2 = exp_tab_factory("tab-2", ctrl, caps)
    snap2 = _snapshot(
        "tab-2",
        has_run=True,
        has_analysis=True,
        analysis_mode=AnalysisMode.FIT,
        post_cap=False,
        load_cap=False,
    )
    tab2.attach(snap2, MagicMock())
    c2 = tab2._save_center
    assert c2.status_text(ArtifactKind.ANALYSIS) == "○ NOT SAVED"
    c2.notify_save_failed(ArtifactKind.ANALYSIS)
    assert c2.status_text(ArtifactKind.ANALYSIS) == "○ NOT SAVED"
    tab.deleteLater()
    tab2.deleteLater()
    _require_qapp().processEvents()


def test_data_async_success_failure_and_drift(exp_tab_factory):
    ctrl = _mock_ctrl()
    caps = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=False, load_data=False
    )
    tab = exp_tab_factory("tab-1", ctrl, caps)
    snap = _snapshot(
        "tab-1",
        has_run=True,
        has_analysis=False,
        analysis_mode=AnalysisMode.FIT,
        post_cap=False,
        load_cap=False,
    )
    tab.attach(snap, MagicMock())
    center = tab._save_center
    assert center.status_text(ArtifactKind.DATA) == "○ NOT SAVED"
    center.notify_save_started(ArtifactKind.DATA)
    assert center.status_text(ArtifactKind.DATA) == "○ NOT SAVED"
    center.handle_data_finished(None)
    assert center.status_text(ArtifactKind.DATA) == "✓ SAVED"
    center.notify_save_started(ArtifactKind.DATA)
    center.set_data_path("/tmp/drift.h5")
    assert center.status_text(ArtifactKind.DATA) == "● UNSAVED CHANGES"
    center.handle_data_finished(None)
    assert center.status_text(ArtifactKind.DATA) == "● UNSAVED CHANGES"
    tab2 = exp_tab_factory("tab-2", ctrl, caps)
    tab2.attach(snap, MagicMock())
    c2 = tab2._save_center
    c2.notify_save_started(ArtifactKind.DATA)
    c2.handle_data_finished("disk full")
    assert c2.status_text(ArtifactKind.DATA) == "○ NOT SAVED"

    # ---------------------------------------------------------------------------
    # A4 Save All ordering / Fast Fail
    # ---------------------------------------------------------------------------
    tab.deleteLater()
    tab2.deleteLater()
    _require_qapp().processEvents()


def test_save_all_dispatch_only_result_present_and_order(qapp, monkeypatch):
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = 0
    ctrl.has_agent_connected.return_value = False
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
    from zcu_tools.gui.app.main.ui.exp_tab_widget import ExpTabWidget

    tab_ctrl = _mock_ctrl()
    tab = ExpTabWidget("tab-1", tab_ctrl, caps)
    tab.attach(snap, MagicMock())
    ctrl.get_tab_snapshot.return_value = snap
    ctrl.has_tab.return_value = True
    window._tab_widgets["tab-1"] = tab

    snap_data_only = _snapshot(
        "tab-1",
        has_run=True,
        has_analysis=False,
        has_post=False,
        analysis_mode=AnalysisMode.FIT,
        post_cap=True,
        load_cap=True,
        has_active_context=True,
    )
    ctrl.get_tab_snapshot.return_value = snap_data_only
    tab.update_interaction_state(snap_data_only)
    window._on_save_all_clicked("tab-1")
    ctrl.save_image.assert_not_called()
    ctrl.save_post_image.assert_not_called()
    assert ctrl.save_data.call_count == 1
    ctrl.save_data.reset_mock()
    ctrl.save_image.reset_mock()
    ctrl.save_post_image.reset_mock()

    ctrl.get_tab_snapshot.return_value = snap
    tab.update_interaction_state(snap)
    call_order: list[str] = []

    def fake_save_image(tab_id, path):
        call_order.append("analysis")
        return "/tmp/a.png"

    def fake_save_post(tab_id, path):
        call_order.append("post")
        return "/tmp/p.png"

    def fake_save_data(tab_id, path, comment=""):
        call_order.append("data")
        return "/tmp/data.h5"

    ctrl.save_image.side_effect = fake_save_image
    ctrl.save_post_image.side_effect = fake_save_post
    ctrl.save_data.side_effect = fake_save_data
    window._on_save_all_clicked("tab-1")
    assert call_order == ["analysis", "post", "data"]
    window.deleteLater()
    tab.deleteLater()
    qapp.processEvents()


def test_save_all_fast_fail_and_no_rollback(qapp, monkeypatch):
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = 0
    ctrl.has_agent_connected.return_value = False
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
    from zcu_tools.gui.app.main.ui.exp_tab_widget import ExpTabWidget

    tab_ctrl = _mock_ctrl()
    tab = ExpTabWidget("tab-1", tab_ctrl, caps)
    tab.attach(snap, MagicMock())
    ctrl.get_tab_snapshot.return_value = snap
    ctrl.has_tab.return_value = True
    window._tab_widgets["tab-1"] = tab
    ctrl.save_image = MagicMock(return_value="/tmp/a.png")
    ctrl.save_post_image = MagicMock(side_effect=OSError("disk full"))
    ctrl.save_data = MagicMock(return_value="/tmp/d.h5")
    assert tab._save_center.status_text(ArtifactKind.ANALYSIS) == "○ NOT SAVED"
    window._on_save_all_clicked("tab-1")
    assert tab._save_center.status_text(ArtifactKind.ANALYSIS) == "✓ SAVED"
    assert tab._save_center.status_text(ArtifactKind.POST_ANALYSIS) == "○ NOT SAVED"
    assert tab._save_center.status_text(ArtifactKind.DATA) == "○ NOT SAVED"
    assert ctrl.save_data.call_count == 0

    ctrl.save_post_image.side_effect = None
    ctrl.save_post_image.return_value = "/tmp/p.png"
    ctrl.save_data.return_value = "/tmp/d.h5"
    window._on_save_all_clicked("tab-1")
    assert tab._save_center.status_text(ArtifactKind.DATA) == "○ NOT SAVED"
    payload_fail = SaveDataFinishedPayload(
        tab_id="tab-1", data_path="/tmp/d.h5", error="fail"
    )
    window.handle_save_data_finished(payload_fail)
    assert tab._save_center.status_text(ArtifactKind.ANALYSIS) == "✓ SAVED"
    assert tab._save_center.status_text(ArtifactKind.POST_ANALYSIS) == "✓ SAVED"
    assert tab._save_center.status_text(ArtifactKind.DATA) == "○ NOT SAVED"
    window.deleteLater()
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

    # ---------------------------------------------------------------------------
    # A5 Load Data gates
    # ---------------------------------------------------------------------------
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


def test_unmatched_remote_save_completion_does_not_mark_gui_sig_as_saved(
    exp_tab_factory, qapp
):
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = 0
    ctrl.has_agent_connected.return_value = False
    window = MainWindow(ctrl)

    caps = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=False, load_data=False
    )
    snap = _snapshot(
        "tab-1",
        has_run=True,
        has_analysis=False,
        analysis_mode=AnalysisMode.FIT,
        post_cap=False,
        load_cap=False,
        has_active_context=True,
        data_path="/gui.h5",
    )
    tab_ctrl = _mock_ctrl()
    tab = exp_tab_factory("tab-1", tab_ctrl, caps)
    tab.attach(snap, MagicMock())
    tab._save_center.set_data_path("/gui.h5")
    tab._save_center.set_comment_text("gui")
    tab.update_interaction_state(snap)
    center = tab._save_center
    assert center.status_text(ArtifactKind.DATA) == "○ NOT SAVED"
    assert center.get_data_path() == "/gui.h5"
    assert center.get_comment() == "gui"

    ctrl.get_tab_snapshot.return_value = snap
    ctrl.has_tab.return_value = True
    window._tab_widgets["tab-1"] = tab

    payload = SaveDataFinishedPayload(
        tab_id="tab-1", data_path="/remote.h5", error=None
    )
    window.handle_save_data_finished(payload)

    assert center.status_text(ArtifactKind.DATA) == "○ NOT SAVED"
    center.notify_save_started(ArtifactKind.DATA)
    center.handle_data_finished(None)
    assert center.status_text(ArtifactKind.DATA) == "✓ SAVED"


def test_comment_edit_does_not_trigger_data_path_update(exp_tab_factory, qapp):
    ctrl = _mock_ctrl()
    # Make update_tab_data_path observable
    ctrl.update_tab_data_path = MagicMock()
    ctrl.update_tab_analysis_image_path = MagicMock()
    ctrl.update_tab_post_analysis_image_path = MagicMock()
    caps = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=False, load_data=False
    )
    snap = _snapshot(
        "tab-1",
        has_run=True,
        has_analysis=False,
        analysis_mode=AnalysisMode.FIT,
        post_cap=False,
        load_cap=False,
        has_active_context=True,
    )
    tab = exp_tab_factory("tab-1", ctrl, caps)
    tab.attach(snap, MagicMock())
    # Clear any calls from attach (path sync)
    ctrl.update_tab_data_path.reset_mock()
    center = tab._save_center
    assert center.status_text(ArtifactKind.DATA) == "○ NOT SAVED"
    center.set_comment_text("new comment")
    assert center.status_text(ArtifactKind.DATA) == "○ NOT SAVED"
    ctrl.update_tab_data_path.assert_not_called()
    ctrl.update_tab_analysis_image_path.assert_not_called()
    ctrl.update_tab_post_analysis_image_path.assert_not_called()
    ctrl.update_tab_data_path.reset_mock()
    center._comment_edit.setPlainText("another")
    _require_qapp().processEvents()
    assert center.status_text(ArtifactKind.DATA) == "○ NOT SAVED"
    ctrl.update_tab_data_path.assert_not_called()
    tab.deleteLater()
    _require_qapp().processEvents()


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


def test_save_all_skips_analysis_without_figure(qapp, monkeypatch):
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = 0
    ctrl.has_agent_connected.return_value = False
    ctrl.save_data = MagicMock(return_value="/tmp/d.h5")
    ctrl.save_image = MagicMock(return_value="/tmp/a.png")
    ctrl.save_post_image = MagicMock(return_value="/tmp/p.png")
    window = MainWindow(ctrl)
    caps = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=False, load_data=False
    )
    # Snapshot: analysis result true but figure None -> not saveable
    snap = _snapshot(
        "tab-1",
        has_run=True,
        has_analysis=True,
        has_post=False,
        analysis_mode=AnalysisMode.FIT,
        post_cap=False,
        load_cap=False,
        has_active_context=True,
        analysis_has_figure=False,
    )
    from zcu_tools.gui.app.main.ui.exp_tab_widget import ExpTabWidget

    tab_ctrl = _mock_ctrl()
    tab = ExpTabWidget("tab-1", tab_ctrl, caps)
    tab.attach(snap, MagicMock())
    ctrl.get_tab_snapshot.return_value = snap
    ctrl.has_tab.return_value = True
    window._tab_widgets["tab-1"] = tab
    tab.update_interaction_state(snap)
    window._on_save_all_clicked("tab-1")
    # Analysis should be skipped (no figure), only data dispatched
    ctrl.save_image.assert_not_called()
    assert ctrl.save_data.call_count == 1
    window.deleteLater()
    tab.deleteLater()
    qapp.processEvents()


def test_individual_image_save_dispatch_requires_figure(qapp):
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

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
    from zcu_tools.gui.app.main.ui.exp_tab_widget import ExpTabWidget

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


def test_replaced_result_invalidates_saved_via_monotonic_token(exp_tab_factory):
    ctrl = _mock_ctrl()
    caps = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=False, load_data=False
    )
    tab = exp_tab_factory("tab-1", ctrl, caps)
    snap1 = _snapshot(
        "tab-1",
        has_run=True,
        has_analysis=False,
        analysis_mode=AnalysisMode.FIT,
        post_cap=False,
        load_cap=False,
    )
    tab.attach(snap1, MagicMock())
    center = tab._save_center
    center.notify_save_succeeded(ArtifactKind.DATA)
    assert center.status_text(ArtifactKind.DATA) == "✓ SAVED"
    # New result object with same path/comment but different identity should invalidate SAVED
    snap2 = _snapshot(
        "tab-1",
        has_run=True,
        has_analysis=False,
        analysis_mode=AnalysisMode.FIT,
        post_cap=False,
        load_cap=False,
    )
    assert snap1.run is not None
    assert snap2.run is not None
    assert snap1.run.result is not snap2.run.result
    tab.update_interaction_state(snap2)
    assert center.status_text(ArtifactKind.DATA) == "● UNSAVED CHANGES"
    tab2 = exp_tab_factory("tab-2", ctrl, caps)
    snap_a = _snapshot(
        "tab-2",
        has_run=True,
        analysis_mode=AnalysisMode.FIT,
        post_cap=False,
        load_cap=False,
    )
    tab2.attach(snap_a, MagicMock())
    c2 = tab2._save_center
    c2.notify_save_succeeded(ArtifactKind.DATA)
    assert c2.status_text(ArtifactKind.DATA) == "✓ SAVED"
    # Updating with same snapshot (same object identity) should keep SAVED
    # We need to keep same result object; create snapshot that reuses same object
    assert snap_a.run is not None
    same_obj = snap_a.run.result
    # Build snapshot manually reusing same_obj
    from zcu_tools.gui.app.main.services.ports import (
        AnalysisPaneSnapshot,
        PathResourceSnapshot,
        PostAnalysisPaneSnapshot,
        RunPaneSnapshot,
        SavePaneSnapshot,
        TabPathsSnapshot,
    )

    snap_same = TabSnapshot(
        adapter_name="fake",
        cfg_schema=MagicMock(),
        tab_id="tab-2",
        interaction=snap_a.interaction,
        capabilities=snap_a.capabilities,
        run=RunPaneSnapshot(result=same_obj, source_path=None),
        analysis=snap_a.analysis,
        post_analysis=snap_a.post_analysis,
        save=snap_a.save,
        paths=snap_a.paths,
    )
    tab2.update_interaction_state(snap_same)
    assert c2.status_text(ArtifactKind.DATA) == "✓ SAVED"
    tab.deleteLater()
    tab2.deleteLater()
    _require_qapp().processEvents()
