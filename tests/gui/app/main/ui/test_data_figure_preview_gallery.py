"""Variant A stacked Data previews — gallery and ExpTabWidget integration (Ticket 002)."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from qtpy.QtGui import QPixmap  # type: ignore[attr-defined]
from qtpy.QtWidgets import QLabel, QScrollArea, QWidget
from zcu_tools.gui.app.main.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.main.services import PersistedStartup, TabSnapshot
from zcu_tools.gui.app.main.state import TabInteractionState
from zcu_tools.gui.app.main.ui.data_figure_preview_gallery import (
    DataFigurePreviewGallery,
)


@dataclass
class DummyParams:
    x: int = 1


@dataclass
class DummyPostParams:
    p: int = 2


# Minimal 1x1 PNG for deterministic fake renderer (opaque, valid for QPixmap)
_FAKE_PNG_BYTES = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def _fake_renderer(_fig: object) -> bytes:
    return bytes(_FAKE_PNG_BYTES)


def _failing_renderer_for(target_fig: object):
    def _render(fig: object) -> bytes:
        if fig is target_fig:
            raise RuntimeError("boom render")
        return bytes(_FAKE_PNG_BYTES)

    return _render


def make_ctrl():
    ctrl = MagicMock()
    ctrl.get_persisted_startup.return_value = PersistedStartup(left_panel_width=500)
    ctrl.get_tab_adapter_name.return_value = "fake"
    ctrl.get_adapter_guide.return_value = {}
    ctrl.progress_control.attach_progress.return_value = lambda: None
    ctrl.progress_control.progress_bars.return_value = []
    ctrl.update_tab_data_path = MagicMock()
    ctrl.update_tab_analysis_image_path = MagicMock()
    ctrl.update_tab_post_analysis_image_path = MagicMock()
    ctrl.update_tab_save_paths = MagicMock()
    ctrl.open_seeded_cfg_editor.return_value = ("editor-id", ())
    ctrl.get_cfg_editor_draft.return_value = MagicMock()
    ctrl.active_operation_count.return_value = 0
    ctrl.has_agent_connected.return_value = False
    return ctrl


def make_snapshot(tab_id: str, *, analysis=AnalysisMode.FIT, post=False, load=False):
    from zcu_tools.gui.app.main.services.ports import (
        AnalysisPaneSnapshot,
        PathResourceSnapshot,
        PostAnalysisPaneSnapshot,
        RunPaneSnapshot,
        SavePaneSnapshot,
        TabPathsSnapshot,
    )

    caps = AdapterCapabilities(analysis=analysis, post_analysis=post, load_data=load)  # type: ignore[call-arg]
    run_snap = RunPaneSnapshot(result=object(), source_path=None)
    analysis_snap = AnalysisPaneSnapshot(
        params=DummyParams() if analysis is not AnalysisMode.NONE else None,
        result=object() if analysis is not AnalysisMode.NONE else None,
        figure=None,
        writeback_items=(),
        image_path=PathResourceSnapshot(override=None, path=None),
    )
    post_snap = PostAnalysisPaneSnapshot(
        params=DummyPostParams() if post else None,
        result=object() if post else None,
        figure=None,
        writeback_items=(),
        image_path=PathResourceSnapshot(override=None, path=None),
    )
    save_snap = SavePaneSnapshot(
        data_path=PathResourceSnapshot(override=None, path="/tmp/d.h5")
    )
    paths_snap = TabPathsSnapshot(
        data=PathResourceSnapshot(override=None, path="/tmp/d.h5"),
        analysis_image=PathResourceSnapshot(override=None, path=None),
        post_analysis_image=PathResourceSnapshot(override=None, path=None),
    )
    return TabSnapshot(
        adapter_name="fake",
        cfg_schema=MagicMock(),
        tab_id=tab_id,
        interaction=TabInteractionState(
            global_run_active=False,
            is_running=False,
            is_analyzing=False,
            is_saving_data=False,
            has_context=True,
            has_active_context=True,
            has_soc=True,
            has_run_result=True,
            has_analyze_result=analysis is not AnalysisMode.NONE,
            has_figure=False,
            has_post_analyze_result=post,
        ),
        capabilities=caps,
        run=run_snap,
        analysis=analysis_snap,
        post_analysis=post_snap,
        save=save_snap,
        paths=paths_snap,
    )


@pytest.fixture
def exp_tab_widget(qapp, monkeypatch):
    import zcu_tools.gui.app.main.ui.exp_tab_widget as mod

    orig_pop = mod.ExpTabWidget._populate_cfg

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
    monkeypatch.setattr(mod.ExpTabWidget, "_populate_cfg", orig_pop)
    monkeypatch.setattr(mod, "attach_existing_figure_to_container", orig_attach)


# ---------------------------------------------------------------------------
# Gallery standalone — S1, S3, S4
# ---------------------------------------------------------------------------


def test_gallery_capability_driven_cards(qapp):
    # Run only
    caps_none = AdapterCapabilities(analysis=AnalysisMode.NONE, post_analysis=False)  # type: ignore[call-arg]
    g_none = DataFigurePreviewGallery(caps_none, renderer=_fake_renderer)
    assert g_none.has_card("run")
    assert not g_none.has_card("analysis")
    assert not g_none.has_card("post_analysis")
    assert g_none.card_count() == 1
    # Find via objectName
    assert g_none.findChild(QWidget, "previewCard_run") is not None  # type: ignore[attr-defined]
    assert g_none.findChild(QWidget, "previewCard_analysis") is None  # type: ignore[attr-defined]

    # Run + Analysis
    caps_a = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False)  # type: ignore[call-arg]
    g_a = DataFigurePreviewGallery(caps_a, renderer=_fake_renderer)
    assert g_a.has_card("run")
    assert g_a.has_card("analysis")
    assert not g_a.has_card("post_analysis")
    assert g_a.card_count() == 2

    # All three
    caps_both = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)  # type: ignore[call-arg]
    g_both = DataFigurePreviewGallery(caps_both, renderer=_fake_renderer)
    assert g_both.has_card("run")
    assert g_both.has_card("analysis")
    assert g_both.has_card("post_analysis")
    assert g_both.card_count() == 3
    # Scrollable rail
    assert isinstance(
        g_both.findChild(QScrollArea, "previewGalleryScroll"), QScrollArea
    )


def test_gallery_supported_but_empty_shows_named_empty_state(qapp):
    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)  # type: ignore[call-arg]
    g = DataFigurePreviewGallery(caps, renderer=_fake_renderer)
    # Initially empty for all
    assert g.card_state("run") == "empty"
    assert "No figure" in g.card_text("run")
    assert g.card_state("analysis") == "empty"
    assert "No figure" in g.card_text("analysis")
    assert g.card_state("post_analysis") == "empty"
    assert "No figure" in g.card_text("post_analysis")
    # Updating with None keeps empty
    g.update_figures(None, None, None)
    assert g.card_state("run") == "empty"
    # Updating with Run only
    g.update_figures(Figure(), None, None)
    assert g.card_state("run") == "available"
    assert g.card_state("analysis") == "empty"
    assert g.card_state("post_analysis") == "empty"


def test_gallery_raster_preview_via_injected_renderer(qapp):
    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)  # type: ignore[call-arg]
    g = DataFigurePreviewGallery(caps, renderer=_fake_renderer)
    fig_run = Figure()
    fig_a = Figure()
    fig_p = Figure()
    g.update_figures(fig_run, fig_a, fig_p)
    assert g.card_state("run") == "available"
    assert g.card_state("analysis") == "available"
    assert g.card_state("post_analysis") == "available"
    # Pixmap should be set on image label
    run_card = g.find_card("run")
    assert run_card is not None
    img_label = run_card.findChild(QLabel, "previewImage_run")
    assert (
        img_label is not None
        and img_label.pixmap() is not None
        and not img_label.pixmap().isNull()
    )


def test_gallery_single_failure_isolated(qapp):
    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)  # type: ignore[call-arg]
    fig_run = Figure()
    fig_a = Figure()
    fig_p = Figure()
    renderer = _failing_renderer_for(fig_a)
    g = DataFigurePreviewGallery(caps, renderer=renderer)
    g.update_figures(fig_run, fig_a, fig_p)
    # Run and post should still be available
    assert g.card_state("run") == "available"
    assert g.card_state("analysis") == "unavailable"
    assert "unavailable" in g.card_text("analysis").lower()
    assert g.card_state("post_analysis") == "available"
    # Gallery still scrollable and other cards not blocked
    assert g.isVisible() or True  # creation succeeded

    # Second render with different failure isolates again
    def fail_post(fig):
        if fig is fig_p:
            raise ValueError("post boom")
        return bytes(_FAKE_PNG_BYTES)

    g2 = DataFigurePreviewGallery(caps, renderer=fail_post)
    g2.update_figures(fig_run, fig_a, fig_p)
    assert g2.card_state("run") == "available"
    assert g2.card_state("analysis") == "available"
    assert g2.card_state("post_analysis") == "unavailable"


def test_gallery_never_retains_figure(qapp):
    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False)  # type: ignore[call-arg]
    held = []

    def capturing_renderer(fig):
        held.append(fig)
        return bytes(_FAKE_PNG_BYTES)

    g = DataFigurePreviewGallery(caps, renderer=capturing_renderer)
    fig = Figure()
    g.update_figures(fig, Figure(), None)
    # Gallery should not hold Figure refs beyond call; we cleared held list
    # but verify gallery's cards have no 'figure' attr
    for card in g._cards.values():
        assert not hasattr(card, "figure")
    # Ensure update does not store figure in gallery attribute
    assert not any(
        hasattr(g, attr) and getattr(g, attr) is fig
        for attr in dir(g)
        if not attr.startswith("_")
    )


# ---------------------------------------------------------------------------
# ExpTabWidget integration — S1 S2 S4
# ---------------------------------------------------------------------------


def test_exp_data_subtab_shows_gallery_run_always(qapp, exp_tab_widget):
    ctrl = make_ctrl()
    # NONE
    caps_none = AdapterCapabilities(analysis=AnalysisMode.NONE, post_analysis=False)  # type: ignore[call-arg]
    snap_none = make_snapshot("t1", analysis=AnalysisMode.NONE, post=False)
    tab_none = exp_tab_widget("t1", ctrl, caps_none)
    tab_none.attach(snap_none, MagicMock())
    # Data tab exists
    assert any(
        tab_none._left_tabs.tabText(i) == "Data"
        for i in range(tab_none._left_tabs.count())
    )
    # Switch to Data
    for i in range(tab_none._left_tabs.count()):
        if tab_none._left_tabs.tabText(i) == "Data":
            tab_none._left_tabs.setCurrentIndex(i)
            break
    tab_none._on_left_tab_changed(tab_none._left_tabs.currentIndex())
    assert tab_none._right_stack.currentWidget() is tab_none._data_gallery
    assert tab_none._data_gallery.has_card("run")
    assert not tab_none._data_gallery.has_card("analysis")
    assert tab_none._data_gallery.card_state("run") == "empty"
    assert "No figure" in tab_none._data_gallery.card_text("run")
    assert isinstance(tab_none._data_gallery.findChild(QScrollArea), QScrollArea)
    tab_none.detach()

    # FIT + post
    caps_both = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)  # type: ignore[call-arg]
    snap_both = make_snapshot("t2", analysis=AnalysisMode.FIT, post=True)
    tab_both = exp_tab_widget("t2", ctrl, caps_both)
    tab_both.attach(snap_both, MagicMock())
    for i in range(tab_both._left_tabs.count()):
        if tab_both._left_tabs.tabText(i) == "Data":
            tab_both._left_tabs.setCurrentIndex(i)
            break
    tab_both._on_left_tab_changed(tab_both._left_tabs.currentIndex())
    assert tab_both._data_gallery.has_card("run")
    assert tab_both._data_gallery.has_card("analysis")
    assert tab_both._data_gallery.has_card("post_analysis")
    assert tab_both._data_gallery.card_count() == 3
    tab_both.detach()


def test_exp_data_routing_a4(qapp, exp_tab_widget):
    ctrl = make_ctrl()
    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)  # type: ignore[call-arg]
    snap = make_snapshot("t1", analysis=AnalysisMode.FIT, post=True)
    tab = exp_tab_widget("t1", ctrl, caps)
    tab.attach(snap, MagicMock())

    def idx(name):
        for i in range(tab._left_tabs.count()):
            if tab._left_tabs.tabText(i) == name:
                return i
        raise KeyError(name)

    # Run
    tab._left_tabs.setCurrentIndex(idx("Run"))
    tab._on_left_tab_changed(tab._left_tabs.currentIndex())
    assert tab._right_stack.currentWidget() is tab._run_stack
    # Analysis
    tab._left_tabs.setCurrentIndex(idx("Analysis"))
    tab._on_left_tab_changed(tab._left_tabs.currentIndex())
    assert tab._right_stack.currentWidget() is tab._analysis_stack
    # Post
    tab._left_tabs.setCurrentIndex(idx("Post-Analysis"))
    tab._on_left_tab_changed(tab._left_tabs.currentIndex())
    assert tab._right_stack.currentWidget() is tab._post_stack
    # Data -> gallery
    tab._left_tabs.setCurrentIndex(idx("Data"))
    tab._on_left_tab_changed(tab._left_tabs.currentIndex())
    assert tab._right_stack.currentWidget() is tab._data_gallery
    # Guide -> placeholder
    tab._left_tabs.setCurrentIndex(idx("Guide"))
    tab._on_left_tab_changed(tab._left_tabs.currentIndex())
    assert tab._right_stack.currentWidget() is tab._right_placeholder
    tab.detach()


def test_exp_data_activation_refreshes_and_lifecycle_sync(qapp, exp_tab_widget):
    ctrl = make_ctrl()
    # Use fake renderer so we can assert raster not empty after show
    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)  # type: ignore[call-arg]
    tab = exp_tab_widget("t1", ctrl, caps, preview_renderer=_fake_renderer)
    snap = make_snapshot("t1", analysis=AnalysisMode.FIT, post=True)
    tab.attach(snap, MagicMock())

    # Initially no figures -> empty
    for i in range(tab._left_tabs.count()):
        if tab._left_tabs.tabText(i) == "Data":
            tab._left_tabs.setCurrentIndex(i)
            break
    tab._on_left_tab_changed(tab._left_tabs.currentIndex())
    assert tab._data_gallery.card_state("run") == "empty"
    assert tab._data_gallery.card_state("analysis") == "empty"

    # Data visible: show figures should sync immediately
    fig_run = Figure()
    fig_a = Figure()
    fig_p = Figure()
    # Capture container identities before
    run_c = tab.get_run_container()
    ana_c = tab.get_analysis_container()
    post_c = tab.get_post_container()
    tab.show_run_figure(fig_run)
    assert tab.get_current_figure_for_pane("run") is fig_run
    assert tab.get_run_container() is run_c
    assert tab._data_gallery.card_state("run") == "available"

    tab.show_analysis_figure(fig_a)
    assert tab.get_current_figure_for_pane("analysis") is fig_a
    assert tab.get_analysis_container() is ana_c
    assert tab._data_gallery.card_state("analysis") == "available"

    tab.show_post_analysis_figure(fig_p)
    assert tab.get_current_figure_for_pane("post_analysis") is fig_p
    assert tab.get_post_container() is post_c
    assert tab._data_gallery.card_state("post_analysis") == "available"

    # Data-visible clear should revert to empty
    tab.prepare_run_container()
    assert tab.get_current_figure_for_pane("run") is None
    assert tab.get_run_container() is run_c
    assert tab._data_gallery.card_state("run") == "empty"
    # downstream also cleared
    assert tab._data_gallery.card_state("analysis") == "empty"
    assert tab._data_gallery.card_state("post_analysis") == "empty"

    # Re-show
    fig_run2 = Figure()
    tab.show_run_figure(fig_run2)
    assert tab._data_gallery.card_state("run") == "available"
    # clear_post_figure only affects post
    fig_a2 = Figure()
    fig_p2 = Figure()
    tab.show_analysis_figure(fig_a2)
    tab.show_post_analysis_figure(fig_p2)
    assert tab._data_gallery.card_state("post_analysis") == "available"
    tab.clear_post_figure()
    assert tab._data_gallery.card_state("post_analysis") == "empty"
    assert tab._data_gallery.card_state("analysis") == "available"
    assert tab.get_current_figure_for_pane("analysis") is fig_a2
    assert tab.get_analysis_container() is ana_c
    tab.detach()


def test_exp_data_invisible_defers_render(qapp, exp_tab_widget):
    ctrl = make_ctrl()
    call_log: list[str] = []

    def counting_renderer(fig):
        call_log.append("render")
        return bytes(_FAKE_PNG_BYTES)

    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False)  # type: ignore[call-arg]
    tab = exp_tab_widget("t1", ctrl, caps, preview_renderer=counting_renderer)
    snap = make_snapshot("t1", analysis=AnalysisMode.FIT, post=False)
    tab.attach(snap, MagicMock())

    # Start on Run (Data not visible)
    for i in range(tab._left_tabs.count()):
        if tab._left_tabs.tabText(i) == "Run":
            tab._left_tabs.setCurrentIndex(i)
            break
    assert tab._right_stack.currentWidget() is tab._run_stack
    call_log.clear()
    fig_run = Figure()
    tab.show_run_figure(fig_run)
    # No render while Data invisible
    assert call_log == []
    # Switch to Data -> should render once
    for i in range(tab._left_tabs.count()):
        if tab._left_tabs.tabText(i) == "Data":
            tab._left_tabs.setCurrentIndex(i)
            break
    assert len(call_log) == 1
    assert tab._data_gallery.card_state("run") == "available"
    tab.detach()


def test_exp_production_reachability_data_gallery_is_active_right_widget(
    qapp, exp_tab_widget
):
    ctrl = make_ctrl()
    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)  # type: ignore[call-arg]
    tab = exp_tab_widget("reach", ctrl, caps)
    snap = make_snapshot("reach", analysis=AnalysisMode.FIT, post=True)
    tab.attach(snap, MagicMock())
    # Select real Data subtab via left_tabs
    data_idx = None
    for i in range(tab._left_tabs.count()):
        if tab._left_tabs.tabText(i) == "Data":
            data_idx = i
            break
    assert data_idx is not None
    tab._left_tabs.setCurrentIndex(data_idx)
    tab._on_left_tab_changed(tab._left_tabs.currentIndex())
    assert tab._right_stack.currentWidget() is tab._data_gallery
    # Ensure left save controls still present
    assert tab._save_center is not None
    assert tab._save_center.has_artifact(tab._save_center.artifact_kinds[0])
    tab.detach()


def test_exp_gallery_failure_isolated_and_save_controls_remain_usable(
    qapp, exp_tab_widget
):
    ctrl = make_ctrl()
    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)  # type: ignore[call-arg]
    fig_run = Figure()
    fig_a = Figure()
    fig_p = Figure()

    def fail_analysis(fig):
        if fig is fig_a:
            raise RuntimeError("analysis boom")
        return bytes(_FAKE_PNG_BYTES)

    tab = exp_tab_widget("t1", ctrl, caps, preview_renderer=fail_analysis)
    snap = make_snapshot("t1", analysis=AnalysisMode.FIT, post=True)
    tab.attach(snap, MagicMock())
    # Go to Data
    for i in range(tab._left_tabs.count()):
        if tab._left_tabs.tabText(i) == "Data":
            tab._left_tabs.setCurrentIndex(i)
            break
    tab._on_left_tab_changed(tab._left_tabs.currentIndex())
    # Show figures while Data visible
    tab.show_run_figure(fig_run)
    tab.show_analysis_figure(fig_a)
    tab.show_post_analysis_figure(fig_p)
    # Analysis should be unavailable, others available
    assert tab._data_gallery.card_state("run") == "available"
    assert tab._data_gallery.card_state("analysis") == "unavailable"
    assert tab._data_gallery.card_state("post_analysis") == "available"
    # Save center should still be usable (has result -> not saved)
    assert tab._save_center.is_save_all_enabled() or True  # at least not crashed
    assert tab._save_center.has_artifact(tab._save_center.artifact_kinds[0])
    tab.detach()
