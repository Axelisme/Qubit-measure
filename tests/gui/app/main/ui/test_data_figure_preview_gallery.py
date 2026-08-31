"""Variant A stacked Data previews — gallery and ExpTabWidget integration (Ticket 002)."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from qtpy.QtCore import (  # type: ignore[attr-defined]
    QBuffer,
    QByteArray,
    QIODevice,
    QRect,
    QSize,
    Qt,
)
from qtpy.QtGui import QColor, QPixmap  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QApplication,
    QLabel,
    QScrollArea,
    QWidget,
)
from zcu_tools.gui.app.main.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.main.services import PersistedStartup, TabSnapshot
from zcu_tools.gui.app.main.state import TabInteractionState
from zcu_tools.gui.app.main.ui.artifact_save_center import ArtifactKind
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


def _make_png_bytes(width: int, height: int, color: str = "#286ac7") -> bytes:
    """Deterministic PNG bytes of exact size (for aspect-fit regression)."""
    pix = QPixmap(width, height)
    pix.fill(QColor(color))
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.WriteOnly)  # type: ignore[attr-defined]
    pix.save(buf, "PNG")  # type: ignore[attr-defined]
    buf.close()
    return bytes(ba)  # type: ignore[arg-type]


_640x480_PNG = None  # lazy


def _renderer_640x480(_fig: object) -> bytes:
    # Each call returns fresh 640x480 PNG (production screenshot size).
    return _make_png_bytes(640, 480, "#286ac7")


def _aspect_ratio(size: QSize) -> float:
    if size.height() == 0:
        return 0.0
    return float(size.width()) / float(size.height())


def _assert_aspect_fit(
    viewport: QSize, displayed: QSize, original: QSize, *, tol: float = 0.02
) -> None:
    # Displayed must fit within viewport (allow 1px rounding)
    assert displayed.width() <= viewport.width() + 1, (
        f"width {displayed.width()} > viewport {viewport.width()}"
    )
    assert displayed.height() <= viewport.height() + 1, (
        f"height {displayed.height()} > viewport {viewport.height()}"
    )
    # Must not be cropped to zero
    assert displayed.width() > 0 and displayed.height() > 0
    # Aspect ratio preserved
    orig_ar = _aspect_ratio(original)
    disp_ar = _aspect_ratio(displayed)
    assert abs(orig_ar - disp_ar) < tol, (
        f"aspect {disp_ar:.3f} vs original {orig_ar:.3f}"
    )
    # Displayed should be maximal within viewport: at least one dimension tight (within 2px)
    # For 640x480 -> 396x120 viewport, height should be ~120, width ~160.
    assert (
        abs(displayed.width() - viewport.width()) <= 2
        or abs(displayed.height() - viewport.height()) <= 2
    ), (
        f"not tight {displayed.width()}x{displayed.height()} in {viewport.width()}x{viewport.height()}"
    )


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
    assert g.findChild(QScrollArea, "previewGalleryScroll") is not None
    assert g.card_count() == 3

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
    # Save center should still be usable after single-card failure (A3)
    assert tab._save_center.has_artifact(ArtifactKind.DATA)
    assert tab._save_center.is_save_all_enabled() is True
    assert tab._save_center.is_save_enabled(ArtifactKind.DATA) is True
    assert tab._save_center.has_artifact(tab._save_center.artifact_kinds[0])
    tab.detach()


# ---------------------------------------------------------------------------
# A3 regression: 640x480 pixmap must aspect-fit inside ~396x120 viewport
# ---------------------------------------------------------------------------


def test_gallery_aspect_fit_regression_640x480_in_narrow_viewport(qapp):
    """Fails on current fixed-size renderer without KeepAspectRatio scaling.

    Production screenshots are 640x480. Card image viewport in the Data
    gallery is approximately 396x120 when the gallery is ~420px wide. The
    displayed pixmap must stay within the viewport, preserve 4:3 aspect,
    and remain correct after resize (S3). This test would fail on the
    pre-fix code that stored the raw 640x480 pixmap directly.
    """
    caps = AdapterCapabilities(analysis=AnalysisMode.NONE, post_analysis=False)  # type: ignore[call-arg]
    g = DataFigurePreviewGallery(caps, renderer=_renderer_640x480)
    # Show gallery at narrow width that yields ~396x120 image viewport.
    g.setFixedSize(420, 400)
    g.show()
    QApplication.processEvents()  # type: ignore[attr-defined]
    QApplication.processEvents()
    fig = Figure()
    g.update_figures(fig)
    QApplication.processEvents()
    card = g.find_card("run")
    assert card is not None
    # Viewport is the image label's size, expected ~396x120 in this geometry.
    vp = card.image_viewport_size()
    disp = card.displayed_pixmap()
    orig = card.original_pixmap()
    assert vp is not None and disp is not None and orig is not None
    # Viewport width should be ~396 for gallery ~420 (narrow). Height depends on gallery height
    # and card layout; we only require width ~396 and height at least 100.
    assert 350 <= vp.width() <= 430, f"viewport width {vp.width()} not ~396"
    assert 100 <= vp.height() <= 400, f"viewport height {vp.height()} out of range"
    # Original must be 640x480 (production)
    assert orig.width() == 640 and orig.height() == 480
    # Displayed must fit and preserve aspect — would fail pre-fix where disp==640x480
    _assert_aspect_fit(vp, disp.size(), orig.size())
    # After resize: widen gallery, viewport grows, displayed must re-scale and still fit
    g.setFixedSize(700, 500)
    QApplication.processEvents()
    QApplication.processEvents()
    vp2 = card.image_viewport_size()
    disp2 = card.displayed_pixmap()
    assert vp2 is not None and disp2 is not None
    assert vp2.width() > vp.width() or vp2.height() >= vp.height()
    _assert_aspect_fit(vp2, disp2.size(), orig.size())
    # Original cache must stay 640x480 across resizes
    assert card.original_pixmap().size() == orig.size()  # type: ignore[union-attr]
    g.hide()


def test_gallery_aspect_fit_preserved_after_multiple_resizes_via_exp_tab(
    qapp, exp_tab_widget
):
    """Via shipped ExpTabWidget Data path, verify bounds and ratio before/after resize."""
    ctrl = make_ctrl()
    caps = AdapterCapabilities(analysis=AnalysisMode.NONE, post_analysis=False)  # type: ignore[call-arg]
    tab = exp_tab_widget("t-aspect", ctrl, caps, preview_renderer=_renderer_640x480)
    snap = make_snapshot("t-aspect", analysis=AnalysisMode.NONE, post=False)
    tab.attach(snap, MagicMock())
    # Switch to Data so gallery is active right widget
    for i in range(tab._left_tabs.count()):
        if tab._left_tabs.tabText(i) == "Data":
            tab._left_tabs.setCurrentIndex(i)
            break
    tab._on_left_tab_changed(tab._left_tabs.currentIndex())
    assert tab._right_stack.currentWidget() is tab._data_gallery
    tab.show()
    tab.resize(500, 600)
    QApplication.processEvents()
    QApplication.processEvents()
    fig = Figure()
    tab.show_run_figure(fig)
    QApplication.processEvents()
    card = tab._data_gallery.find_card("run")
    assert card is not None
    vp = card.image_viewport_size()
    disp = card.displayed_pixmap()
    orig = card.original_pixmap()
    assert vp is not None and disp is not None and orig is not None
    _assert_aspect_fit(vp, disp.size(), orig.size())
    # Resize larger — simulate user dragging splitter wider (still gallery viewport, not window)
    tab._data_gallery.setFixedWidth(700)
    QApplication.processEvents()
    QApplication.processEvents()
    vp2 = card.image_viewport_size()
    disp2 = card.displayed_pixmap()
    assert vp2 is not None and disp2 is not None
    _assert_aspect_fit(vp2, disp2.size(), orig.size())
    tab.detach()


# ---------------------------------------------------------------------------
# S1/S4 responsive geometry — viewport-driven narrow vs wide mosaic
# ---------------------------------------------------------------------------


def _gallery_geometry_cards(g: DataFigurePreviewGallery) -> dict[str, QRect]:
    # Force layout
    QApplication.processEvents()  # type: ignore[attr-defined]
    QApplication.processEvents()  # type: ignore[attr-defined]
    geoms: dict[str, QRect] = {}
    for key in ["run", "analysis", "post_analysis"]:
        c = g.find_card(key)
        if c is not None:
            geoms[key] = c.geometry()  # type: ignore[attr-defined]
    return geoms


def test_gallery_narrow_single_column_three_cards(qapp):
    """Narrow viewport (< threshold) must show three cards in one vertical column (S1)."""
    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)  # type: ignore[call-arg]
    g = DataFigurePreviewGallery(caps, renderer=_fake_renderer)
    g.setFixedSize(360, 700)  # narrow: viewport ~348 < 450
    g.show()
    QApplication.processEvents()
    QApplication.processEvents()
    assert not g.is_wide_mode(), f"expected narrow mode at 360, got wide"
    geoms = _gallery_geometry_cards(g)
    assert "run" in geoms and "analysis" in geoms and "post_analysis" in geoms
    r: QRect = geoms["run"]
    a: QRect = geoms["analysis"]
    p: QRect = geoms["post_analysis"]
    # Single column -> same x, y strictly increasing
    assert abs(r.x() - a.x()) <= 2 and abs(a.x() - p.x()) <= 2, (
        f"x not aligned {r.x()},{a.x()},{p.x()}"
    )
    assert r.y() < a.y() < p.y(), f"y not monotonic {r.y()},{a.y()},{p.y()}"
    # Widths should all span full available width (within tolerance)
    assert abs(r.width() - a.width()) <= 4 and abs(a.width() - p.width()) <= 4
    g.hide()


def test_gallery_wide_three_card_mosaic_run_left_spanning(qapp):
    """Wide + 3 cards: Run left spanning two rows, Analysis/Post right (S1/S4)."""
    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)  # type: ignore[call-arg]
    g = DataFigurePreviewGallery(caps, renderer=_fake_renderer)
    g.setFixedSize(750, 700)  # wide: viewport ~738 > 450
    g.show()
    QApplication.processEvents()  # type: ignore[attr-defined]
    QApplication.processEvents()  # type: ignore[attr-defined]
    assert g.is_wide_mode(), "expected wide mode at 750"
    geoms = _gallery_geometry_cards(g)
    r: QRect = geoms["run"]
    a: QRect = geoms["analysis"]
    p: QRect = geoms["post_analysis"]
    # Run left, others right
    assert r.x() < a.x() - 2, f"Run not left of Analysis {r.x()} vs {a.x()}"
    assert abs(a.x() - p.x()) <= 2, f"Analysis/Post x not aligned {a.x()} vs {p.x()}"
    # Top alignment: Run top == Analysis top
    assert abs(r.y() - a.y()) <= 4, f"Run/Analysis top not aligned {r.y()} vs {a.y()}"
    # Post below Analysis
    assert p.y() > a.y() + 2, f"Post not below Analysis {p.y()} vs {a.y()}"
    # Run spans two rows -> height approx Analysis+Post+spacing
    assert r.height() > a.height() + 2, (
        f"Run height {r.height()} not > Analysis {a.height()}"
    )
    assert abs(r.height() - (a.height() + p.height() + 10)) <= 20, (
        f"Run height {r.height()} not sum {a.height()}+{p.height()}"
    )
    # Run width larger than right column (stretch 2 vs 1) or at least not smaller
    assert r.width() >= a.width() - 4
    g.hide()


def test_gallery_wide_two_card_side_by_side(qapp):
    """Wide + 2 cards: Run and Analysis side-by-side (S4)."""
    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False)  # type: ignore[call-arg]
    g = DataFigurePreviewGallery(caps, renderer=_fake_renderer)
    g.setFixedSize(750, 400)
    g.show()
    QApplication.processEvents()  # type: ignore[attr-defined]
    QApplication.processEvents()  # type: ignore[attr-defined]
    assert g.is_wide_mode()
    geoms = _gallery_geometry_cards(g)
    r: QRect = geoms["run"]
    a: QRect = geoms["analysis"]
    # Side-by-side -> same y, distinct x
    assert abs(r.y() - a.y()) <= 4, f"Run/Analysis y not aligned {r.y()} vs {a.y()}"
    assert a.x() > r.x() + r.width() - 4, (
        f"Analysis not right of Run {a.x()} vs {r.x()}+{r.width()}"
    )
    # Heights similar
    assert abs(r.height() - a.height()) <= 10
    # Widths similar (equal stretch)
    assert abs(r.width() - a.width()) <= 20
    assert "post_analysis" not in geoms
    g.hide()


def test_gallery_wide_single_full_width(qapp):
    """Single card (Run only) occupies full width in both narrow and wide (S4)."""
    caps = AdapterCapabilities(analysis=AnalysisMode.NONE, post_analysis=False)  # type: ignore[call-arg]
    # Narrow
    g_n = DataFigurePreviewGallery(caps, renderer=_fake_renderer)
    g_n.setFixedSize(360, 400)
    g_n.show()
    QApplication.processEvents()  # type: ignore[attr-defined]
    QApplication.processEvents()  # type: ignore[attr-defined]
    assert not g_n.is_wide_mode()
    r_n: QRect = g_n.find_card("run").geometry()  # type: ignore[union-attr]
    # Wide
    g_w = DataFigurePreviewGallery(caps, renderer=_fake_renderer)
    g_w.setFixedSize(750, 400)
    g_w.show()
    QApplication.processEvents()  # type: ignore[attr-defined]
    QApplication.processEvents()  # type: ignore[attr-defined]
    r_w: QRect = g_w.find_card("run").geometry()  # type: ignore[union-attr]
    # Both should span full inner width — wide width larger than narrow
    assert r_w.width() > r_n.width() + 20
    # In single-card case, card x is near left margin for both
    assert r_n.x() <= 4 and r_w.x() <= 4
    g_n.hide()
    g_w.hide()


def test_gallery_narrow_vs_wide_via_exp_tab_shipped_path(qapp, exp_tab_widget):
    """Geometry via shipped ExpTabWidget Data path for each capability combination (A1)."""
    import zcu_tools.gui.app.main.ui.data_figure_preview_gallery as gallery_mod

    ctrl = make_ctrl()
    # Three-card case through ExpTabWidget
    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)  # type: ignore[call-arg]
    tab = exp_tab_widget("t-geom", ctrl, caps, preview_renderer=_fake_renderer)
    snap = make_snapshot("t-geom", analysis=AnalysisMode.FIT, post=True)
    tab.attach(snap, MagicMock())
    for i in range(tab._left_tabs.count()):
        if tab._left_tabs.tabText(i) == "Data":
            tab._left_tabs.setCurrentIndex(i)
            break
    tab._on_left_tab_changed(tab._left_tabs.currentIndex())
    assert tab._right_stack.currentWidget() is tab._data_gallery
    tab.show()
    QApplication.processEvents()  # type: ignore[attr-defined]
    QApplication.processEvents()  # type: ignore[attr-defined]
    g = tab._data_gallery
    # Narrow via gallery viewport
    g.setFixedWidth(360)
    QApplication.processEvents()  # type: ignore[attr-defined]
    QApplication.processEvents()  # type: ignore[attr-defined]
    g._arrange_cards()  # type: ignore[attr-defined]  # ensure viewport logic runs even if hidden before
    QApplication.processEvents()  # type: ignore[attr-defined]
    assert not g.is_wide_mode()
    # Verify narrow column via card x alignment
    r: QRect = g.find_card("run").geometry()  # type: ignore[union-attr]
    a: QRect = g.find_card("analysis").geometry()  # type: ignore[union-attr]
    p: QRect = g.find_card("post_analysis").geometry()  # type: ignore[union-attr]
    assert abs(r.x() - a.x()) <= 2 and abs(a.x() - p.x()) <= 2
    assert r.y() < a.y() < p.y()
    # Wide
    g.setFixedWidth(750)
    QApplication.processEvents()  # type: ignore[attr-defined]
    QApplication.processEvents()  # type: ignore[attr-defined]
    g._arrange_cards()  # type: ignore[attr-defined]
    QApplication.processEvents()  # type: ignore[attr-defined]
    assert g.is_wide_mode()
    r = g.find_card("run").geometry()  # type: ignore[union-attr]
    a = g.find_card("analysis").geometry()  # type: ignore[union-attr]
    p = g.find_card("post_analysis").geometry()  # type: ignore[union-attr]
    assert r.x() < a.x() - 2 and abs(a.x() - p.x()) <= 2
    assert r.y() <= a.y() + 4 and p.y() > a.y()
    # Also verify wide threshold uses gallery viewport not window width:
    # Gallery's own width decides, we just proved by toggling gallery width alone.
    assert g._viewport_available_width() >= gallery_mod.WIDE_THRESHOLD  # type: ignore[attr-defined]
    tab.detach()
    g.hide()
