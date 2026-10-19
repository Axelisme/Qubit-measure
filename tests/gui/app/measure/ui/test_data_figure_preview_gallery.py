"""Standalone DataFigurePreviewGallery capabilities, failure isolation, and geometry contracts."""

from __future__ import annotations

import pytest
from matplotlib.figure import Figure
from qtpy.QtCore import QCoreApplication, QEvent, QRect, QSize
from qtpy.QtWidgets import QApplication, QLabel, QScrollArea, QWidget
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.measure.ui.data_figure_preview_gallery import (
    DataFigurePreviewGallery,
)

from tests.gui.app.measure.ui._preview_gallery_support import (
    FAKE_PNG_BYTES,
    assert_aspect_fit,
    fake_renderer,
    gallery_geometry_cards,
    renderer_640x480,
)

_640x480_PNG = None  # lazy


def _failing_renderer_for(target_fig: object):
    def _render(fig: object) -> bytes:
        if fig is target_fig:
            raise RuntimeError("boom render")
        return bytes(FAKE_PNG_BYTES)

    return _render


# ---------------------------------------------------------------------------
# Gallery standalone — S1, S3, S4
# ---------------------------------------------------------------------------


class _ViewportAccessError(Exception):
    """An unexpected Qt collaborator failure, not a deleted-wrapper error."""


@pytest.mark.parametrize("entry", ["constructor", "is_wide_mode"])
@pytest.mark.parametrize("failure_type", [RuntimeError, _ViewportAccessError])
def test_gallery_only_recovers_viewport_lifetime_errors(
    qapp, monkeypatch, entry: str, failure_type: type[Exception]
):
    caps = AdapterCapabilities(analysis=AnalysisMode.NONE)
    gallery = DataFigurePreviewGallery(caps) if entry == "is_wide_mode" else None
    if gallery is not None:
        gallery.setFixedWidth(360)
    original_viewport = QScrollArea.viewport
    failure = failure_type("viewport unavailable")
    pending = True

    def viewport(scroll: QScrollArea):
        nonlocal pending
        if pending:
            pending = False
            raise failure
        return original_viewport(scroll)

    def invoke_entry() -> bool:
        if gallery is not None:
            return not gallery.is_wide_mode()
        created = DataFigurePreviewGallery(caps)
        try:
            return created.card_count() == 1
        finally:
            created.deleteLater()

    try:
        # Restore the Qt collaborator before any owner-loop delivery.
        with monkeypatch.context() as patch:
            patch.setattr(QScrollArea, "viewport", viewport)
            if failure_type is RuntimeError:
                assert invoke_entry()
            else:
                with pytest.raises(_ViewportAccessError) as caught:
                    invoke_entry()
                assert caught.value is failure
    finally:
        if gallery is not None:
            gallery.deleteLater()


def test_gallery_capability_driven_cards(qapp):
    caps_none = AdapterCapabilities(analysis=AnalysisMode.NONE, post_analysis=False)
    g_none = DataFigurePreviewGallery(caps_none, renderer=fake_renderer)
    assert g_none.has_card("run")
    assert not g_none.has_card("analysis")
    assert not g_none.has_card("post_analysis")
    assert g_none.card_count() == 1
    # Find via objectName
    assert g_none.findChild(QWidget, "previewCard_run") is not None
    assert g_none.findChild(QWidget, "previewCard_analysis") is None

    # Run + Analysis
    caps_a = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False)
    g_a = DataFigurePreviewGallery(caps_a, renderer=fake_renderer)
    assert g_a.has_card("run")
    assert g_a.has_card("analysis")
    assert not g_a.has_card("post_analysis")
    assert g_a.card_count() == 2

    # All three
    caps_both = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)
    g_both = DataFigurePreviewGallery(caps_both, renderer=fake_renderer)
    assert g_both.has_card("run")
    assert g_both.has_card("analysis")
    assert g_both.has_card("post_analysis")
    assert g_both.card_count() == 3
    # Scrollable rail
    assert isinstance(
        g_both.findChild(QScrollArea, "previewGalleryScroll"), QScrollArea
    )


def test_gallery_card_delegates_nullable_image_event(qapp):
    caps = AdapterCapabilities(analysis=AnalysisMode.NONE, post_analysis=False)
    gallery = DataFigurePreviewGallery(caps, renderer=fake_renderer)
    try:
        card = gallery.find_card("run")
        assert card is not None
        image_label = card.findChild(QLabel, "previewImage_run")
        assert image_label is not None

        assert card.eventFilter(image_label, None) is False
    finally:
        gallery.close()
        gallery.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_gallery_supported_but_empty_shows_named_empty_state(qapp):
    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)
    g = DataFigurePreviewGallery(caps, renderer=fake_renderer)
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
    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)
    g = DataFigurePreviewGallery(caps, renderer=fake_renderer)
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
    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)
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
        return bytes(FAKE_PNG_BYTES)

    g2 = DataFigurePreviewGallery(caps, renderer=fail_post)
    g2.update_figures(fig_run, fig_a, fig_p)
    assert g2.card_state("run") == "available"
    assert g2.card_state("analysis") == "available"
    assert g2.card_state("post_analysis") == "unavailable"


def test_gallery_never_retains_figure(qapp):
    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False)
    held = []

    def capturing_renderer(fig):
        held.append(fig)
        return bytes(FAKE_PNG_BYTES)

    g = DataFigurePreviewGallery(caps, renderer=capturing_renderer)
    fig = Figure()
    g.update_figures(fig, Figure(), None)
    # Gallery should not hold Figure refs beyond call; we cleared held list
    # but verify gallery's cards have no 'figure' attr
    for key in ("run", "analysis"):
        card = g.find_card(key)
        assert card is not None
        assert not hasattr(card, "figure")
    # Ensure update does not store figure in gallery attribute
    assert not any(
        hasattr(g, attr) and getattr(g, attr) is fig
        for attr in dir(g)
        if not attr.startswith("_")
    )


# ---------------------------------------------------------------------------
# A3 regression: 640x480 pixmap must aspect-fit inside ~396x120 viewport
# ---------------------------------------------------------------------------


def test_gallery_aspect_fit_regression_640x480_in_narrow_viewport(qapp):
    """Fails on current fixed-size renderer without KeepAspectRatio scaling.

    Production screenshots are 640x480. Card image viewport in the Data
    gallery is approximately 396x120 when the gallery is ~434px wide. The
    displayed pixmap must stay within the viewport, preserve 4:3 aspect,
    and remain correct after resize (S3). This test would fail on the
    pre-fix code that stored the raw 640x480 pixmap directly.
    """
    caps = AdapterCapabilities(analysis=AnalysisMode.NONE, post_analysis=False)
    g = DataFigurePreviewGallery(caps, renderer=renderer_640x480)
    # Show gallery at size that reproducibly yields ~396x120 image viewport.
    # 434x200 was determined to produce 396x120 label viewport with 160x120 display
    # (see probe); using narrow tolerance ensures clipping regression is real.
    g.setFixedSize(434, 200)
    g.show()
    QApplication.processEvents()
    QApplication.processEvents()
    fig = Figure()
    g.update_figures(fig)
    QApplication.processEvents()
    QApplication.processEvents()
    card = g.find_card("run")
    assert card is not None
    vp = card.image_viewport_size()
    disp = card.displayed_pixmap()
    orig = card.original_pixmap()
    assert vp is not None and disp is not None and orig is not None
    # Narrow tolerance for the ~396x120 regression geometry.
    assert 390 <= vp.width() <= 402, f"viewport width {vp.width()} not ~396"
    assert 115 <= vp.height() <= 125, f"viewport height {vp.height()} not ~120"
    # Original must be 640x480 (production)
    assert orig.width() == 640 and orig.height() == 480
    # Displayed must be ~160x120 (4:3) and tight within viewport.
    assert 155 <= disp.width() <= 165, f"displayed width {disp.width()} not ~160"
    assert 115 <= disp.height() <= 125, f"displayed height {disp.height()} not ~120"
    # Displayed must fit and preserve aspect — would fail pre-fix where disp==640x480
    assert_aspect_fit(vp, disp.size(), orig.size())
    # After resize: widen gallery, viewport grows, displayed must re-scale and still fit
    g.setFixedSize(700, 500)
    QApplication.processEvents()
    QApplication.processEvents()
    vp2 = card.image_viewport_size()
    disp2 = card.displayed_pixmap()
    assert vp2 is not None and disp2 is not None
    # Width comes from fixed margins only, so it stays narrow. Height is the
    # gallery height minus the header/hint/title text rows, whose line heights
    # follow platform font metrics (~407 where first captured, ~417 on headless Linux);
    # so only require growth here and lock the exact fit below.
    assert 670 <= vp2.width() <= 682, f"vp2 width {vp2.width()} not ~676"
    assert vp.height() < vp2.height() < 500, f"vp2 height {vp2.height()} did not grow"
    # Largest 4:3 size inside vp2, as QSize.scaled(KeepAspectRatio) rounds it.
    fit_w = vp2.height() * orig.width() // orig.height()
    if fit_w <= vp2.width():
        expected = QSize(fit_w, vp2.height())
    else:
        expected = QSize(vp2.width(), vp2.width() * orig.height() // orig.width())
    assert disp2.size() == expected, f"disp2 {disp2.size()} != fit {expected}"
    assert_aspect_fit(vp2, disp2.size(), orig.size())
    # Original cache must stay 640x480 across resizes
    # tolerated: The successfully rendered card retains its original pixmap across viewport resizes.
    assert card.original_pixmap().size() == orig.size()  # pyright: ignore[reportOptionalMemberAccess]
    g.hide()


# ---------------------------------------------------------------------------
# S1/S4 responsive geometry — viewport-driven narrow vs wide mosaic
# ---------------------------------------------------------------------------


def test_gallery_narrow_single_column_three_cards(qapp):
    """Narrow viewport (< threshold) must show three cards in one vertical column (S1)."""
    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)
    g = DataFigurePreviewGallery(caps, renderer=fake_renderer)
    g.setFixedSize(360, 700)  # narrow: viewport ~348 < 450
    g.show()
    QApplication.processEvents()
    QApplication.processEvents()
    assert not g.is_wide_mode(), "expected narrow mode at 360, got wide"
    geoms = gallery_geometry_cards(g)
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
    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)
    g = DataFigurePreviewGallery(caps, renderer=fake_renderer)
    g.setFixedSize(750, 700)  # wide: viewport ~738 > 450
    g.show()
    QApplication.processEvents()
    QApplication.processEvents()
    assert g.is_wide_mode(), "expected wide mode at 750"
    geoms = gallery_geometry_cards(g)
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
    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False)
    g = DataFigurePreviewGallery(caps, renderer=fake_renderer)
    g.setFixedSize(750, 400)
    g.show()
    QApplication.processEvents()
    QApplication.processEvents()
    assert g.is_wide_mode()
    geoms = gallery_geometry_cards(g)
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
    caps = AdapterCapabilities(analysis=AnalysisMode.NONE, post_analysis=False)
    # Narrow
    g_n = DataFigurePreviewGallery(caps, renderer=fake_renderer)
    g_n.setFixedSize(360, 400)
    g_n.show()
    QApplication.processEvents()
    QApplication.processEvents()
    assert not g_n.is_wide_mode()
    # tolerated: Run-only capabilities always create the run card queried here.
    r_n: QRect = g_n.find_card("run").geometry()  # pyright: ignore[reportOptionalMemberAccess]
    # Wide
    g_w = DataFigurePreviewGallery(caps, renderer=fake_renderer)
    g_w.setFixedSize(750, 400)
    g_w.show()
    QApplication.processEvents()
    QApplication.processEvents()
    # tolerated: Run-only capabilities always create the run card queried here.
    r_w: QRect = g_w.find_card("run").geometry()  # pyright: ignore[reportOptionalMemberAccess]
    # Both should span full inner width — wide width larger than narrow
    assert r_w.width() > r_n.width() + 20
    # In single-card case, card x is near left margin for both
    assert r_n.x() <= 4 and r_w.x() <= 4
    g_n.hide()
    g_w.hide()
