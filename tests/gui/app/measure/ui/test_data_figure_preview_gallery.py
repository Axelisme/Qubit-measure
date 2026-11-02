"""Standalone DataFigurePreviewGallery capabilities, failure isolation, and geometry contracts."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from matplotlib.figure import Figure
from PIL import Image, ImageChops
from qtpy.QtCore import QCoreApplication, QEvent, QRect, QSize
from qtpy.QtWidgets import QApplication, QLabel, QScrollArea, QWidget
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.measure.ui.data_figure_preview_gallery import (
    DataFigurePreviewGallery,
)
from zcu_tools.plotting.figures import FigureCollection

from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure.ui._preview_gallery_support import (
    FAKE_PNG_BYTES,
    assert_aspect_fit,
    fake_renderer,
    framed_renderer,
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


@pytest.fixture(params=[2, 3], ids=["two-cards", "three-cards"])
def named_gallery(
    qapp, request: pytest.FixtureRequest
) -> Iterator[tuple[ComponentDriver, int]]:
    """Present synthetic named groups through the gallery's public snapshot API."""
    count: int = request.param
    gallery = DataFigurePreviewGallery(
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=count == 3)
    )
    groups = []
    for stage in ("Run", "Analysis", "Post")[:count]:
        group = FigureCollection()
        for index in (1, 2):
            figure = Figure(figsize=(3, 2))
            axes = figure.subplots()
            axes.plot([0, 1, 2], [0, index, 0])
            axes.set_xlabel("Time (s)")
            axes.set_ylabel("Signal")
            group.adopt(f"{stage} tone {index}", figure)
        group.seal()
        groups.append(group)
    gallery.update_figures(*(next(iter(group.values())) for group in groups))
    gallery.update_collections(*groups)
    gallery.resize(1000, 1000)
    gallery.show()
    QApplication.processEvents()
    try:
        yield ComponentDriver(gallery), count
    finally:
        gallery.close()
        gallery.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _visible_card_bounds(driver: ComponentDriver, key: str) -> QRect:
    card = driver.observe(object_name=f"previewCard_{key}")
    assert card.fully_visible, f"{key} card is clipped or hidden"
    assert card.bounds is not None
    return QRect(*card.bounds)


def test_gallery_named_groups_follow_viewport(
    named_gallery: tuple[ComponentDriver, int], tmp_path: Path
) -> None:
    """Named groups use the same wide mosaic and narrow column after resizing."""
    driver, count = named_gallery
    evidence = Path(os.environ.get("GALLERY_LAYOUT_EVIDENCE_DIR", str(tmp_path)))
    for step, width in (("01-wide", 1000), ("02-narrow", 360), ("03-wide-again", 1000)):
        driver.resize_window(width, 1000)
        driver.screenshot(evidence / f"{count}-cards-{step}.png")
        run = _visible_card_bounds(driver, "run")
        analysis = _visible_card_bounds(driver, "analysis")
        assert driver.observe("Run tone 1").visible
        assert driver.observe("Analysis tone 2").visible
        if width == 360:
            assert run.left() == analysis.left()
            assert run.width() == analysis.width()
            assert run.bottom() < analysis.top()
            if count == 3:
                post = _visible_card_bounds(driver, "post")
                assert post.left() == analysis.left()
                assert post.width() == analysis.width()
                assert analysis.bottom() < post.top()
        else:
            assert run.right() < analysis.left(), "Run must be left of Analysis"
            assert run.top() == analysis.top()
            if count == 2:
                assert run.bottom() == analysis.bottom()
            else:
                post = _visible_card_bounds(driver, "post")
                assert post.left() == analysis.left()
                assert analysis.bottom() < post.top()
                assert run.bottom() == post.bottom()
                assert run.height() > analysis.height()
                assert run.width() >= analysis.width()


@pytest.fixture
def member_gallery(qapp) -> Iterator[ComponentDriver]:
    """Show six perimeter-marked rasters through public collection snapshots."""
    gallery = DataFigurePreviewGallery(
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True),
        renderer=framed_renderer,
    )
    gallery.setObjectName("memberFitGallery")
    groups = []
    colours = iter(("#c02020", "#208020", "#2020c0", "#a020a0", "#208080", "#a06020"))
    for stage in ("Run", "Analysis", "Post"):
        group = FigureCollection()
        for index in (1, 2):
            group.adopt(f"{stage} tone {index}", Figure(label=next(colours)))
        group.seal()
        groups.append(group)
    gallery.update_figures(*(next(iter(group.values())) for group in groups))
    gallery.update_collections(*groups)
    gallery.resize(1000, 1000)
    gallery.show()
    QApplication.processEvents()
    try:
        yield ComponentDriver(gallery)
    finally:
        gallery.close()
        gallery.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@dataclass(frozen=True)
class _RasterObservation:
    """One member raster in window coordinates, with its four visible perimeter edges."""

    stage_key: str
    member_index: int
    stage_bounds: QRect
    member_bounds: QRect
    raster_bounds: QRect
    edges: list[bool]


def test_gallery_member_rasters_fit_after_resize(
    member_gallery: ComponentDriver, tmp_path: Path
) -> None:
    """Every raster keeps all four perimeter edges inside its member and stage."""
    driver = member_gallery
    evidence = Path(os.environ.get("GALLERY_MEMBER_EVIDENCE_DIR", str(tmp_path)))
    colours = ("#c02020", "#208020", "#2020c0", "#a020a0", "#208080", "#a06020")
    measurements = {}
    all_rows: list[_RasterObservation] = []
    for step, width in (("01-wide", 1000), ("02-narrow", 360), ("03-wide-again", 1000)):
        driver.resize_window(width, 1000)
        png = driver.screenshot(evidence / f"member-{step}.png")
        image = Image.open(png).convert("RGB")
        window = driver.observe(object_name="memberFitGallery")
        assert window.bounds is not None
        origin_x, origin_y, _, _ = window.bounds
        widgets = driver.visible().widgets
        rows: list[_RasterObservation] = []
        for stage_index, key in enumerate(("run", "analysis", "post")):
            stage = _visible_card_bounds(driver, key)
            members = sorted(
                (
                    QRect(*widget.bounds)
                    for widget in widgets
                    if widget.object_name == f"previewCard_{key}_member"
                    and widget.bounds is not None
                ),
                key=lambda rect: rect.x(),
            )
            assert len(members) == 2
            for index, member in enumerate(members):
                colour = colours[2 * stage_index + index]
                mask = (
                    ImageChops.difference(image, Image.new("RGB", image.size, colour))
                    .convert("L")
                    .point([255] + [0] * 255)
                )
                bounds = mask.getbbox()
                assert bounds is not None, f"{key} member {index} raster disappeared"
                x0, y0, x1, y1 = bounds
                raster = QRect(x0 + origin_x, y0 + origin_y, x1 - x0, y1 - y0)
                edges = [
                    bool(mask.getpixel(point))
                    for point in (
                        ((x0 + x1) // 2, y0),
                        ((x0 + x1) // 2, y1 - 1),
                        (x0, (y0 + y1) // 2),
                        (x1 - 1, (y0 + y1) // 2),
                    )
                ]
                rows.append(
                    _RasterObservation(
                        stage_key=key,
                        member_index=index,
                        stage_bounds=stage,
                        member_bounds=member,
                        raster_bounds=raster,
                        edges=edges,
                    )
                )
        measurements[step] = [
            {
                "stage": row.stage_key,
                "member": row.member_index,
                "stage_bounds": row.stage_bounds.getRect(),
                "member_bounds": row.member_bounds.getRect(),
                "raster_bounds": row.raster_bounds.getRect(),
                "edges": row.edges,
            }
            for row in rows
        ]
        (evidence / "dimensions.json").write_text(json.dumps(measurements, indent=2))
        all_rows.extend(rows)
    for row in all_rows:
        detail = (row.stage_key, row.member_index, row.raster_bounds.getRect())
        assert row.member_bounds.contains(row.raster_bounds), detail
        assert row.stage_bounds.contains(row.raster_bounds), detail
        assert all(row.edges), f"Raster perimeter is clipped: {detail}, edges={row.edges}"
        assert (
            abs(row.raster_bounds.width() * 3 - row.raster_bounds.height() * 4) <= 4
        ), detail


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
