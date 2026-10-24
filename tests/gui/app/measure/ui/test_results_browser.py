"""Results browsing through visible controls over synthetic Database content."""

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import QApplication, QLabel, QTreeWidget, QWidget
from zcu_tools.datafile import JsonObject, load, save
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import ExperimentResolution, Registry
from zcu_tools.gui.app.measure.services.load import LoadTabResultOutcome
from zcu_tools.gui.app.measure.services.results_index import ResultsIndex, scan_results
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.app.measure.ui.results_browser import ResultsBrowser
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
    PointUseRequest,
)

from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._native_save_support import (
    AnalysisSavingAdapter,
    NativeSaveSession,
    SavingAdapter,
)
from tests.gui.app.measure._record_support import require_record


def test_menu_opens_independent_results_window(
    browser_host: MainWindow,
    tmp_path: Path,
) -> None:
    ui = ComponentDriver(browser_host)
    assert "Results ▾" in tuple(w.text for w in ui.visible().widgets)
    ui.select_menu("Results ▾", ("Results browser…",))
    assert ui.observe(object_name="results.browser").visible
    assert ui.observe(object_name="results.focus").text == "All results · 0 runs"
    ui.screenshot(tmp_path / "empty-browser.png")


@pytest.mark.parametrize("run_focus", [False, True])
def test_grouping_reuses_visible_cards_and_decoded_images(
    populated_browser: ComponentDriver,
    browser_host: MainWindow,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_focus: bool,
) -> None:
    ui = populated_browser
    ui.enter_text(value="Q2", object_name="results.search")
    ui.click("Filter by date (UTC)")
    if run_focus:
        ui.click(object_name="results.figure.0")
        ui.click(object_name="results.figure.1")
    ui.invoke_context_action(action="Add to comparison", object_name="results.figure.0")
    focus = ui.observe(object_name="results.focus").text
    current = ui.observe(object_name="results.current_figure").text
    images = tuple(
        widget
        for widget in browser_host.findChildren(QWidget)
        if widget.objectName().startswith(("results.figure.", "results.card."))
    )
    reads: list[str] = []
    decodes: list[int] = []
    original_read = ResultsIndex.read_figure
    original_decode = QPixmap.loadFromData

    def read(index: ResultsIndex, key: str, selector: str) -> bytes:
        reads.append(selector)
        return original_read(index, key, selector)

    def decode(image: QPixmap, png: bytes) -> bool:
        decodes.append(len(png))
        return original_decode(image, png)

    with monkeypatch.context() as boundary:
        boundary.setattr(ResultsIndex, "read_figure", read)
        boundary.setattr(QPixmap, "loadFromData", decode)
        ui.select_combo("Experiment → point", object_name="results.group")
        ui.select_combo("Point → experiment", object_name="results.group")
        assert reads == []
        assert decodes == []
    assert images
    assert images == tuple(
        widget
        for widget in browser_host.findChildren(QWidget)
        if widget.objectName().startswith(("results.figure.", "results.card."))
    )
    assert ui.observe(object_name="results.focus").text == focus
    assert ui.observe(object_name="results.current_figure").text == current
    assert ui.observe(object_name="results.search").text == "Q2"
    assert ui.observe("Filter by date (UTC)").checked
    ui.select_tab("Comparison (1)")
    assert ui.observe(object_name="results.pin_count").text == "Pinned runs: 1"
    ui.select_tab("Gallery")
    if run_focus:
        ui.click("Back")
        assert (
            ui.observe(object_name="results.current_figure").text
            == "Current figure: plot-0.png"
        )
        ui.click("Forward")
        assert ui.observe(object_name="results.current_figure").text == current
    ui.screenshot(tmp_path / "grouping-reused.png")


def test_gallery_comparison_and_analysis_have_independent_tabs(
    populated_browser: ComponentDriver, tmp_path: Path
) -> None:
    ui = populated_browser
    assert ui.tabs("Gallery").captions == ("Gallery", "Comparison")
    ui.click(object_name="results.figure.0")
    ui.invoke_context_action(action="Add to comparison", object_name="results.figure.1")
    ui.select_tab("Analysis")
    assert any(
        row.columns == ("freq", "5100.0")
        for row in ui.visible().tree_rows
        if row.tree_name == "results.analysis"
    )
    ui.screenshot(tmp_path / "analysis-tab.png")
    ui.select_tab("Comparison (1)")
    assert ui.observe(object_name="results.pin_count").text == "Pinned runs: 1"
    assert ui.table_rows(object_name="results.comparison_table")
    ui.screenshot(tmp_path / "comparison-tab.png")
    ui.select_tab("Gallery")
    assert ui.observe(object_name="results.figure.0").visible


def test_comparison_tab_counts_unique_pins_remove_and_clear(
    populated_browser: ComponentDriver,
    tmp_path: Path,
) -> None:
    ui = populated_browser
    assert ui.tabs("Gallery").captions == ("Gallery", "Comparison")
    ui.click(object_name="results.figure.0")
    ui.click("Add to comparison")
    assert ui.tabs("Gallery").captions == ("Gallery", "Comparison (1)")
    ui.click("Add to comparison")
    assert ui.tabs("Gallery").captions == ("Gallery", "Comparison (1)")
    ui.click("Back")
    ui.click(object_name="results.figure.1")
    ui.click("Add to comparison")
    ui.select_tab("Comparison (2)")
    ui.screenshot(tmp_path / "count-two.png")
    ui.click(object_name="results.compare.remove.1")
    assert ui.tabs("Gallery").captions == ("Gallery", "Comparison (1)")
    ui.click("Clear comparison")
    assert ui.tabs("Gallery").captions == ("Gallery", "Comparison")
    ui.screenshot(tmp_path / "count-cleared.png")


def test_cached_gallery_cards_share_available_width_after_history(
    populated_browser: ComponentDriver, tmp_path: Path
) -> None:
    ui = populated_browser
    ui.resize_window(1920, 1200)
    ui.click(object_name="results.figure.0")
    ui.click("Back")
    ui.resize_window(1400, 850)
    ui.click("Forward")
    ui.screenshot(tmp_path / "cached-gallery-columns.png")
    first = ui.observe(object_name="results.figure.0").bounds
    second = ui.observe(object_name="results.figure.1").bounds
    assert first is not None and second is not None
    assert abs(first[2] - second[2]) <= 1
    ui.click("All results")
    ui.enter_text(value="bbbbbb", object_name="results.search")
    ui.screenshot(tmp_path / "cached-gallery-single-column.png")
    single = ui.observe(object_name="results.figure.0").bounds
    gallery = ui.observe(object_name="results.gallery").bounds
    assert single is not None and gallery is not None
    assert single[2] > gallery[2] * 0.8


def test_preview_height_tracks_available_card_width(
    populated_browser: ComponentDriver, tmp_path: Path
) -> None:
    ui = populated_browser
    ui.resize_window(1280, 800)
    small = ui.observe(object_name="results.figure.0").bounds
    ui.screenshot(tmp_path / "gallery-1280.png")
    ui.resize_window(1920, 1200)
    large = ui.observe(object_name="results.figure.0").bounds
    assert small is not None and large is not None
    assert large[2] > small[2]
    assert large[3] > small[3]
    assert abs(large[3] / large[2] - 220 / 400) < 0.02
    ui.screenshot(tmp_path / "gallery-1920.png")
    for number in range(2):
        ui.select_tab("Gallery")
        ui.invoke_context_action(
            action="Add to comparison", object_name=f"results.figure.{number}"
        )
    ui.select_tab("Comparison (2)")
    ui.resize_window(1280, 800)
    small_compare = ui.observe(object_name="results.compare.image.0.0").bounds
    ui.screenshot(tmp_path / "comparison-1280.png")
    ui.resize_window(1920, 1200)
    large_compare = ui.observe(object_name="results.compare.image.0.0").bounds
    assert small_compare is not None and large_compare is not None
    assert large_compare[2] > small_compare[2]
    assert large_compare[3] > small_compare[3]
    assert abs(large_compare[3] / large_compare[2] - 220 / 400) < 0.02
    ui.screenshot(tmp_path / "comparison-1920.png")


def test_date_range_can_be_enabled_edited_and_disabled(
    populated_browser: ComponentDriver, tmp_path: Path
) -> None:
    ui = populated_browser
    assert not ui.observe(object_name="results.date_from").enabled
    assert not ui.observe(object_name="results.date_to").enabled
    ui.screenshot(tmp_path / "date-initial-disabled.png")
    ui.click(object_name="results.dates_enabled")
    assert ui.observe(object_name="results.date_from").enabled
    assert ui.observe(object_name="results.date_to").enabled
    ui.screenshot(tmp_path / "date-enabled.png")
    ui.click(object_name="results.date_from")
    popup = QApplication.activePopupWidget()
    assert popup is not None and popup.isVisible()
    ComponentDriver(popup).screenshot(tmp_path / "date-from-calendar.png")
    ui.choose_calendar_day(22)
    assert ui.observe(object_name="results.date_from").text == "2026-10-22"
    # The two registered p1 runs use their acquired UTC day (21), not metadata day.
    assert ui.observe(object_name="results.visible_count").text == "1 / 3 runs"
    assert navigation_paths(ui) == {
        ("alpha",),
        ("alpha", "p2"),
        ("alpha", "p2", "mux/readout"),
    }
    ui.expand_tree(
        navigation_path(ui, ("alpha", "p2", "mux/readout")),
        object_name="results.navigation",
    )
    run_rows = [path for path in navigation_paths(ui) if len(path) == 4]
    assert len(run_rows) == 1 and run_rows[0][-1].startswith("bbbbbb")
    ui.screenshot(tmp_path / "date-from-filter.png")
    ui.open_calendar(object_name="results.date_from")
    ui.choose_calendar_day(21)
    assert ui.observe(object_name="results.visible_count").text == "3 / 3 runs"
    ui.open_calendar(object_name="results.date_to")
    ui.choose_calendar_day(21)
    assert ui.observe(object_name="results.visible_count").text == "2 / 3 runs"
    ui.screenshot(tmp_path / "date-valid-range.png")
    ui.open_calendar(object_name="results.date_from")
    ui.choose_calendar_day(22)
    assert (
        "From date must not follow To date"
        in ui.observe(object_name="results.feedback").text
    )
    assert ui.observe(object_name="results.visible_count").text == "0 / 3 runs"
    ui.screenshot(tmp_path / "date-invalid-range.png")
    ui.click("Filter by date (UTC)")
    assert ui.observe(object_name="results.visible_count").text == "3 / 3 runs"
    ui.screenshot(tmp_path / "date-disabled.png")


def test_comparison_double_click_joins_focus_history(
    populated_browser: ComponentDriver, tmp_path: Path
) -> None:
    ui = populated_browser
    ui.invoke_context_action(action="Add to comparison", object_name="results.figure.0")
    ui.select_tab("Comparison (1)")
    ui.select_table_row(
        "alpha", object_name="results.comparison_table", double_click=True
    )
    assert ui.tabs("Gallery").current == "Gallery"
    assert "aaaaaa · 1 runs" in ui.observe(object_name="results.focus").text
    ui.click("Back")
    assert ui.observe(object_name="results.focus").text == "All results · 3 runs"
    ui.click("Forward")
    assert "aaaaaa · 1 runs" in ui.observe(object_name="results.focus").text
    ui.select_tab("Comparison (1)")
    assert ui.observe(object_name="results.pin_count").text == "Pinned runs: 1"
    ui.screenshot(tmp_path / "comparison-return-history.png")


def test_detail_trees_bound_values_copy_full_json_and_allow_column_drag(
    populated_browser: ComponentDriver,
    browser_host: MainWindow,
    tmp_path: Path,
    saved_cfg: JsonObject,
) -> None:
    ui = populated_browser
    ui.click(object_name="results.figure.0")
    rows = [
        row.columns
        for row in ui.visible().tree_rows
        if row.tree_name == "results.details"
    ]
    assert ("branch", '{"leaf": "bounded"}') in rows
    assert ("long", f"{'x' * 200} …") in rows
    ui.screenshot(tmp_path / "bounded-cfg.png")
    ui.select_tree_row(("Cfg values",), object_name="results.details")
    ui.invoke_context_action(action="Copy JSON", object_name="results.details")
    clipboard = QApplication.clipboard()
    assert clipboard is not None
    assert json.loads(clipboard.text()) == saved_cfg["values"]
    for caption, name in (
        ("Metadata / cfg", "results.details"),
        ("Analysis", "results.analysis"),
    ):
        ui.select_tab(caption)
        tree = browser_host.findChild(QTreeWidget, name)
        assert tree is not None
        width = tree.columnWidth(0)
        ui.drag_column(0, -30, object_name=name)
        assert tree.columnWidth(0) == width - 30
        ui.screenshot(tmp_path / f"{name}-resized.png")


def test_initial_navigation_exposes_point_groups(
    populated_browser: ComponentDriver, tmp_path: Path
) -> None:
    ui = populated_browser
    assert ui.observe(object_name="results.group").text == "Point → experiment"
    paths = {row.path for row in ui.visible().tree_rows}
    assert paths >= {
        ("alpha  (2)", "p1  (1)", "mux/readout  (1)"),
        ("alpha  (2)", "p2  (1)", "mux/readout  (1)"),
        ("beta  (1)", "p1  (1)", "mux/readout  (1)"),
    }
    ui.screenshot(tmp_path / "initial-point-groups.png")


def navigation_paths(ui: ComponentDriver) -> set[tuple[str, ...]]:
    """Read visible navigation paths without the separate count decoration."""
    return {
        tuple(part.split("  (", 1)[0] for part in row.path)
        for row in ui.visible().tree_rows
        if row.tree_name == "results.navigation"
    }


def navigation_path(ui: ComponentDriver, path: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve semantic captions to their currently visible decorated path."""
    matches = [
        row.path
        for row in ui.visible().tree_rows
        if tuple(part.split("  (", 1)[0] for part in row.path) == path
    ]
    if len(matches) != 1:
        raise LookupError(f"{path!r}: {len(matches)} visible paths")
    return matches[0]


@pytest.mark.parametrize("point_first", [False, True])
def test_entry_tree_paths_keep_entries_isolated(
    populated_browser: ComponentDriver, tmp_path: Path, point_first: bool
) -> None:
    ui = populated_browser
    ui.select_combo(
        "Point → experiment" if point_first else "Experiment → point",
        object_name="results.group",
    )
    assert {path for path in navigation_paths(ui) if len(path) == 1} == {
        ("alpha",),
        ("beta",),
    }
    for entry, experiment, point, run_id, count in (
        ("alpha", "mux/readout", "p1", "20261021T010000Z-aaaaaa", 2),
        ("beta", "mux/readout", "p1", "20261021T010000Z-aaaaaa", 1),
    ):
        ui.select_tree_row(
            navigation_path(ui, (entry,)), object_name="results.navigation"
        )
        assert ui.observe(object_name="results.focus").text == f"{entry} · {count} runs"
        path = (entry,)
        for caption in (point, experiment) if point_first else (experiment, point):
            ui.expand_tree(navigation_path(ui, path), object_name="results.navigation")
            path = (*path, caption)
        ui.expand_tree(navigation_path(ui, path), object_name="results.navigation")
        path = (*path, "aaaaaa  ·  01:05:00")
        assert path in navigation_paths(ui)
        ui.select_tree_row(navigation_path(ui, path), object_name="results.navigation")
        assert ui.observe(object_name="results.focus").text == f"{run_id} · 1 runs"
        assert (
            f"{entry} / {point}" in ui.observe(object_name="results.figure.0").tooltip
        )
    ui.screenshot(tmp_path / f"entry-tree-{point_first}.png")


def test_group_gallery_has_one_representative_per_run(
    populated_browser: ComponentDriver, tmp_path: Path
) -> None:
    ui = populated_browser
    assert {
        w.object_name
        for w in ui.visible().widgets
        if w.object_name.startswith("results.figure.")
    } == {"results.figure.0", "results.figure.1", "results.figure.2"}
    assert not ui.scroll_area(object_name="results.gallery").vertical_scrollable
    ui.screenshot(tmp_path / "group-one-card-per-run.png")
    for path, count in (
        (("alpha",), 2),
        (("alpha", "p1"), 1),
        (("alpha", "p1", "mux/readout"), 1),
    ):
        ui.select_tree_row(navigation_path(ui, path), object_name="results.navigation")
        assert {
            w.object_name
            for w in ui.visible().widgets
            if w.object_name.startswith("results.figure.")
        } == {f"results.figure.{i}" for i in range(count)}
        ui.expand_tree(navigation_path(ui, path), object_name="results.navigation")


def test_group_image_click_opens_run_and_current_figure_in_one_step(
    populated_browser: ComponentDriver, tmp_path: Path
) -> None:
    ui = populated_browser
    ui.click(object_name="results.figure.0")
    assert (
        ui.observe(object_name="results.focus").text
        == "20261021T010000Z-aaaaaa · 1 runs"
    )
    assert (
        ui.observe(object_name="results.current_figure").text
        == "Current figure: plot-0.png"
    )
    assert ui.scroll_area(object_name="results.gallery").vertical_scrollable
    ui.click(object_name="results.figure.1")
    assert (
        ui.observe(object_name="results.current_figure").text
        == "Current figure: plot-1.png"
    )
    ui.click("Back")
    assert (
        ui.observe(object_name="results.current_figure").text
        == "Current figure: plot-0.png"
    )
    ui.click("Back")
    assert ui.observe(object_name="results.focus").text == "All results · 3 runs"
    ui.click("Forward")
    assert (
        ui.observe(object_name="results.focus").text
        == "20261021T010000Z-aaaaaa · 1 runs"
    )
    ui.screenshot(tmp_path / "direct-run-current-figure.png")


def test_run_image_opens_one_replaceable_tool_and_escape_closes_it(
    populated_browser: ComponentDriver,
    browser_host: MainWindow,
    tmp_path: Path,
) -> None:
    ui = populated_browser
    ui.click(object_name="results.figure.0")
    ui.click(object_name="results.figure.1")
    assert ui.observe(object_name="results.image_viewer").visible
    viewer = browser_host.findChild(QWidget, "results.image_viewer")
    assert viewer is not None
    assert viewer.windowFlags() & Qt.WindowType.Tool
    assert viewer.windowTitle() == "aaaaaa / plot-1.png"
    assert viewer.width() <= 400 and viewer.height() <= 220
    preview = ComponentDriver(viewer)
    preview.screenshot(tmp_path / "floating-second.png")
    preview.resize_window(300, 180)
    preview.screenshot(tmp_path / "floating-resized.png")
    ui.click(object_name="results.figure.0")
    assert browser_host.findChild(QWidget, "results.image_viewer") is viewer
    assert viewer.windowTitle() == "aaaaaa / plot-0.png"
    ui.press_key(Qt.Key.Key_Escape, object_name="results.image_viewer")
    assert not ui.observe(object_name="results.image_viewer").visible
    ui.screenshot(tmp_path / "floating-closed.png")
    ui.click(object_name="results.figure.0")
    assert viewer.isVisible()
    browser = browser_host.findChild(QWidget, "results.browser")
    assert browser is not None
    browser.close()
    assert not viewer.isVisible()


def test_history_reuses_images_and_restores_the_matching_gallery(
    populated_browser: ComponentDriver,
    browser_host: MainWindow,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ui = populated_browser
    ui.enter_text(value="Q2", object_name="results.search")
    ui.click("Filter by date (UTC)")
    ui.select_combo("Experiment → point", object_name="results.group")
    ui.screenshot(tmp_path / "history-00-all.png")
    ui.click(object_name="results.figure.0")
    ui.screenshot(tmp_path / "history-01-run.png")
    ui.click("Add to comparison")
    ui.click(object_name="results.figure.1")
    ui.screenshot(tmp_path / "history-02-second.png")
    reads: list[str] = []
    decodes: list[int] = []
    original_read = ResultsIndex.read_figure
    original_decode = QPixmap.loadFromData

    def read(index: ResultsIndex, key: str, selector: str) -> bytes:
        reads.append(selector)
        return original_read(index, key, selector)

    def decode(image: QPixmap, png: bytes) -> bool:
        decodes.append(len(png))
        return original_decode(image, png)

    monkeypatch.setattr(ResultsIndex, "read_figure", read)
    monkeypatch.setattr(QPixmap, "loadFromData", decode)
    for step, command, focus, current, cards in (
        (3, "Back", "20261021T010000Z-aaaaaa · 1 runs", "plot-0.png", 8),
        (4, "Forward", "20261021T010000Z-aaaaaa · 1 runs", "plot-1.png", 8),
        (5, "Back", "20261021T010000Z-aaaaaa · 1 runs", "plot-0.png", 8),
        (6, "Back", "All results · 3 runs", "none", 3),
        (7, "Forward", "20261021T010000Z-aaaaaa · 1 runs", "plot-0.png", 8),
    ):
        ui.click(command)
        assert ui.observe(object_name="results.focus").text == focus
        assert (
            ui.observe(object_name="results.current_figure").text
            == f"Current figure: {current}"
        )
        assert (
            len(
                [
                    child
                    for child in browser_host.findChildren(QWidget)
                    if child.objectName().startswith("results.card.")
                    and child.isVisible()
                ]
            )
            == cards
        )
        ui.screenshot(tmp_path / f"history-{step:02d}.png")
        assert reads == []
        assert decodes == []
    assert ui.observe(object_name="results.search").text == "Q2"
    assert ui.observe("Filter by date (UTC)").checked
    assert ui.observe(object_name="results.group").text == "Experiment → point"
    ui.select_tab("Comparison (1)")
    assert ui.observe(object_name="results.pin_count").text == "Pinned runs: 1"
    ui.select_tab("Gallery")
    ui.enter_text(value="no-such-tone", object_name="results.search")
    assert ui.observe(object_name="results.focus").text.endswith("0 runs")
    assert not any(
        w.object_name.startswith("results.figure.") for w in ui.visible().widgets
    )
    ui.enter_text(value="Q2", object_name="results.search")
    assert "plot-0.png" in ui.observe(object_name="results.figure.0").tooltip
    ui.screenshot(tmp_path / "history-08-filter-restored.png")
    assert reads == []
    assert decodes == []


def test_history_invalidates_a_changed_png_instead_of_showing_stale_pixels(
    populated_browser: ComponentDriver,
    browser_host: MainWindow,
    tmp_path: Path,
) -> None:
    ui = populated_browser
    ui.click(object_name="results.figure.0")
    path = scan_results(tmp_path).runs[0].figure_paths[0]
    changed = QImage(400, 220, QImage.Format.Format_RGB32)
    changed.fill(Qt.GlobalColor.magenta)
    assert changed.save(str(path))
    ui.click("Back")
    image = browser_host.findChild(QLabel, "results.figure.0")
    assert image is not None
    raster = image.pixmap()
    assert raster is not None and raster.toImage().pixelColor(0, 0).name() == "#ff00ff"
    ui.screenshot(tmp_path / "history-changed-png.png")


def test_filters_grouping_and_focus_history(
    populated_browser: ComponentDriver, tmp_path: Path
) -> None:
    ui = populated_browser
    assert ui.observe(object_name="results.focus").text == "All results · 3 runs"
    ui.select_combo("Experiment → point", object_name="results.group")
    ui.expand_tree(navigation_path(ui, ("alpha",)), object_name="results.navigation")
    assert ("alpha", "mux/readout") in navigation_paths(ui)
    ui.select_combo("Point → experiment", object_name="results.group")
    ui.expand_tree(navigation_path(ui, ("alpha",)), object_name="results.navigation")
    ui.expand_tree(navigation_path(ui, ("beta",)), object_name="results.navigation")
    assert navigation_paths(ui) >= {
        ("alpha", "p1"),
        ("alpha", "p2"),
        ("beta", "p1"),
    }
    ui.click("All results")
    ui.enter_text(value="Q2", object_name="results.search")
    ui.select_combo("mux/readout", object_name="results.experiment")
    ui.click("Filter by date (UTC)")
    ui.open_calendar(object_name="results.date_from")
    ui.choose_calendar_day(22)
    assert ui.observe(object_name="results.focus").text == "All results · 1 runs"
    ui.select_tree_row(
        navigation_path(ui, ("alpha", "p2")), object_name="results.navigation"
    )
    assert ui.observe(object_name="results.focus").text == "alpha / p2 · 1 runs"
    ui.select_combo("Experiment → point", object_name="results.group")
    assert ui.observe(object_name="results.focus").text == "alpha / p2 · 1 runs"
    ui.click("Back")
    assert ui.observe(object_name="results.focus").text == "All results · 1 runs"
    ui.click("Forward")
    assert ui.observe(object_name="results.focus").text == "alpha / p2 · 1 runs"
    ui.enter_text(value="no-such-tone", object_name="results.search")
    assert ui.observe(object_name="results.focus").text == "alpha / p2 · 0 runs"
    assert not any(
        w.object_name.startswith("results.figure.") for w in ui.visible().widgets
    )
    ui.enter_text(value="Q2", object_name="results.search")
    assert "alpha / p2" in ui.observe(object_name="results.figure.0").tooltip
    ui.screenshot(tmp_path / "filtered-focus.png")


def test_figure_drilldown_details_scroll_and_pins(
    populated_browser: ComponentDriver, tmp_path: Path, saved_cfg: JsonObject
) -> None:
    ui = populated_browser
    assert "results.gallery" in {w.object_name for w in ui.visible().widgets}
    ui.click(object_name="results.figure.0")
    ui.screenshot(tmp_path / "run-first-details.png")
    assert ui.scroll_area(object_name="results.gallery").vertical_scrollable
    assert "20261021T010000Z-aaaaaa" in ui.observe(object_name="results.focus").text
    ui.click(object_name="results.figure.1")
    assert (
        ui.observe(object_name="results.current_figure").text
        == "Current figure: plot-1.png"
    )
    ui.screenshot(tmp_path / "run-current-details.png")
    details = {
        row.columns
        for row in ui.visible().tree_rows
        if row.tree_name == "results.details"
    }
    for field in ("cfg_type", "schema_version"):
        assert (field, str(saved_cfg[field])) in details
    assert ("frequency", "5100.0") in details
    ui.select_tab("Analysis")
    rows = {
        row.columns
        for row in ui.visible().tree_rows
        if row.tree_name == "results.analysis"
    }
    assert ("freq", "5100.0") in rows
    assert ("Parameters · analysis.json", "") in rows and ("n_iter", "12") in rows
    ui.select_tab("Metadata / cfg")
    ui.click(object_name="results.details")
    ui.press_key(Qt.Key.Key_PageDown, object_name="results.details")
    ui.screenshot(tmp_path / "run-values-details.png")
    assert "alpha / p1" in ui.observe(object_name="results.figure.0").tooltip
    ui.invoke_context_action(action="Add to comparison", object_name="results.figure.1")
    ui.click("Add to comparison")
    ui.select_tab("Comparison (1)")
    assert ui.observe(object_name="results.pin_count").text == "Pinned runs: 1"
    assert len(ui.table_rows(object_name="results.comparison_table")) == 1
    ui.select_tab("Gallery")
    ui.invoke_context_action(action="Copy path", object_name="results.figure.1")
    assert "Copied path:" in ui.observe(object_name="results.feedback").text
    ui.invoke_context_action(action="Copy image", object_name="results.figure.1")
    assert ui.observe(object_name="results.feedback").text == "Copied image: plot-1.png"
    ui.screenshot(tmp_path / "run-details-pinned.png")
    ui.press_key(Qt.Key.Key_PageDown, object_name="results.gallery")
    assert ui.observe(object_name="results.figure.7").fully_visible
    ui.screenshot(tmp_path / "gallery-bottom.png")
    ui.select_tab("Comparison (1)")
    for _ in range(8):
        ui.press_key(Qt.Key.Key_PageUp, object_name="results.compare.images")
    ui.screenshot(tmp_path / "comparison-before-unpin.png")
    ui.click(object_name="results.compare.remove.0")
    assert ui.observe(object_name="results.pin_count").text == "Pinned runs: 0"


def test_run_without_images_is_reachable_from_tree(
    browser_host: MainWindow, tmp_path: Path
) -> None:
    (tmp_path / "Database" / "bare" / "runs" / "unknown").mkdir(parents=True)
    ui = ComponentDriver(browser_host)
    ui.select_menu("Results ▾", ("Results browser…",))
    path = ("bare",)
    for caption in (
        "Global / unknown",
        "Unknown experiment",
        "nknown  ·  Unknown time",
    ):
        ui.expand_tree(navigation_path(ui, path), object_name="results.navigation")
        path = (*path, caption)
    ui.select_tree_row(navigation_path(ui, path), object_name="results.navigation")
    assert ui.observe(object_name="results.focus").text == "unknown · 1 runs"
    assert ("registration", "Unregistered") in {
        row.columns
        for row in ui.visible().tree_rows
        if row.tree_name == "results.details"
    }
    assert (
        ui.observe(object_name="results.current_figure").text == "Current figure: none"
    )
    assert not any(
        w.object_name.startswith("results.figure.") for w in ui.visible().widgets
    )
    ui.screenshot(tmp_path / "no-images-tree-details.png")


def test_figure_only_run_explains_open_refusal(
    browser_host: MainWindow, tmp_path: Path
) -> None:
    run = tmp_path / "Database" / "orphan" / "runs" / "unknown"
    (run / "figures").mkdir(parents=True)
    image = QImage(100, 100, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.red)
    assert image.save(str(run / "figures" / "only.png"))
    ui = ComponentDriver(browser_host)
    ui.select_menu("Results ▾", ("Results browser…",))
    assert ui.observe(object_name="results.focus").text == "All results · 1 runs"
    assert "results.figure.0" in {w.object_name for w in ui.visible().widgets}
    ui.click(object_name="results.figure.0")
    button = ui.observe("Open in new analysis tab")
    assert not button.enabled
    assert "No data.h5" in button.tooltip
    assert ("registration", "Unregistered") in {
        row.columns
        for row in ui.visible().tree_rows
        if row.tree_name == "results.details"
    }
    ui.screenshot(tmp_path / "figure-only.png")


@pytest.fixture
def load_browser(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
    request: pytest.FixtureRequest,
    qapp: QApplication,
) -> Iterator[tuple[ComponentDriver, str]]:
    session, ctrl, window, _tab_id = live_ui
    variant = request.param
    if variant in {"missing", "pointless"}:
        tab = session.state.get_tab("tab")
        history = replace(
            session.metadata.snapshot, point="missing" if variant == "missing" else None
        )
        session.state.update_tab_result(
            "tab",
            replace(
                require_record(tab.run),
                metadata=session.metadata.model_copy(update={"snapshot": history}),
            ),
            parameters=tab.run.parameters,
        )
    session.save()
    image = QImage(400, 220, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.yellow)
    figures = session.native_data_path.parent / "figures"
    figures.mkdir()
    assert image.save(str(figures / "saved.png"))
    ctrl.setup_control.open_entry(EntryRequest("original"))
    if variant != "empty":
        ctrl.context_control.use_point(PointUseRequest("p"))
    if variant == "foreign":
        ctrl.setup_control.new_entry(EntryRequest("foreign"))
        ctrl.setup_control.new_point(
            PointCreateRequest("other", "", CommandOrigin("gui"))
        )
    ctrl.add_view(window)
    window.show()
    qapp.processEvents()
    before = set(ctrl.list_tab_ids())
    yield ComponentDriver(window), variant
    for tab_id in set(ctrl.list_tab_ids()) - before:
        ctrl.close_tab(tab_id)


@pytest.mark.parametrize(
    "load_browser",
    ["available", "empty", "foreign", "pointless", "missing"],
    indirect=True,
)
def test_new_analysis_tab_projects_the_same_load_source_reason(
    load_browser: tuple[ComponentDriver, str],
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    ui, variant = load_browser
    ui.select_menu("Results ▾", ("Results browser…",))
    ui.click(object_name="results.figure.0")
    ui.click("Open in new analysis tab")
    feedback = ui.observe(object_name="results.feedback").text
    assert qapp.activeWindow() is live_ui[2]
    tabs = ui.tabs("saving")
    assert tabs.current == tabs.captions[-1]
    if variant == "available":
        assert feedback.startswith("Loaded history:")
        assert "history.analysis_source" not in {
            w.object_name for w in ui.visible().widgets
        }
    else:
        expected = {
            "empty": "No active context is available",
            "foreign": "Historical entry differs from the opened entry",
            "pointless": "Historical metadata has no point",
            "missing": "Historical point 'missing' is unavailable",
        }[variant]
        assert (
            "History loaded, no analysis source:" in feedback and expected in feedback
        )
        tab_message = ui.observe(object_name="history.analysis_source")
        assert tab_message.visible and expected in tab_message.text
        assert "History loaded, no analysis source:" in tab_message.text
    ui.screenshot(tmp_path / f"load-{variant}.png")


@pytest.fixture
def identity_browser(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
    live_registry: Registry,
    qapp: QApplication,
) -> ComponentDriver:
    session, ctrl, window, _tab_id = live_ui
    live_registry.register("other", SavingAdapter)
    session.save()
    original = load(session.native_data_path)
    for name, experiment in (
        ("alias-run", "saving"),
        ("other-run", "other"),
        ("unknown-run", "old_tag"),
    ):
        run = session.native_data_path.parent.parent / name
        run.mkdir()
        metadata = session.metadata.model_copy(update={"experiment": experiment})
        docs = dict(original.docs)
        docs["metadata"] = metadata.model_dump(mode="json")
        save(run / "data.h5", replace(original, docs=docs))
        figures = run / "figures"
        figures.mkdir()
        image = QImage(120, 100, QImage.Format.Format_RGB32)
        image.fill(Qt.GlobalColor.cyan)
        assert image.save(str(figures / "saved.png"))
    ctrl.add_view(window)
    window.show()
    qapp.processEvents()
    ui = ComponentDriver(window)
    ui.select_menu("Results ▾", ("Results browser…",))
    return ui


def test_saved_experiment_names_stay_separate_from_registry_aliases(
    identity_browser: ComponentDriver, tmp_path: Path
) -> None:
    ui = identity_browser
    ui.select_combo("Experiment → point", object_name="results.group")
    assert navigation_paths(ui) >= {
        ("original", "saving"),
        ("original", "test/saving"),
        ("original", "other"),
        ("original", "old_tag"),
    }
    ui.select_combo("saving", object_name="results.experiment")
    assert ui.observe(object_name="results.focus").text == "All results · 1 runs"
    ui.screenshot(tmp_path / "saved-identity-groups.png")


def test_browsing_only_resolves_the_experiment_when_open_is_clicked(
    populated_browser: ComponentDriver,
    tmp_path: Path,
    qapp: QApplication,
) -> None:
    calls: list[str | None] = []

    def resolve(experiment: str | None) -> ExperimentResolution:
        calls.append(experiment)
        return Registry().resolve_experiment(experiment)

    def unexpected_open(_adapter: str, _path: str) -> LoadTabResultOutcome:
        pytest.fail("unresolved experiment cannot open a tab")

    browser = ResultsBrowser(
        scan_results(tmp_path), unexpected_open, resolve_experiment=resolve
    )
    browser.resize(1400, 850)
    browser.show()
    qapp.processEvents()
    ui = ComponentDriver(browser)
    try:
        ui.select_combo("Experiment → point", object_name="results.group")
        ui.click(object_name="results.figure.0")
        ui.click("Add to comparison")
        ui.select_tab("Comparison (1)")
        ui.select_tab("Gallery")
        ui.select_tab("Analysis")
        assert calls == []
        ui.click("Open in new analysis tab")
        assert calls == ["mux/readout"]
        assert (
            "No load-capable adapter" in ui.observe(object_name="results.feedback").text
        )
        ui.screenshot(tmp_path / "open-only-resolution.png")
    finally:
        browser.close()
        browser.deleteLater()
        qapp.processEvents()


def test_unresolved_identity_remains_browsable_and_refuses_new_tab(
    identity_browser: ComponentDriver,
    tmp_path: Path,
) -> None:
    ui = identity_browser
    ui.select_combo("Experiment → point", object_name="results.group")
    ui.expand_tree(navigation_path(ui, ("original",)), object_name="results.navigation")
    ui.select_tree_row(
        navigation_path(ui, ("original", "old_tag")), object_name="results.navigation"
    )
    ui.click(object_name="results.figure.0")
    assert ("experiment", "old_tag") in {
        row.columns
        for row in ui.visible().tree_rows
        if row.tree_name == "results.details"
    }
    ui.click("Open in new analysis tab")
    feedback = ui.observe(object_name="results.feedback").text
    assert "old_tag" in feedback and "No load-capable adapter" in feedback
    assert ui.tabs("saving").captions == ("saving",)
    ui.screenshot(tmp_path / "unknown-identity.png")


def test_ambiguous_tag_is_browsable_but_does_not_choose_an_adapter(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
    live_registry: Registry,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    session, ctrl, window, _tab_id = live_ui
    live_registry.register("alternative", AnalysisSavingAdapter)
    session.save()
    figures = session.native_data_path.parent / "figures"
    figures.mkdir()
    image = QImage(120, 100, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.cyan)
    assert image.save(str(figures / "saved.png"))
    ctrl.add_view(window)
    window.show()
    qapp.processEvents()
    ui = ComponentDriver(window)
    ui.select_menu("Results ▾", ("Results browser…",))
    ui.click(object_name="results.figure.0")
    assert ("experiment", "test/saving") in {
        row.columns
        for row in ui.visible().tree_rows
        if row.tree_name == "results.details"
    }
    ui.click("Open in new analysis tab")
    feedback = ui.observe(object_name="results.feedback").text
    assert "Multiple load-capable adapters" in feedback
    assert all(name in feedback for name in ("saving", "alternative", "test/saving"))
    assert ui.tabs("saving").captions == ("saving",)
    ui.screenshot(tmp_path / "ambiguous-identity.png")


@pytest.mark.parametrize("load_browser", ["available"], indirect=True)
@pytest.mark.parametrize("rejection", ["capability", "tag", "format"])
def test_browser_keeps_existing_load_admission_and_failed_tab_cleanup(
    load_browser: tuple[ComponentDriver, str],
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
    live_registry: Registry,
    rejection: str,
    tmp_path: Path,
) -> None:
    ui, _variant = load_browser
    session, _ctrl, _window, _tab_id = live_ui
    if rejection == "capability":
        live_registry.register("test/saving", SavingAdapter)
    elif rejection == "tag":
        live_registry.register("test/saving", WrongTagAdapter)
    ui.select_menu("Results ▾", ("Results browser…",))
    if rejection == "format":
        # A file can change after indexing; Load, not the browser, admits bytes.
        session.native_data_path.write_bytes(b"not a native container")
    ui.click(object_name="results.figure.0")
    ui.click("Open in new analysis tab")
    feedback = ui.observe(object_name="results.feedback").text
    expected = {
        "capability": "does not support loading data",
        "tag": "Expected experiment tag",
        "format": "Could not read the data file",
    }[rejection]
    assert "Cannot open analysis tab:" in feedback and expected in feedback
    assert ui.tabs("saving").captions == ("saving",)
    ui.screenshot(tmp_path / f"load-rejected-{rejection}.png")


class WrongTagAdapter(AnalysisSavingAdapter):
    """Expose a distinct declared tag to exercise the existing Load admission."""

    experiment_tag: ClassVar[str | None] = "different-tag"
