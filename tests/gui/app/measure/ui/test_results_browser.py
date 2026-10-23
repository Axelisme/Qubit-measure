"""Results browsing through visible controls over synthetic Database content."""

from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import ClassVar
from uuid import UUID, uuid4

import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import QApplication, QWidget
from zcu_tools.datafile import Container, ContainerAttrs, JsonObject, load, save
from zcu_tools.experiment.metadata import RunMetadata, RunSnapshot, SoftwareProvenance
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.results_index import ResultsIndex, scan_results
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.control_identity import ControlIdentity
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
    PointUseRequest,
)
from zcu_tools.resources.entry import AcquiredPayload, LedgerEvent, ResultEntry

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._native_save_support import (
    AnalysisSavingAdapter,
    NativeSaveSession,
    SavingAdapter,
)
from tests.gui.app.measure._record_support import require_record
from tests.gui.app.measure.ui._main_window_fakes import apply_window_defaults, mock_ctrl


@pytest.fixture
def browser_host(
    qapp: QApplication,
    tmp_path: Path,
    main_window_control_factory: Callable[[], ControlIdentity],
) -> Iterator[MainWindow]:
    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    ctrl.query_results.side_effect = lambda: scan_results(tmp_path)
    ctrl.resolve_experiment.side_effect = Registry().resolve_experiment
    window = MainWindow(
        ctrl, templates=lambda: {}, dialog_presenter=RecordingDialogPresenter()
    )
    window.show()
    qapp.processEvents()
    yield window
    window.deleteLater()
    qapp.processEvents()


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


@pytest.fixture
def saved_cfg() -> JsonObject:
    return {
        "values": {
            "readout": {"frequency": 5100.0, "tone": "Q2"},
            "deep": {"level": {"branch": {"leaf": "bounded"}}},
            "long": "x" * 240,
        },
        "cfg_type": "synthetic.history.ReadoutCfg",
        "schema_version": "7.3",
    }


@pytest.fixture
def populated_browser(
    browser_host: MainWindow, tmp_path: Path, saved_cfg: JsonObject
) -> ComponentDriver:
    for entry_name, point, run_id, experiment, day in (
        ("alpha", "p1", "20261021T010000Z-aaaaaa", "mux/readout", "21"),
        ("alpha", "p2", "20261022T010000Z-bbbbbb", "mux/readout", "22"),
        ("beta", "p1", "20261021T010000Z-aaaaaa", "mux/readout", "23"),
    ):
        entry_path = tmp_path / "result" / entry_name
        entry = (
            ResultEntry.open(
                entry_name,
                result_root=tmp_path / "result",
                database_root=tmp_path / "Database",
            )
            if entry_path.exists()
            else ResultEntry.create(
                entry_name,
                result_root=tmp_path / "result",
                database_root=tmp_path / "Database",
            )
        )
        run = tmp_path / "Database" / entry_name / "runs" / run_id
        (run / "figures").mkdir(parents=True)
        metadata = RunMetadata(
            run_id=run_id,
            experiment=experiment,
            started_at=f"2026-10-{day}T01:00:00Z",
            finished_at=None,
            completion="unknown",
            snapshot=RunSnapshot(
                entry_id=UUID(entry.entry_id),
                entry_name=entry_name,
                point=point,
                description=None,
                roles={"qubit": "Q1"},
            ),
            provenance=SoftwareProvenance(
                software_versions={},
                git_commit=None,
                git_dirty=None,
                qick_version=None,
                soc_fingerprint=None,
                hostname=None,
            ),
        )
        save(
            run / "data.h5",
            Container(
                vars={},
                docs={
                    "metadata": metadata.model_dump(mode="json"),
                    "cfg": saved_cfg,
                },
                groups={},
                attrs=ContainerAttrs("3.0", "synthetic-browser", None, {}),
            ),
        )
        (run / "analysis").mkdir()
        (run / "analysis" / "analysis.json").write_text(
            '{"format":"zcu.analysis-result","format_version":"1.0",'
            '"pane":"analysis","parameters":{"n_iter":12},'
            '"summary":{"tones":[{"name":"Q2","freq":5100.0}]}}',
            encoding="utf-8",
        )
        for i in range(8 if point == "p1" else 1):
            image = QImage(400, 220, QImage.Format.Format_RGB32)
            image.fill(Qt.GlobalColor.cyan if i % 2 else Qt.GlobalColor.yellow)
            assert image.save(str(run / "figures" / f"plot-{i}.png"))
        if point == "p1":
            entry.ledger.append(
                LedgerEvent(
                    id=str(uuid4()),
                    entry_id=entry.entry_id,
                    kind="acquired",
                    at="2026-10-21T01:05:00Z",
                    origin="notebook",
                    payload=AcquiredPayload(
                        run_id=run_id,
                        tab="Synthetic",
                        experiment=experiment,
                        point=point,
                        roles={"qubit": "Q1"},
                    ),
                )
            )
    ui = ComponentDriver(browser_host)
    ui.select_menu("Results ▾", ("Results browser…",))
    window = browser_host.findChild(QWidget, "results.browser")
    return ComponentDriver(window)


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
    ui.click("UTC dates")
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
    assert ui.observe("UTC dates").checked
    ui.select_tab("Comparison")
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
        row.columns == ("tones[0].freq", "5100.0")
        for row in ui.table_rows(object_name="results.analysis")
    )
    ui.screenshot(tmp_path / "analysis-tab.png")
    ui.select_tab("Comparison")
    assert ui.observe(object_name="results.pin_count").text == "Pinned runs: 1"
    assert ui.table_rows(object_name="results.comparison_table")
    ui.screenshot(tmp_path / "comparison-tab.png")
    ui.select_tab("Gallery")
    assert ui.observe(object_name="results.figure.0").visible


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
        ui.invoke_context_action(
            action="Add to comparison", object_name=f"results.figure.{number}"
        )
    ui.select_tab("Comparison")
    ui.click("2 × 2 figures")
    ui.resize_window(1280, 800)
    small_compare = ui.observe(object_name="results.compare.image.0").bounds
    ui.screenshot(tmp_path / "comparison-1280.png")
    ui.resize_window(1920, 1200)
    large_compare = ui.observe(object_name="results.compare.image.0").bounds
    assert small_compare is not None and large_compare is not None
    assert large_compare[2] > small_compare[2]
    assert large_compare[3] > small_compare[3]
    assert abs(large_compare[3] / large_compare[2] - 220 / 400) < 0.02
    ui.screenshot(tmp_path / "comparison-1920.png")


def test_date_range_can_be_enabled_edited_and_disabled(
    populated_browser: ComponentDriver, tmp_path: Path
) -> None:
    ui = populated_browser
    assert ui.observe(object_name="results.date_from").kind == "QDateEdit"
    assert not ui.observe(object_name="results.date_from").enabled
    ui.click("UTC dates")
    ui.click(object_name="results.date_from")
    ui.press_key(Qt.Key.Key_End, object_name="results.date_from")
    ui.press_key(Qt.Key.Key_Up, object_name="results.date_from")
    ui.press_key(Qt.Key.Key_Return, object_name="results.date_from")
    assert ui.observe(object_name="results.date_from").text == "2026-10-22"
    assert ui.observe(object_name="results.visible_count").text == "1 / 3 runs"
    ui.screenshot(tmp_path / "date-filter.png")
    ui.click("UTC dates")
    assert ui.observe(object_name="results.visible_count").text == "3 / 3 runs"


def test_comparison_double_click_joins_focus_history(
    populated_browser: ComponentDriver, tmp_path: Path
) -> None:
    ui = populated_browser
    ui.invoke_context_action(action="Add to comparison", object_name="results.figure.0")
    ui.select_tab("Comparison")
    ui.select_table_row(
        "alpha", object_name="results.comparison_table", double_click=True
    )
    assert ui.tabs("Gallery").current == "Gallery"
    assert "aaaaaa · 1 runs" in ui.observe(object_name="results.focus").text
    ui.click("Back")
    assert ui.observe(object_name="results.focus").text == "All results · 3 runs"
    ui.click("Forward")
    assert "aaaaaa · 1 runs" in ui.observe(object_name="results.focus").text
    ui.select_tab("Comparison")
    assert ui.observe(object_name="results.pin_count").text == "Pinned runs: 1"
    ui.screenshot(tmp_path / "comparison-return-history.png")


def test_cfg_summary_bounds_mapping_depth_and_value_length(
    populated_browser: ComponentDriver, tmp_path: Path
) -> None:
    ui = populated_browser
    ui.click(object_name="results.figure.0")
    details = ui.observe(object_name="results.details").text
    assert 'deep.level.branch = {"leaf": "bounded"}' in details
    assert f"long = {'x' * 200} …" in details
    ui.screenshot(tmp_path / "bounded-cfg.png")


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
    ui.click("UTC dates")
    ui.click(object_name="results.date_from")
    ui.press_key(Qt.Key.Key_End, object_name="results.date_from")
    ui.press_key(Qt.Key.Key_Up, object_name="results.date_from")
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
    details = ui.observe(object_name="results.details").text
    for field in ("cfg_type", "schema_version"):
        assert f"{field} = {saved_cfg[field]}" in details
    assert "readout.frequency = 5100.0" in details
    ui.select_tab("Analysis")
    rows = {row.columns for row in ui.table_rows(object_name="results.analysis")}
    assert ("tones[0].freq", "5100.0") in rows
    assert ("Parameters · analysis.json", "") in rows and ("n_iter", "12") in rows
    ui.select_tab("Metadata / cfg")
    ui.click(object_name="results.details")
    ui.press_key(Qt.Key.Key_PageDown, object_name="results.details")
    ui.screenshot(tmp_path / "run-values-details.png")
    assert "alpha / p1" in ui.observe(object_name="results.figure.0").tooltip
    ui.invoke_context_action(action="Add to comparison", object_name="results.figure.1")
    ui.click("Add to comparison")
    ui.select_tab("Comparison")
    assert ui.observe(object_name="results.pin_count").text == "Pinned runs: 1"
    assert any("20261021T010000Z-aaaaaa" in row.text for row in ui.list_rows())
    ui.select_tab("Gallery")
    ui.invoke_context_action(action="Copy path", object_name="results.figure.1")
    assert "Copied path:" in ui.observe(object_name="results.feedback").text
    ui.invoke_context_action(action="Copy image", object_name="results.figure.1")
    assert ui.observe(object_name="results.feedback").text == "Copied image: plot-1.png"
    ui.screenshot(tmp_path / "run-details-pinned.png")
    ui.press_key(Qt.Key.Key_PageDown, object_name="results.gallery")
    assert ui.observe(object_name="results.figure.7").fully_visible
    ui.screenshot(tmp_path / "gallery-bottom.png")
    ui.select_tab("Comparison")
    ui.press_key(Qt.Key.Key_Home, object_name="results.pins")
    ui.click("Unpin selected")
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
    assert "Unregistered" in ui.observe(object_name="results.details").text
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
    assert "Unregistered" in ui.observe(object_name="results.details").text
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
    tmp_path: Path,
) -> None:
    ui, variant = load_browser
    ui.select_menu("Results ▾", ("Results browser…",))
    ui.click(object_name="results.figure.0")
    ui.click("Open in new analysis tab")
    feedback = ui.observe(object_name="results.feedback").text
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


def test_adapter_identity_groups_alias_and_tag_and_filters_both(
    identity_browser: ComponentDriver, tmp_path: Path
) -> None:
    ui = identity_browser
    ui.select_combo("Experiment → point", object_name="results.group")
    ui.expand_tree(navigation_path(ui, ("original",)), object_name="results.navigation")
    assert navigation_paths(ui) >= {
        ("original", "saving"),
        ("original", "other"),
        ("original", "old_tag"),
    }
    ui.select_tree_row(
        navigation_path(ui, ("original", "saving")), object_name="results.navigation"
    )
    assert ui.observe(object_name="results.focus").text == "saving · 2 runs"
    ui.click("All results")
    ui.select_combo("saving", object_name="results.experiment")
    assert ui.observe(object_name="results.focus").text == "All results · 2 runs"
    ui.select_combo("Point → experiment", object_name="results.group")
    ui.expand_tree(navigation_path(ui, ("original",)), object_name="results.navigation")
    ui.select_tree_row(
        navigation_path(ui, ("original", "p")), object_name="results.navigation"
    )
    ui.expand_tree(
        navigation_path(ui, ("original", "p")), object_name="results.navigation"
    )
    assert navigation_paths(ui) == {
        ("original",),
        ("original", "p"),
        ("original", "p", "saving"),
    }
    assert ui.observe(object_name="results.focus").text == "original / p · 2 runs"
    ui.select_combo("All experiments", object_name="results.experiment")
    assert navigation_paths(ui) >= {
        ("original", "p", "saving"),
        ("original", "p", "other"),
        ("original", "p", "old_tag"),
    }
    ui.screenshot(tmp_path / "identity-groups.png")


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
    assert "old_tag" in ui.observe(object_name="results.details").text
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
    assert "test/saving" in ui.observe(object_name="results.details").text
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
