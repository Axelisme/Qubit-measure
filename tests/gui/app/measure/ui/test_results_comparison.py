"""Drive the saved-summary comparison table and aligned all-PNG run columns."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from PyQt6.QtGui import QColor, QImage, QPainter
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QScrollArea, QSplitter, QTableWidget
from zcu_tools.gui.app.measure.services.results_index import ResultsIndex
from zcu_tools.gui.app.measure.ui.results_comparison import ResultsComparison

from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._comparison_support import saved_run


@pytest.fixture
def comparison(qapp: QApplication, tmp_path: Path) -> Iterator[ResultsComparison]:
    runs = []
    for number in range(6):
        run = saved_run(
            tmp_path,
            f"run-{number}",
            "archived/arbitrary",
            {
                "tones": [
                    {
                        "name": "Q5",
                        "t1": 21.25,
                        "fit_quality": {"decay": {"r2": 0.03, "invalid": []}},
                    },
                    {"label": "Q9", "t1": 8, "failure": "saved failure"},
                ]
            },
        )
        if number != 4:
            folder = run.run_path / "figures"
            folder.mkdir()
            names = (
                ("first.png", "other.png")
                if number == 5
                else ("first.png", "selected.png")
            )
            paths = tuple(folder / name for name in names)
            for path, colour in zip(paths, ("navy", "darkred"), strict=True):
                image = QImage(480, 300, QImage.Format.Format_RGB32)
                image.fill(QColor("white"))
                painter = QPainter(image)
                painter.setPen(QColor("black"))
                painter.drawText(120, 25, f"Saved trace · {path.stem}")
                painter.drawText(210, 285, "Delay")
                painter.drawText(10, 150, "Signal")
                painter.drawLine(65, 250, 450, 250)
                painter.drawLine(65, 250, 65, 45)
                painter.setPen(QColor(colour))
                for start, end in (
                    ((65, 60), (140, 130)),
                    ((140, 130), (220, 180)),
                    ((220, 180), (300, 210)),
                    ((300, 210), (440, 235)),
                ):
                    painter.drawLine(*start, *end)
                painter.end()
                assert image.save(str(path))
            run = replace(run, figure_paths=paths)
        runs.append(run)
    widget = ResultsComparison(ResultsIndex(tmp_path, tuple(runs)))
    for run in runs:
        widget.pin_run(run)
    widget.resize(1400, 900)
    widget.show()
    qapp.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


def test_pins_table_remove_and_clear(
    comparison: ResultsComparison, tmp_path: Path
) -> None:
    ui = ComponentDriver(comparison)
    assert ui.observe(object_name="results.pin_count").text == "Pinned runs: 6"
    rows = ui.table_rows(object_name="results.comparison_table")
    assert comparison.findChildren(QTableWidget)[0].rowCount() == 6
    assert "0: 21.25\n1: 8" in rows[0].columns
    ui.screenshot(tmp_path / "comparison-table.png")
    ui.click(object_name="results.compare.remove.2")
    assert ui.observe(object_name="results.pin_count").text == "Pinned runs: 5"
    ui.screenshot(tmp_path / "comparison-unpinned.png")
    ui.click("Clear comparison")
    assert ui.observe(object_name="results.pin_count").text == "Pinned runs: 0"
    assert ui.table_rows(object_name="results.comparison_table") == ()
    ui.screenshot(tmp_path / "comparison-cleared.png")


def test_all_run_columns_align_png_rows_and_missing_members(
    comparison: ResultsComparison, tmp_path: Path
) -> None:
    ui = ComponentDriver(comparison)
    ui.resize_window(1000, 850)
    assert ui.observe(object_name="results.compare.row.0").text == "first.png"
    assert ui.observe(object_name="results.compare.row.1").text == "selected.png"
    assert ui.observe(object_name="results.compare.row.2").text == "other.png"
    assert ui.observe(object_name="results.compare.missing.2.0").text == "—"
    assert "run-0" in ui.observe(object_name="results.compare.header.0").text
    assert ui.observe(object_name="results.compare.missing.0.4").text == "—"
    scroll = comparison.findChildren(QScrollArea)[0]
    bar = scroll.horizontalScrollBar()
    assert bar is not None and bar.maximum() > 0
    assert comparison.findChildren(QTableWidget)[0].rowCount() == 6
    ui.screenshot(tmp_path / "comparison-all-images.png")
    splitter = comparison.findChild(QSplitter, "results.compare.splitter")
    assert splitter is not None
    sizes = splitter.sizes()
    ui.drag_splitter(70, object_name="results.compare.splitter")
    assert splitter.sizes()[0] > sizes[0]
    activated: list[str] = []
    comparison.run_activated.connect(activated.append)
    ui.click(object_name="results.compare.header.0", double_click=True)
    assert activated == [comparison.pinned_runs()[0].key]
    ui.screenshot(tmp_path / "comparison-splitter-dragged.png")


def test_unknown_summary_value_is_visible(qapp: QApplication, tmp_path: Path) -> None:
    run = saved_run(
        tmp_path, "historical", "deleted/arbitrary", {"unexpected": {"leaf": 19.75}}
    )
    widget = ResultsComparison(ResultsIndex(tmp_path, (run,)))
    widget.pin_run(run)
    widget.resize(1400, 700)
    widget.show()
    qapp.processEvents()
    ui = ComponentDriver(widget)
    try:
        ui.screenshot(tmp_path / "comparison-neutral-summary.png")
        rows = ui.table_rows(object_name="results.comparison_table")
        assert rows[0].columns == (
            "sample",
            "historical",
            "deleted/arbitrary",
            "—",
            "19.75",
        )
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


def test_table_drag_and_horizontal_navigation_reveal_summary_cells(
    comparison: ResultsComparison, tmp_path: Path
) -> None:
    ui = ComponentDriver(comparison)
    ui.resize_window(850, 900)
    ui.select_table_row("run-0", object_name="results.comparison_table")
    before = ui.table_rows(object_name="results.comparison_table")[0]
    assert before.columns[7] == "0: []\n1: —"
    assert before.columns[-1] == ""
    ui.screenshot(tmp_path / "comparison-before-column-drag.png")
    ui.drag_column(0, 160, object_name="results.comparison_table")
    ui.screenshot(tmp_path / "comparison-after-column-drag.png")
    after = ui.table_rows(object_name="results.comparison_table")[0]
    assert after.columns[7] == ""
    ui.press_key(
        Qt.Key.Key_End,
        object_name="results.comparison_table",
        modifiers=Qt.KeyboardModifier.ControlModifier,
    )
    scrolled = ui.table_rows(object_name="results.comparison_table")
    assert scrolled[-1].columns[-1] == "0: —\n1: saved failure"
    assert scrolled[-1].columns[0] == ""
    ui.screenshot(tmp_path / "comparison-horizontal-end.png")


def test_pin_api_is_atomic_and_preserves_insertion_order(
    qapp: QApplication, tmp_path: Path
) -> None:
    run = saved_run(tmp_path, "one", "t1", {})
    run = replace(
        run, figure_paths=(run.run_path / "first.png", run.run_path / "second.png")
    )
    widget = ResultsComparison(ResultsIndex(tmp_path, (run,)))
    changes = []
    widget.pins_changed.connect(lambda: changes.append("changed"))
    widget.pin_run(run)
    widget.pin_run(run)
    assert widget.pinned_runs() == (run,)
    with pytest.raises(KeyError):
        widget.pin_run(replace(run, key="unknown"))
    assert widget.pinned_runs() == (run,)
    assert len(changes) == 1
    widget.deleteLater()
    qapp.processEvents()
