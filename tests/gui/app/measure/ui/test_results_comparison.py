"""Drive the fixed comparison table and aligned all-PNG run columns."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from PyQt6.QtGui import QColor, QImage, QPainter
from qtpy.QtWidgets import QApplication, QScrollArea, QSplitter, QTableWidget
from zcu_tools.datafile import JsonObject
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
            "mux/t1",
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
    assert "Q5: 21.25\nQ9: 8" in rows[0].columns
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


@pytest.mark.parametrize(
    ("experiment", "summary", "expected", "fact", "scenario"),
    [
        (
            "twotone/t1",
            {"t1": 21.25, "fit_quality": {"decay": {"r2": 0.01, "invalid": []}}},
            "21.25",
            "decay: 0.01",
            "flat",
        ),
        (
            "mux/onetone",
            {
                "resonators": [
                    {
                        "label": "R2",
                        "freq": 6.125,
                        "fit_quality": {"peak": {"r2": 0.8}},
                    },
                    {"name": "R8", "freq": 7.25, "fit_quality": {"peak": {"r2": 0.3}}},
                ]
            },
            "R2: 6.125\nR8: 7.25",
            "R2 / peak: 0.8\nR8 / peak: 0.3",
            "resonators",
        ),
        (
            "unknown/custom",
            {"t1": 123, "failure": "saved failure"},
            "—",
            "saved failure",
            "unknown",
        ),
    ],
    ids=("flat", "resonators", "unknown"),
)
def test_saved_summary_cells_are_visible(
    qapp: QApplication,
    tmp_path: Path,
    experiment: str,
    summary: JsonObject,
    expected: str,
    fact: str,
    scenario: str,
) -> None:
    run = saved_run(tmp_path, "historical", experiment, summary)
    widget = ResultsComparison(ResultsIndex(tmp_path, (run,)))
    widget.pin_run(run)
    widget.resize(1400, 700)
    widget.show()
    qapp.processEvents()
    ui = ComponentDriver(widget)
    try:
        rows = ui.table_rows(object_name="results.comparison_table")
        assert len(rows) == 1
        assert expected in rows[0].columns
        assert fact in rows[0].columns
        ui.select_table_row("historical", object_name="results.comparison_table")
        ui.screenshot(tmp_path / f"comparison-{scenario}.png")
        ui.click("Clear comparison")
        assert ui.observe(object_name="results.pin_count").text == "Pinned runs: 0"
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


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
