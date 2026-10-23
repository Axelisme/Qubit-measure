"""Drive pinned comparison tables and the four-run image selection."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPainter
from qtpy.QtWidgets import QApplication
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
            paths = tuple(folder / name for name in ("first.png", "selected.png"))
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
        widget.pin_run(run, "selected.png" if run.figure_paths else None)
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
    assert len(rows) == 6
    assert "Q5: 21.25\nQ9: 8" in rows[0].columns
    ui.screenshot(tmp_path / "comparison-table.png")
    ui.select_table_row("run-2", object_name="results.comparison_table")
    ui.click("Unpin selected")
    assert ui.observe(object_name="results.pin_count").text == "Pinned runs: 5"
    assert not any(
        row.selected for row in ui.table_rows(object_name="results.comparison_table")
    )
    ui.screenshot(tmp_path / "comparison-unpinned.png")
    ui.click("Clear comparison")
    assert ui.observe(object_name="results.pin_count").text == "Pinned runs: 0"
    assert ui.table_rows(object_name="results.comparison_table") == ()
    ui.screenshot(tmp_path / "comparison-cleared.png")


def test_four_image_selection_keeps_all_pins_and_exact_png(
    comparison: ResultsComparison, tmp_path: Path
) -> None:
    ui = ComponentDriver(comparison)
    ui.click("2 × 2 figures")
    assert ui.observe(object_name="results.compare.caption.0").text.endswith(
        "selected.png"
    )
    assert ui.observe(object_name="results.compare.caption.3").text.endswith(
        "selected.png"
    )
    ui.screenshot(tmp_path / "comparison-four-images.png")
    ui.select_list_row("sample / run-4")
    ui.press_key(Qt.Key.Key_Space, object_name="results.pins")
    assert "at most 4" in ui.observe(object_name="results.compare.feedback").text
    ui.screenshot(tmp_path / "comparison-limit-rejected.png")
    ui.select_list_row("sample / run-0")
    ui.press_key(Qt.Key.Key_Space, object_name="results.pins")
    ui.select_list_row("sample / run-4")
    ui.press_key(Qt.Key.Key_Space, object_name="results.pins")
    assert "No saved PNG" in ui.observe(object_name="results.compare.caption.3").text
    assert ui.observe(object_name="results.pin_count").text == "Pinned runs: 6"
    ui.screenshot(tmp_path / "comparison-no-png.png")
    ui.click("Comparison table")
    assert len(ui.table_rows(object_name="results.comparison_table")) == 6
    ui.click("2 × 2 figures")
    assert "No saved PNG" in ui.observe(object_name="results.compare.caption.3").text
    ui.screenshot(tmp_path / "comparison-mode-retained.png")


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
        ui.click("Unpin selected")
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
    assert widget.selected_figure(run.key) == "first.png"
    widget.pin_run(run, "second.png")
    widget.pin_run(run)
    assert widget.pinned_runs() == (run,)
    assert widget.selected_figure(run.key) == "second.png"
    with pytest.raises(KeyError):
        widget.pin_run(run, "missing.png")
    with pytest.raises(KeyError):
        widget.pin_run(replace(run, key="unknown"))
    with pytest.raises(KeyError):
        widget.selected_figure("unknown")
    assert widget.selected_figure(run.key) == "second.png"
    assert len(changes) == 2
    widget.deleteLater()
    qapp.processEvents()
