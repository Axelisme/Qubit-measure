"""Readout calibration editors accept visible keyboard and mouse input."""

from pathlib import Path

from qtpy.QtWidgets import QApplication

from ._classifier_ui import run_scenarios


def test_classifier_editor_scenarios(qapp: QApplication, tmp_path: Path) -> None:
    run_scenarios(qapp, tmp_path)
