"""Standalone envelope selection is observable through the shared Form editor."""

from pathlib import Path

from qtpy.QtWidgets import QApplication

from ._waveform_ui import run_scenario


def test_standalone_envelope_editor(qapp: QApplication, tmp_path: Path) -> None:
    run_scenario(qapp, tmp_path)
