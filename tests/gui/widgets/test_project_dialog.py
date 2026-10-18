"""Discovery failure diagnostics through the shared ProjectDialog boundary."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication, QComboBox, QPushButton
from zcu_tools.gui.project import ProjectInfo
from zcu_tools.gui.result_scope import write_params_identity
from zcu_tools.gui.widgets.project_dialog import ProjectDialog


def _discovery_controls(widget: ProjectDialog) -> tuple[QComboBox, QPushButton]:
    picker = widget.findChild(QComboBox)
    assert picker is not None
    refresh = next(
        button
        for button in widget.findChildren(QPushButton)
        if button.toolTip() == "Refresh result scopes"
    )
    return picker, refresh


@pytest.mark.parametrize("error_type", [OSError, RuntimeError])
@pytest.mark.parametrize("named", [True, False], ids=["named", "unnamed"])
def test_discovery_failures_keep_tracebacks_and_recover_without_changing_paths(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error_type: type[Exception],
    named: bool,
) -> None:
    params_path = tmp_path / "result" / "ChipA" / "Q1" / "params.json"
    write_params_identity(params_path, chip_name="ChipA", qub_name="Q1")
    project = ProjectInfo(
        chip_name="ChipA" if named else "",
        qub_name="Q1" if named else "",
        result_dir=str(tmp_path / "custom-result"),
        data_root=str(tmp_path / "custom-database"),
        root_dir=str(tmp_path),
    )
    initial_error, manual_error = (
        error_type("initial directory enumeration failed"),
        error_type("manual directory enumeration failed"),
    )

    def fail_initial_scan(path: Path, pattern: str) -> Iterator[Path]:
        raise initial_error

    def fail_manual_scan(path: Path, pattern: str) -> Iterator[Path]:
        raise manual_error

    with caplog.at_level(
        logging.WARNING, logger="zcu_tools.gui.widgets.project_dialog"
    ):
        with monkeypatch.context() as fault:
            fault.setattr(Path, "rglob", fail_initial_scan)
            widget = ProjectDialog(project)
        try:
            picker, refresh = _discovery_controls(widget)
            fallback = "(new generated scope)" if named else "(no result scopes found)"
            assert picker.count() == 1
            assert picker.currentText() == fallback
            assert picker.currentData() is None
            assert picker.toolTip() == ""  # Initial discovery failure stays silent.
            assert widget.result_project() == project

            # A real filesystem scan after context teardown proves recovery.
            refresh.click()
            scope_id = str(params_path.parent.resolve())
            assert picker.findData(scope_id) >= 0
            assert picker.toolTip() == ""
            assert widget.result_project() == project

            with monkeypatch.context() as fault:
                fault.setattr(Path, "rglob", fail_manual_scan)
                refresh.click()
            assert picker.count() == 1
            assert picker.currentText() == fallback
            assert picker.currentData() is None
            assert picker.toolTip() == (
                "Failed to refresh result scopes: manual directory enumeration failed"
            )
            assert widget.result_project() == project

            refresh.click()
            assert picker.findData(scope_id) >= 0
            assert picker.toolTip() == ""
            assert widget.result_project() == project

            records = [
                record
                for record in caplog.records
                if record.name == "zcu_tools.gui.widgets.project_dialog"
            ]
            assert len(records) == 2
            for record, error in zip(
                records, (initial_error, manual_error), strict=True
            ):
                assert record.levelno == logging.WARNING
                assert record.getMessage() == (
                    f"ProjectDialog: failed to list result scopes: {error}"
                )
                assert record.exc_info is not None, (
                    "Discovery warning lost its traceback"
                )
                assert record.exc_info[0] is error_type
                assert record.exc_info[1] is error
                assert record.exc_info[2] is not None
        finally:
            widget.close()
            widget.deleteLater()
            qapp.processEvents()
