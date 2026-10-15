"""User selection and local ledger filters at the Qt Load dialog seam."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QMetaObject, Qt, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QLineEdit,
    QPushButton,
    QTableWidget,
)
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AdapterGuide
from zcu_tools.gui.app.measure.services.load import LoadTabResultOutcome
from zcu_tools.gui.app.measure.services.ports import SavePaneSnapshot, TabSnapshot
from zcu_tools.gui.app.measure.services.save_control import SavedRunInfo
from zcu_tools.gui.app.measure.state import TabInteractionState
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.app.measure.ui.run_load_dialog import RunLoadDialog
from zcu_tools.gui.control_identity import ControlIdentity
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.types import SessionEnv

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui.app.measure._form_fakes import configure_form_lookup


@pytest.fixture
def rows(tmp_path: Path) -> tuple[SavedRunInfo, ...]:
    return (
        SavedRunInfo("r1", "p", "scan", "2026-10-08T01:00:00Z", tmp_path / "r1.h5"),
        SavedRunInfo("r2", "p", "fit", "2026-10-09T02:00:00Z", None),
        SavedRunInfo("r3", "other", "scan", "2026-10-09T03:00:00Z", tmp_path / "r3.h5"),
    )


def test_dialog_filters_and_disables_unpersisted_selection(
    qapp: QApplication, rows: tuple[SavedRunInfo, ...]
) -> None:
    dialog = RunLoadDialog(rows, initial_point="p", start_directory="")
    table = dialog.findChild(QTableWidget, "load_runs")
    point = dialog.findChild(QComboBox, "load_point")
    experiment = dialog.findChild(QComboBox, "load_experiment")
    time = dialog.findChild(QLineEdit, "load_recorded_at")
    load = dialog.findChild(QPushButton, "load_selected")
    assert (
        table is not None
        and point is not None
        and experiment is not None
        and time is not None
        and load is not None
    )
    assert table.rowCount() == 2
    table.selectRow(1)
    assert not load.isEnabled()
    experiment.setCurrentIndex(experiment.findData("scan"))
    assert table.rowCount() == 1
    point.setCurrentIndex(0)
    assert table.rowCount() == 2
    time.setText("2026-10-09")
    assert table.rowCount() == 1
    table.selectRow(0)
    assert load.isEnabled()
    load.click()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.selected_path() == str(rows[2].data_path)
    dialog.deleteLater()
    qapp.processEvents()


def test_dialog_cancel_and_file_selection(
    qapp: QApplication,
    rows: tuple[SavedRunInfo, ...],
    tmp_path: Path,
) -> None:
    dialog = RunLoadDialog(rows, initial_point="", start_directory=str(tmp_path))
    cancel = dialog.findChild(QPushButton, "load_cancel")
    assert cancel is not None
    cancel.click()
    assert dialog.selected_path() is None
    dialog.deleteLater()
    native = tmp_path / "native.h5"
    native.touch()
    path = str(native)
    seen: list[bool] = []

    def choose() -> None:
        chooser = QApplication.activeModalWidget()
        if not isinstance(chooser, QFileDialog):
            seen.append(False)
            if isinstance(chooser, QDialog):
                chooser.reject()
            return
        seen.append(True)
        chooser.selectFile(path)
        QMetaObject.invokeMethod(chooser, "accept", Qt.ConnectionType.DirectConnection)

    selected = RunLoadDialog((), initial_point="", start_directory=str(tmp_path))
    file_button = selected.findChild(QPushButton, "load_file")
    assert file_button is not None
    QTimer.singleShot(0, choose)
    file_button.click()
    assert seen == [True]
    assert selected.selected_path() == path
    selected.deleteLater()
    qapp.processEvents()


@pytest.mark.parametrize("action", ["load", "cancel", "query-error"])
def test_main_window_routes_dialog_selection_to_existing_load(
    qapp: QApplication,
    rows: tuple[SavedRunInfo, ...],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    action: str,
) -> None:
    ctrl = MagicMock(control_identity=ControlIdentity())
    ctrl.get_adapter_guide.return_value = AdapterGuide(
        markdown="# Synthetic guide\n", resource_dir=None
    )
    configure_form_lookup(ctrl)
    ctrl.get_left_panel_width.return_value = 500
    ctrl.get_bus.return_value = BaseEventBus()
    ctrl.active_operation_count.return_value = 0
    ctrl.get_active_context_label.return_value = None
    ctrl.has_project.return_value = False
    ctrl.context_control.get_alias_rows.return_value = ()
    ctrl.get_session_env.return_value = SessionEnv(
        parameters=None,
        soc=None,
        soccfg=None,
        active_label="p",
        database_path=str(tmp_path),
    )
    ctrl.has_tab.return_value = True
    ctrl.get_tab_snapshot.return_value = TabSnapshot(
        adapter_name="fake",
        tab_id="tab",
        cfg_input={},
        capabilities=AdapterCapabilities(load_data=True),
        interaction=TabInteractionState(
            False, False, False, True, True, False, False, False, False
        ),
        save=SavePaneSnapshot(),
    )
    ctrl.save_control.list_saved_runs.return_value = rows
    if action == "query-error":
        ctrl.save_control.list_saved_runs.side_effect = ValueError("corrupt ledger")
    ctrl.load_tab_result.return_value = LoadTabResultOutcome(
        "tab",
        str(rows[0].data_path),
        "Result",
        True,
        False,
    )
    dialogs = RecordingDialogPresenter()
    window = MainWindow(ctrl, dialog_presenter=dialogs)
    window.add_tab_widget("tab", "fake")

    def choose(dialog: RunLoadDialog) -> int:
        if action == "cancel":
            dialog.reject()
        else:
            table = dialog.findChild(QTableWidget, "load_runs")
            button = dialog.findChild(QPushButton, "load_selected")
            assert table is not None and button is not None
            table.selectRow(0)
            button.click()
        return dialog.result()

    monkeypatch.setattr(RunLoadDialog, "exec", choose)
    window.load_tab_data_dialog("tab")
    if action == "load":
        ctrl.load_tab_result.assert_called_once_with("tab", str(rows[0].data_path))
    else:
        ctrl.load_tab_result.assert_not_called()
    if action == "query-error":
        assert "corrupt ledger" in dialogs.consume_message_containing(
            "critical", "corrupt ledger"
        )
    dialogs.assert_no_unexpected_messages()
    window.deleteLater()
    qapp.processEvents()
