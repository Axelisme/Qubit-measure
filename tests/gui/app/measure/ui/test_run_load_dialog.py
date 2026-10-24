"""User selection and native history filters at the Qt Load dialog seam."""

from collections.abc import Iterator
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
from zcu_tools.gui.app.measure.artifacts import SavedRunInfo
from zcu_tools.gui.app.measure.services.load import LoadTabResultOutcome
from zcu_tools.gui.app.measure.services.ports import SavePaneSnapshot, TabSnapshot
from zcu_tools.gui.app.measure.state import TabInteractionState
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.app.measure.ui.run_load_dialog import RunLoadDialog
from zcu_tools.gui.control_identity import ControlIdentity
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.types import SessionEnv

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._form_fakes import configure_form_lookup
from tests.gui.app.measure.ui._qt_interactions import modal_action


@pytest.fixture
def rows(tmp_path: Path) -> tuple[SavedRunInfo, ...]:
    return (
        SavedRunInfo("r1", "p", "scan", "2026-10-08T01:00:00Z", tmp_path / "r1.h5"),
        SavedRunInfo("r2", "p", "fit", "2026-10-09T02:00:00Z", tmp_path / "r2.h5"),
        SavedRunInfo("r3", "other", "scan", "2026-10-09T03:00:00Z", tmp_path / "r3.h5"),
    )


@pytest.fixture
def history_window(
    qapp: QApplication, rows: tuple[SavedRunInfo, ...], tmp_path: Path
) -> Iterator[MainWindow]:
    """Show two synthetic tabs sharing an entry's saved run history."""
    ctrl = MagicMock(control_identity=ControlIdentity())
    configure_form_lookup(ctrl)
    ctrl.get_adapter_guide.return_value = AdapterGuide(
        markdown="# Synthetic guide\n", resource_dir=None
    )
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
    ctrl.tab_control.has_tab.return_value = True
    snapshots = {
        name: TabSnapshot(
            adapter_name=name,
            tab_id=name,
            cfg_input={},
            capabilities=AdapterCapabilities(load_data=True),
            interaction=TabInteractionState(
                False, False, False, True, True, False, False, False, False
            ),
            save=SavePaneSnapshot(),
        )
        for name in ("scan", "missing")
    }
    ctrl.tab_control.get_tab_snapshot.side_effect = snapshots.__getitem__
    ctrl.save_control.list_saved_runs.return_value = rows
    dialogs = RecordingDialogPresenter()
    window = MainWindow(ctrl, templates=lambda: {}, dialog_presenter=dialogs)
    window.add_tab_widget("scan", "scan")
    window.add_tab_widget("missing", "missing")
    window.show()
    qapp.processEvents()
    try:
        yield window
        dialogs.assert_no_unexpected_messages()
    finally:
        window.deleteLater()
        qapp.processEvents()


def test_load_data_preselects_current_experiment_and_allows_all(
    qapp: QApplication, history_window: MainWindow, tmp_path: Path
) -> None:
    ui = ComponentDriver(history_window)
    ui.select_tab("scan")
    ui.screenshot(tmp_path / "01-scan-tab.png")
    ui.select_tab("Data")
    ui.screenshot(tmp_path / "01-scan-data.png")

    def inspect_dialog(dialog: RunLoadDialog) -> None:
        chooser = ComponentDriver(dialog)
        chooser.screenshot(tmp_path / "02-scan-default.png")
        assert chooser.observe(object_name="load_experiment").text == "scan"
        assert chooser.observe(object_name="load_point").text == "p"
        assert tuple(
            row.columns[:3] for row in chooser.table_rows(object_name="load_runs")
        ) == (("r1", "p", "scan"),)
        chooser.select_combo("All experiments", object_name="load_experiment")
        chooser.screenshot(tmp_path / "03-all-experiments.png")
        assert chooser.observe(object_name="load_experiment").text == "All experiments"
        assert chooser.observe(object_name="load_point").text == "p"
        assert tuple(
            row.columns[:3] for row in chooser.table_rows(object_name="load_runs")
        ) == (("r1", "p", "scan"), ("r2", "p", "fit"))
        chooser.click("Cancel")

    modal_action(
        qapp,
        lambda: ui.click("Load Data"),
        [(RunLoadDialog, inspect_dialog)],
    )
    ui.screenshot(tmp_path / "04-scan-cancelled.png")
    assert ui.dialogs() == ()


def test_load_data_without_current_experiment_keeps_all(
    qapp: QApplication, history_window: MainWindow, tmp_path: Path
) -> None:
    ui = ComponentDriver(history_window)
    ui.select_tab("missing")
    ui.screenshot(tmp_path / "01-missing-tab.png")
    ui.select_tab("Data")
    ui.screenshot(tmp_path / "01-missing-data.png")

    def inspect_dialog(dialog: RunLoadDialog) -> None:
        chooser = ComponentDriver(dialog)
        chooser.screenshot(tmp_path / "02-missing-default.png")
        assert chooser.observe(object_name="load_experiment").text == "All experiments"
        assert chooser.observe(object_name="load_point").text == "p"
        assert tuple(
            row.columns[:3] for row in chooser.table_rows(object_name="load_runs")
        ) == (("r1", "p", "scan"), ("r2", "p", "fit"))
        chooser.click("Cancel")

    modal_action(
        qapp,
        lambda: ui.click("Load Data"),
        [(RunLoadDialog, inspect_dialog)],
    )
    ui.screenshot(tmp_path / "03-missing-cancelled.png")
    assert ui.dialogs() == ()


def test_dialog_filters_native_history_and_selects_file(
    qapp: QApplication, rows: tuple[SavedRunInfo, ...]
) -> None:
    dialog = RunLoadDialog(rows, initial_point="p", start_directory="")
    table = dialog.findChild(QTableWidget, "load_runs")
    point = dialog.findChild(QComboBox, "load_point")
    experiment = dialog.findChild(QComboBox, "load_experiment")
    time = dialog.findChild(QLineEdit, "load_started_at")
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
    assert load.isEnabled()
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
    ctrl.tab_control.has_tab.return_value = True
    ctrl.tab_control.get_tab_snapshot.return_value = TabSnapshot(
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
    ctrl.run_analyze_control.load_tab_result.return_value = LoadTabResultOutcome(
        tab_id="tab",
        data_path=str(rows[0].data_path),
        result_type="Result",
        cfg_issue=None,
        has_analyze_params=False,
    )
    dialogs = RecordingDialogPresenter()
    window = MainWindow(ctrl, templates=lambda: {}, dialog_presenter=dialogs)
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
        ctrl.run_analyze_control.load_tab_result.assert_called_once_with(
            "tab", str(rows[0].data_path)
        )
    else:
        ctrl.run_analyze_control.load_tab_result.assert_not_called()
    if action == "query-error":
        assert "corrupt ledger" in dialogs.consume_message_containing(
            "critical", "corrupt ledger"
        )
    dialogs.assert_no_unexpected_messages()
    window.deleteLater()
    qapp.processEvents()
