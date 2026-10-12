"""Ordered alias drafts commit atomically through the real session owner."""

from collections.abc import Generator
from pathlib import Path

import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QDialogButtonBox, QLabel, QPushButton, QTableWidget
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote.dialogs import DialogName
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.session.ui.alias_dialog import AliasDialog
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._progress_fakes import DirectProgressTransport
from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry


@pytest.fixture
def alias_editor(qapp) -> Generator[tuple[AliasDialog, ContextControlFacet]]:
    service = ContextService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        BaseEventBus(),
        bind_assets=ArbWaveformDatabase.info_at,
    )
    control = ContextControlFacet(context=service)
    control.set_alias_rows(
        (AliasRow("qubit", "Q1"), AliasRow("resonator", "$qubit.resonator"))
    )
    dialog = AliasDialog(control)
    dialog.show()
    try:
        yield dialog, control
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_invalid_apply_keeps_installed_rows_and_editable_draft(alias_editor) -> None:
    dialog, control = alias_editor
    original = control.get_alias_rows()
    table = dialog.findChild(QTableWidget)
    buttons = dialog.findChild(QDialogButtonBox)
    assert table is not None and buttons is not None
    linked = table.item(1, 1)
    qubit = table.item(0, 1)
    assert linked is not None and qubit is not None
    assert linked.text() == "$qubit.resonator"
    qubit.setText("$resonator")
    apply = buttons.button(QDialogButtonBox.StandardButton.Apply)
    assert apply is not None
    apply.click()
    assert control.get_alias_rows() == original
    assert qubit.text() == "$resonator"
    assert dialog.findChildren(QLabel)[-1].text()
    assert dialog.isVisible()
    qubit.setText("Q2")
    apply.click()
    assert control.get_alias_rows() == (
        AliasRow("qubit", "Q2"),
        AliasRow("resonator", "$qubit.resonator"),
    )
    assert dialog.findChildren(QLabel)[-1].text() == ""
    assert dialog.isVisible()


def test_add_remove_preserves_order_and_empty_apply_clears_owner(alias_editor) -> None:
    dialog, control = alias_editor
    table = dialog.findChild(QTableWidget)
    buttons = dialog.findChild(QDialogButtonBox)
    assert table is not None and buttons is not None
    add = next(
        button for button in dialog.findChildren(QPushButton) if button.text() == "Add"
    )
    remove = next(
        button
        for button in dialog.findChildren(QPushButton)
        if button.text() == "Remove"
    )
    add.click()
    alias = table.item(2, 0)
    value = table.item(2, 1)
    assert alias is not None and value is not None
    alias.setText("spare")
    value.setText("Q3")
    table.clearSelection()
    table.selectRow(1)
    remove.click()
    assert control.get_alias_rows() == (
        AliasRow("qubit", "Q1"),
        AliasRow("resonator", "$qubit.resonator"),
    )
    apply = buttons.button(QDialogButtonBox.StandardButton.Apply)
    assert apply is not None
    apply.click()
    assert control.get_alias_rows() == (
        AliasRow("qubit", "Q1"),
        AliasRow("spare", "Q3"),
    )
    table.selectAll()
    remove.click()
    apply.click()
    assert control.get_alias_rows() == ()


def test_cancel_discards_unapplied_edits_but_keeps_prior_apply(alias_editor) -> None:
    dialog, control = alias_editor
    table = dialog.findChild(QTableWidget)
    buttons = dialog.findChild(QDialogButtonBox)
    assert table is not None and buttons is not None
    qubit = table.item(0, 1)
    assert qubit is not None
    qubit.setText("Q2")
    apply = buttons.button(QDialogButtonBox.StandardButton.Apply)
    cancel = buttons.button(QDialogButtonBox.StandardButton.Cancel)
    assert apply is not None and cancel is not None
    apply.click()
    committed = control.get_alias_rows()
    qubit.setText("Q3")
    cancel.click()
    assert not dialog.isVisible()
    assert control.get_alias_rows() == committed
    assert committed[0] == AliasRow("qubit", "Q2")


def test_toolbar_and_named_dialog_use_the_same_alias_owner(
    qapp, tmp_path, monkeypatch
) -> None:
    background = BackgroundRunner()
    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.services.app_services.BackgroundRunner",
        lambda: background,
    )
    ctrl = Controller(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        Registry(),
        None,
        BaseEventBus(),
        progress_transport=DirectProgressTransport(),
        project_root=str(tmp_path),
    )
    window = MainWindow(ctrl, dialog_presenter=RecordingDialogPresenter())
    try:
        ctrl.context_control.set_alias_rows((AliasRow("qubit", "Q2"),))
        button = next(
            button
            for button in window.findChildren(QPushButton)
            if button.text() == "Aliases…"
        )
        button.click()
        qapp.processEvents()
        assert window.list_open_dialogs() == [DialogName.ALIASES]
        dialog = window.findChild(AliasDialog)
        assert dialog is not None
        table = dialog.findChild(QTableWidget)
        assert table is not None
        value = table.item(0, 1)
        assert value is not None and value.text() == "Q2"
        window.close_dialog(DialogName.ALIASES)
        assert window.list_open_dialogs() == []
        window.open_dialog(DialogName.ALIASES)
        assert window.list_open_dialogs() == [DialogName.ALIASES]
        assert ctrl.context_control.get_alias_rows() == (AliasRow("qubit", "Q2"),)
    finally:
        window.close_dialog(DialogName.ALIASES)
        assert background.quiesce()
        window.deleteLater()
        qapp.processEvents()


@pytest.fixture
def entry_editor(
    qapp, tmp_path: Path
) -> Generator[tuple[AliasDialog, ContextControlFacet, ResultEntry]]:
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "result", database_root=tmp_path / "Database"
    )
    service = ContextService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        BaseEventBus(),
        bind_assets=ArbWaveformDatabase.info_at,
        alias_rows=(AliasRow("qubit", "Q1"),),
    )
    service.install_entry(entry)
    control = ContextControlFacet(context=service)
    dialog = AliasDialog(control)
    dialog.show()
    try:
        yield dialog, control, entry
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_apply_persists_and_cancel_only_discards_later_draft(entry_editor) -> None:
    dialog, control, entry = entry_editor
    table = dialog.findChild(QTableWidget)
    buttons = dialog.findChild(QDialogButtonBox)
    assert table is not None and buttons is not None
    value = table.item(0, 1)
    assert value is not None
    apply = buttons.button(QDialogButtonBox.StandardButton.Apply)
    cancel = buttons.button(QDialogButtonBox.StandardButton.Cancel)
    assert apply is not None and cancel is not None
    value.setText("Q2")
    apply.click()
    assert dialog.isVisible()
    fresh = ContextService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        BaseEventBus(),
        bind_assets=ArbWaveformDatabase.info_at,
    )
    fresh.install_entry(entry)
    assert fresh.get_alias_rows() == (AliasRow("qubit", "Q2"),)
    value.setText("Q3")
    cancel.click()
    fresh.install_entry(entry)
    assert (
        fresh.get_alias_rows() == control.get_alias_rows() == (AliasRow("qubit", "Q2"),)
    )


def test_storage_error_keeps_dialog_draft_and_installed_table_for_retry(
    entry_editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    dialog, control, entry = entry_editor
    table = dialog.findChild(QTableWidget)
    buttons = dialog.findChild(QDialogButtonBox)
    assert table is not None and buttons is not None
    value = table.item(0, 1)
    assert value is not None
    apply = buttons.button(QDialogButtonBox.StandardButton.Apply)
    assert apply is not None
    value.setText("Q2")

    def fail_replace(source: Path, destination: Path) -> Path:
        raise OSError("synthetic disk failure")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_replace)
        apply.click()
    assert control.get_alias_rows() == (AliasRow("qubit", "Q1"),)
    assert dialog.findChildren(QLabel)[-1].text() == "synthetic disk failure"
    assert value.text() == "Q2"
    assert dialog.isVisible()
    apply.click()
    assert control.get_alias_rows() == (AliasRow("qubit", "Q2"),)
    assert dialog.findChildren(QLabel)[-1].text() == ""
    assert dialog.isVisible()
    fresh = ContextService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        BaseEventBus(),
        bind_assets=ArbWaveformDatabase.info_at,
    )
    fresh.install_entry(entry)
    assert fresh.get_alias_rows() == (AliasRow("qubit", "Q2"),)


@pytest.fixture
def resolved_editor(qapp, tmp_path: Path):
    """An opened entry and point with real owner-backed component links."""
    before = registry_state()
    register_fakes()
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "result", database_root=tmp_path / "Database"
    )
    entry.new_point("working")
    service = ContextService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        BaseEventBus(),
        bind_assets=ArbWaveformDatabase.info_at,
    )
    service.install_entry(entry)
    control = ContextControlFacet(context=service)
    control.use_point(PointUseRequest("working"))
    control.add_component("R1", "fake/sensor", {})
    control.add_component("Q1", "fake/drive/a", {"sense": "R1"})
    control.set_alias_rows(
        (
            AliasRow("qubit", "Q1"),
            AliasRow("resonator", "$qubit.sense"),
            AliasRow("flux_dev", ""),
            AliasRow("broken", "$qubit.no_such_field"),
        )
    )
    dialog = AliasDialog(control)
    dialog.show()
    try:
        yield dialog, control
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()
        restore_registry(before)


def test_resolution_column_shows_owner_projection_and_read_only_cells(
    resolved_editor,
) -> None:
    dialog, control = resolved_editor
    table = dialog.findChild(QTableWidget)
    assert table is not None
    assert table.columnCount() == 3
    resolved = [table.item(index, 2) for index in range(table.rowCount())]
    assert all(item is not None for item in resolved)
    direct, linked, unbound, broken = resolved
    assert direct is not None and linked is not None
    assert unbound is not None and broken is not None
    assert direct.text() == "Q1"
    assert linked.text() == "R1"
    assert unbound.text() == "(unbound)"
    assert unbound.foreground().color() == QColor("gray")
    error = control.read_context().resolved_aliases()[3].error
    assert error is not None
    assert broken.text() == error
    assert broken.toolTip() == error
    assert broken.foreground().color() == QColor("red")
    for item in resolved:
        assert item is not None
        assert not item.flags() & Qt.ItemFlag.ItemIsEditable
    for column in (0, 1):
        item = table.item(0, column)
        assert item is not None
        assert item.flags() & Qt.ItemFlag.ItemIsEditable


def test_resolution_stays_applied_until_successful_apply(resolved_editor) -> None:
    dialog, control = resolved_editor
    table = dialog.findChild(QTableWidget)
    buttons = dialog.findChild(QDialogButtonBox)
    assert table is not None and buttons is not None
    assert table.columnCount() == 3
    value, result = table.item(0, 1), table.item(0, 2)
    assert value is not None and result is not None
    apply = buttons.button(QDialogButtonBox.StandardButton.Apply)
    assert apply is not None
    value.setText("$resonator")
    apply.click()
    assert table.item(0, 2).text() == "Q1"
    value.setText("R1")
    assert table.item(0, 2).text() == "Q1"
    apply.click()
    assert control.get_alias_rows()[0].value == "R1"
    assert table.item(0, 2).text() == "R1"
    linked = table.item(1, 2)
    assert linked is not None
    assert linked.text() == control.read_context().resolved_aliases()[1].error
    assert linked.toolTip() == linked.text()


@pytest.mark.parametrize("fixture_name", ["alias_editor", "entry_editor"])
def test_resolution_without_entry_or_selected_point_is_empty_and_read_only(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    dialog = request.getfixturevalue(fixture_name)[0]
    table = dialog.findChild(QTableWidget)
    assert table is not None
    assert table.columnCount() == 3
    for index in range(table.rowCount()):
        result = table.item(index, 2)
        assert result is not None
        assert result.text() == ""
        assert not result.flags() & Qt.ItemFlag.ItemIsEditable


def test_added_resolution_cell_is_read_only_and_updates_on_apply(
    resolved_editor,
) -> None:
    dialog, _ = resolved_editor
    table = dialog.findChild(QTableWidget)
    buttons = dialog.findChild(QDialogButtonBox)
    assert table is not None and buttons is not None
    add = next(b for b in dialog.findChildren(QPushButton) if b.text() == "Add")
    add.click()
    result = table.item(4, 2)
    assert result is not None
    assert result.text() == ""
    assert not result.flags() & Qt.ItemFlag.ItemIsEditable
    alias, value = table.item(4, 0), table.item(4, 1)
    assert alias is not None and value is not None
    alias.setText("spare")
    value.setText("R1")
    apply = buttons.button(QDialogButtonBox.StandardButton.Apply)
    assert apply is not None
    apply.click()
    assert table.item(4, 2).text() == "R1"
