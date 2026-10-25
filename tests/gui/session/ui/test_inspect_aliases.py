"""Inspect commits completed valid alias edits atomically through its owner."""

from __future__ import annotations

import os
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QPalette
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QWidget,
)
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.session.ui.inspect_dialog import InspectDialog
from zcu_tools.gui.widgets import error_colour, mixed_colour
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.entry.aliases import EntryAliases
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._user_input import ComponentDriver
from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry


class _ArbNames:
    """Arb asset names offered to arb data fields; no asset database is read."""

    def list_data_keys(self) -> list[str]:
        return []


@dataclass
class _Session:
    control: ContextControlFacet
    bus: BaseEventBus
    entry: ResultEntry | None = None
    dialogs: list[InspectDialog] = field(default_factory=list)
    owner: ContextService | None = None


def _service(bus: BaseEventBus, rows: tuple[AliasRow, ...] = ()) -> ContextService:
    return ContextService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        bus,
        bind_assets=ArbWaveformDatabase.info_at,
        alias_rows=rows,
    )


def _close_all(session: _Session, qapp: QApplication) -> None:
    # hide() skips closeEvent, so a draft left by a case asks nothing here.
    for dialog in session.dialogs:
        dialog.hide()
        dialog.deleteLater()
    qapp.processEvents()


@pytest.fixture
def memory_session(qapp: QApplication, tmp_path: Path) -> Generator[_Session]:
    """A selected point with authored rows that need not resolve."""
    entry = ResultEntry.create(
        "draft", result_root=tmp_path / "result", database_root=tmp_path / "Database"
    )
    entry.new_point("working")
    bus = BaseEventBus()
    service = _service(bus)
    service.install_entry(entry)
    service.use_point(PointUseRequest("working"))
    control = ContextControlFacet(context=service)
    control.set_alias_rows(
        (AliasRow("qubit", "Q1"), AliasRow("resonator", "$qubit.resonator"))
    )
    session = _Session(control, bus)
    try:
        yield session
    finally:
        _close_all(session, qapp)


@pytest.fixture
def entry_session(qapp: QApplication, tmp_path: Path) -> Generator[_Session]:
    """A selected point; Completed edits save its point-local aliases.yaml."""
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "result", database_root=tmp_path / "Database"
    )
    bus = BaseEventBus()
    service = _service(bus)
    entry.new_point("working")
    service.install_entry(entry)
    service.use_point(PointUseRequest("working"))
    service.set_alias_rows((AliasRow("qubit", "Q1"),))
    session = _Session(ContextControlFacet(context=service), bus, entry, owner=service)
    try:
        yield session
    finally:
        _close_all(session, qapp)


@pytest.fixture
def resolved_session(qapp: QApplication, tmp_path: Path) -> Generator[_Session]:
    """An opened entry and point with real owner-backed component links."""
    before = registry_state()
    register_fakes()
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "result", database_root=tmp_path / "Database"
    )
    entry.new_point("working")
    bus = BaseEventBus()
    service = _service(bus)
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
    session = _Session(control, bus, entry)
    try:
        yield session
    finally:
        _close_all(session, qapp)
        restore_registry(before)


def _open(
    session: _Session, presenter: RecordingDialogPresenter | None = None
) -> InspectDialog:
    """Build the shared Inspect over the session and show it."""
    dialog = InspectDialog(
        session.control,
        session.bus,
        templates={},
        arb_waveforms=_ArbNames(),
        open_arb_waveforms=lambda: None,
        dialog_presenter=presenter or RecordingDialogPresenter(),
    )
    session.dialogs.append(dialog)
    dialog.show()
    dialog.activateWindow()
    QApplication.processEvents()
    return dialog


def _aliases_page(dialog: InspectDialog) -> QWidget:
    """Make the Aliases page current, as a user clicking its tab does."""
    tabs = dialog.findChild(QTabWidget)
    assert tabs is not None
    titles = [tabs.tabText(index) for index in range(tabs.count())]
    assert "Aliases" in titles, f"Inspect pages are {titles}"
    tabs.setCurrentIndex(titles.index("Aliases"))
    page = tabs.currentWidget()
    assert page is not None
    return page


def _child[W: QWidget](parent: QWidget, kind: type[W], name: str) -> W:
    widget = parent.findChild(kind, name)
    assert widget is not None, f"{name!r} is missing"
    return widget


def _table(page: QWidget) -> QTableWidget:
    return _child(page, QTableWidget, "inspect.aliases.table")


def _button(page: QWidget, name: str) -> QPushButton:
    return _child(page, QPushButton, f"inspect.aliases.{name}")


def _error(page: QWidget) -> str:
    return _child(page, QLabel, "inspect.aliases.error").text()


def _cell(table: QTableWidget, row: int, column: int) -> QTableWidgetItem:
    item = table.item(row, column)
    assert item is not None, f"cell ({row}, {column}) is missing"
    return item


def _shown_rows(table: QTableWidget) -> list[tuple[str, str]]:
    return [
        (_cell(table, row, 0).text(), _cell(table, row, 1).text())
        for row in range(table.rowCount())
    ]


def _fresh_rows(entry: ResultEntry) -> tuple[AliasRow, ...]:
    """Reopen entry in a new owner, reading only its saved aliases.yaml."""
    fresh = _service(BaseEventBus())
    fresh.install_entry(entry)
    fresh.use_point(PointUseRequest("working"))
    return fresh.get_alias_rows()


def test_alias_cell_completion_add_and_remove_write_through(
    entry_session, tmp_path: Path
) -> None:
    ui = ComponentDriver(_open(entry_session))
    tmp_path = Path(os.environ.get("INSPECT_IMPLICIT_EVIDENCE", str(tmp_path)))
    ui.select_tab("Aliases")
    ui.edit_table_value("Q1", "Q2", object_name="inspect.aliases.table", column="Value")
    ui.screenshot(
        Path(os.environ.get("INSPECT_IMPLICIT_EVIDENCE", str(tmp_path)))
        / "aliases-cell-written.png"
    )
    assert _fresh_rows(entry_session.entry) == (AliasRow("qubit", "Q2"),)
    ui.click(object_name="inspect.aliases.add")
    ui.screenshot(
        Path(os.environ.get("INSPECT_IMPLICIT_EVIDENCE", str(tmp_path)))
        / "aliases-empty-new-row.png"
    )
    assert len(ui.table_rows(object_name="inspect.aliases.table")) == 2
    assert _fresh_rows(entry_session.entry) == (AliasRow("qubit", "Q2"),)
    ui.type_text("spare")
    ui.press_key(Qt.Key.Key_Return)
    ui.screenshot(
        Path(os.environ.get("INSPECT_IMPLICIT_EVIDENCE", str(tmp_path)))
        / "aliases-new-row-complete.png"
    )
    assert _fresh_rows(entry_session.entry) == (
        AliasRow("qubit", "Q2"),
        AliasRow("spare", ""),
    )
    ui.select_table_row("spare", object_name="inspect.aliases.table")
    ui.click(object_name="inspect.aliases.remove")
    ui.screenshot(
        Path(os.environ.get("INSPECT_IMPLICIT_EVIDENCE", str(tmp_path)))
        / "aliases-row-removed.png"
    )
    assert _fresh_rows(entry_session.entry) == (AliasRow("qubit", "Q2"),)


def test_edit_without_point_shows_actionable_error(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BaseEventBus()
    control = ContextControlFacet(context=_service(bus, (AliasRow("qubit", "Q1"),)))
    session = _Session(control, bus)
    try:
        ui = ComponentDriver(_open(session))
        ui.select_tab("Aliases")
        ui.edit_table_value(
            "Q1", "Q2", object_name="inspect.aliases.table", column="Value"
        )
        ui.screenshot(
            Path(os.environ.get("INSPECT_IMPLICIT_EVIDENCE", str(tmp_path)))
            / "aliases-no-point-error.png"
        )
        error = ui.observe(object_name="inspect.aliases.error")
        assert error.visible
        assert "Select or create a working point" in error.text
    finally:
        _close_all(session, qapp)


def test_clean_alias_draft_follows_selected_point(entry_session) -> None:
    entry = entry_session.entry
    assert entry is not None
    entry.new_point("other")
    entry.save_aliases(EntryAliases(aliases={"other_alias": "R2"}), point="other")
    page = _aliases_page(_open(entry_session))
    entry_session.control.use_point(PointUseRequest("other"))
    assert _shown_rows(_table(page)) == [("other_alias", "R2")]


def test_driver_edits_list_commits_and_reopens(entry_session, tmp_path: Path) -> None:
    dialog = _open(entry_session)
    driver = ComponentDriver(dialog)
    driver.select_tab("Aliases")
    driver.edit_table_value(
        "qubit", "participants", object_name="inspect.aliases.table", column="Alias"
    )
    driver.edit_table_value(
        "Q1", "[Q3, Q1]", object_name="inspect.aliases.table", column="Value"
    )
    driver.screenshot(
        Path(os.environ.get("INSPECT_IMPLICIT_EVIDENCE", str(tmp_path)))
        / "list-draft.png"
    )
    driver.screenshot(
        Path(os.environ.get("INSPECT_IMPLICIT_EVIDENCE", str(tmp_path)))
        / "list-applied.png"
    )
    assert driver.observe(object_name="inspect.aliases.error").text == ""
    assert driver.table_rows(object_name="inspect.aliases.table")[0].columns == (
        "participants",
        "[Q3, Q1]",
        "[Q3, Q1]",
    )
    driver.click("Close")
    reopened = ComponentDriver(_open(entry_session))
    reopened.select_tab("Aliases")
    reopened.screenshot(
        Path(os.environ.get("INSPECT_IMPLICIT_EVIDENCE", str(tmp_path)))
        / "list-reopened.png"
    )
    assert reopened.table_rows(object_name="inspect.aliases.table")[0].columns == (
        "participants",
        "[Q3, Q1]",
        "[Q3, Q1]",
    )


def test_aliases_page_shows_the_installed_ordered_rows(memory_session) -> None:
    page = _aliases_page(_open(memory_session))
    table = _table(page)
    assert table.columnCount() == 3
    headers = [table.horizontalHeaderItem(column) for column in range(3)]
    assert [item.text() if item else None for item in headers] == [
        "Alias",
        "Value",
        "Resolves to",
    ]
    assert _shown_rows(table) == [("qubit", "Q1"), ("resonator", "$qubit.resonator")]


def test_invalid_cell_is_red_without_writing_and_repair_commits(
    entry_session,
    tmp_path: Path,
) -> None:
    ui = ComponentDriver(_open(entry_session))
    ui.select_tab("Aliases")
    ui.edit_table_value(
        "Q1", "$missing.field", object_name="inspect.aliases.table", column="Value"
    )
    ui.screenshot(
        Path(os.environ.get("INSPECT_IMPLICIT_EVIDENCE", str(tmp_path)))
        / "aliases-invalid-red.png"
    )
    assert ui.observe(object_name="inspect.aliases.error").text
    assert ui.table_rows(object_name="inspect.aliases.table")[0].tooltips[1]
    assert _fresh_rows(entry_session.entry) == (AliasRow("qubit", "Q1"),)
    ui.edit_table_value(
        "$missing.field", "Q2", object_name="inspect.aliases.table", column="Value"
    )
    ui.screenshot(
        Path(os.environ.get("INSPECT_IMPLICIT_EVIDENCE", str(tmp_path)))
        / "aliases-repaired.png"
    )
    assert ui.observe(object_name="inspect.aliases.error").text == ""
    assert _fresh_rows(entry_session.entry) == (AliasRow("qubit", "Q2"),)


def test_point_switch_discards_invalid_alias_input(
    entry_session, tmp_path: Path
) -> None:
    entry_session.entry.new_point("second")
    ui = ComponentDriver(_open(entry_session))
    ui.select_tab("Aliases")
    ui.edit_table_value(
        "Q1", "not a name", object_name="inspect.aliases.table", column="Value"
    )
    assert ui.observe(object_name="inspect.aliases.error").text
    entry_session.control.use_point(PointUseRequest("second"))
    ui.screenshot(
        Path(os.environ.get("INSPECT_IMPLICIT_EVIDENCE", str(tmp_path)))
        / "aliases-switch-discards-invalid.png"
    )
    assert ui.table_rows(object_name="inspect.aliases.table") == ()
    assert _fresh_rows(entry_session.entry) == (AliasRow("qubit", "Q1"),)
    ui.click("Close")


def test_each_invalid_alias_row_is_marked(memory_session, tmp_path: Path) -> None:
    ui = ComponentDriver(_open(memory_session))
    ui.select_tab("Aliases")
    ui.edit_table_value(
        "Q1", "bad name", object_name="inspect.aliases.table", column="Value"
    )
    ui.edit_table_value(
        "$qubit.resonator",
        "bad second",
        object_name="inspect.aliases.table",
        column="Value",
    )
    ui.screenshot(
        Path(os.environ.get("INSPECT_IMPLICIT_EVIDENCE", str(tmp_path)))
        / "aliases-two-invalid.png"
    )
    rows = ui.table_rows(object_name="inspect.aliases.table")
    assert rows[0].tooltips[1]
    assert rows[1].tooltips[1]
    assert memory_session.control.get_alias_rows() == (
        AliasRow("qubit", "Q1"),
        AliasRow("resonator", "$qubit.resonator"),
    )


def test_edit_saves_to_the_opened_entry(entry_session) -> None:
    entry = entry_session.entry
    assert entry is not None
    page = _aliases_page(_open(entry_session))
    _cell(_table(page), 0, 1).setText("Q2")
    assert _fresh_rows(entry) == (AliasRow("qubit", "Q2"),)
    assert entry_session.control.get_alias_rows() == (AliasRow("qubit", "Q2"),)


def test_storage_error_keeps_draft_and_installed_table_for_retry(
    entry_session,
) -> None:
    entry = entry_session.entry
    assert entry is not None
    control = entry_session.control
    page = _aliases_page(_open(entry_session))
    value = _cell(_table(page), 0, 1)
    # A directory where aliases.yaml belongs makes the atomic replace fail.
    blocker = entry.result_path / "points" / "working" / "aliases.yaml"
    original = blocker.read_bytes()
    blocker.unlink()
    blocker.mkdir()
    value.setText("Q2")
    assert control.get_alias_rows() == (AliasRow("qubit", "Q1"),)
    assert _error(page)
    assert value.text() == "Q2"

    blocker.rmdir()
    # Restore the captured revision; removing the blocker alone is a stale edit.
    blocker.write_bytes(original)
    value.setText("Q3")
    value.setText("Q2")
    assert control.get_alias_rows() == (AliasRow("qubit", "Q2"),)
    assert _error(page) == ""
    assert _fresh_rows(entry) == (AliasRow("qubit", "Q2"),)


def test_resolution_column_shows_owner_projection_and_read_only_cells(
    resolved_session,
) -> None:
    control = resolved_session.control
    table = _table(_aliases_page(_open(resolved_session)))
    direct, linked, unbound, broken = (
        _cell(table, row, 2) for row in range(table.rowCount())
    )
    assert direct.text() == "Q1"
    assert linked.text() == "R1"
    assert unbound.text() == "(unbound)"
    assert unbound.foreground().color() == mixed_colour(
        table.palette().color(QPalette.ColorRole.Text),
        table.palette().color(QPalette.ColorRole.Window),
        0.42,
    )
    error = control.read_context().resolved_aliases()[3].error
    assert error is not None
    assert broken.text() == error
    assert broken.toolTip() == error
    assert broken.foreground().color() == error_colour(table)
    for row in range(table.rowCount()):
        assert not _cell(table, row, 2).flags() & Qt.ItemFlag.ItemIsEditable
    for column in (0, 1):
        assert _cell(table, 0, column).flags() & Qt.ItemFlag.ItemIsEditable


def test_resolution_without_selected_point_is_empty(qapp: QApplication) -> None:
    bus = BaseEventBus()
    control = ContextControlFacet(context=_service(bus, (AliasRow("qubit", "Q1"),)))
    session = _Session(control, bus)
    dialog = _open(session)
    table = _table(_aliases_page(dialog))
    assert table.rowCount() > 0
    for row in range(table.rowCount()):
        result = _cell(table, row, 2)
        assert result.text() == ""
        assert not result.flags() & Qt.ItemFlag.ItemIsEditable
    _close_all(session, qapp)


def test_added_resolution_cell_is_read_only_and_updates_on_completion(
    resolved_session,
) -> None:
    page = _aliases_page(_open(resolved_session))
    table = _table(page)
    _button(page, "add").click()
    result = _cell(table, 4, 2)
    assert result.text() == ""
    assert not result.flags() & Qt.ItemFlag.ItemIsEditable
    _cell(table, 4, 0).setText("spare")
    _cell(table, 4, 1).setText("R1")
    assert _cell(table, 4, 2).text() == "R1"


def test_read_only_locks_alias_writes_but_keeps_the_table_readable(
    memory_session,
) -> None:
    control = memory_session.control
    dialog = _open(memory_session)
    page = _aliases_page(dialog)
    table = _table(page)
    dialog.set_read_only(read_only=True)
    for name in ("add", "remove"):
        assert not _button(page, name).isEnabled(), name
    assert table.isEnabled()
    assert table.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers
    assert _shown_rows(table) == [("qubit", "Q1"), ("resonator", "$qubit.resonator")]
    table.selectRow(1)
    assert table.selectedItems()

    dialog.set_read_only(read_only=False)
    for name in ("add", "remove"):
        assert _button(page, name).isEnabled(), name
    assert table.editTriggers() != QAbstractItemView.EditTrigger.NoEditTriggers
    _cell(table, 0, 1).setText("Q2")
    assert control.get_alias_rows()[0] == AliasRow("qubit", "Q2")


def test_alias_resolution_error_uses_table_theme(resolved_session, window_palette):
    page = _aliases_page(_open(resolved_session))
    table = _table(page)
    table.setPalette(window_palette)
    _cell(table, 0, 1).setText("R1")
    broken = _cell(table, 3, 2)
    assert (
        broken.text()
        == resolved_session.control.read_context().resolved_aliases()[3].error
    )
    assert broken.foreground().color() == error_colour(table)
