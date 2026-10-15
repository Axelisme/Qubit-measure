"""The shared Inspect's Aliases page edits the ordered global alias table.

Every case drives the page through the real session owner: Apply commits the
whole draft atomically, the resolution column projects the installed table,
unsaved drafts follow Inspect's Save/Discard/Cancel close decision and the
read-only mode only locks writes. Synthetic named presets replace the draft
through the page's Load preset menu.
"""

from __future__ import annotations

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
from zcu_tools.gui.session.alias_presets import AliasPresets
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.session.ui.inspect_dialog import InspectDialog
from zcu_tools.gui.widgets import error_colour, mixed_colour
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.gui._dialog_fakes import RecordingDialogPresenter
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
def memory_session(qapp: QApplication) -> Generator[_Session]:
    """No opened entry: Apply stays memory-only and nothing resolves."""
    bus = BaseEventBus()
    control = ContextControlFacet(context=_service(bus))
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
    """An opened entry without a selected point; Apply saves aliases.yaml."""
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "result", database_root=tmp_path / "Database"
    )
    bus = BaseEventBus()
    service = _service(bus, (AliasRow("qubit", "Q1"),))
    service.install_entry(entry)
    session = _Session(ContextControlFacet(context=service), bus, entry)
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
        alias_presets=None,
    )
    session.dialogs.append(dialog)
    dialog.show()
    return dialog


def _open_with_presets(
    session: _Session,
    presets: AliasPresets | None,
    presenter: RecordingDialogPresenter | None = None,
) -> InspectDialog:
    """Build the shared Inspect with a composition's preset catalog and show it."""
    try:
        dialog = InspectDialog(
            session.control,
            session.bus,
            templates={},
            arb_waveforms=_ArbNames(),
            open_arb_waveforms=lambda: None,
            dialog_presenter=presenter or RecordingDialogPresenter(),
            alias_presets=presets,
        )
    except TypeError as error:
        pytest.fail(f"Inspect must accept the composition's alias presets: {error}")
    session.dialogs.append(dialog)
    dialog.show()
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
    return fresh.get_alias_rows()


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


def test_invalid_apply_keeps_installed_rows_and_editable_draft(memory_session) -> None:
    control = memory_session.control
    page = _aliases_page(_open(memory_session))
    table = _table(page)
    original = control.get_alias_rows()
    _cell(table, 0, 1).setText("$resonator")
    _button(page, "apply").click()
    assert control.get_alias_rows() == original
    assert _cell(table, 0, 1).text() == "$resonator"
    assert _error(page)
    error = _child(page, QLabel, "inspect.aliases.error")
    assert error.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse

    _cell(table, 0, 1).setText("Q2")
    _button(page, "apply").click()
    assert control.get_alias_rows() == (
        AliasRow("qubit", "Q2"),
        AliasRow("resonator", "$qubit.resonator"),
    )
    assert _error(page) == ""


def test_add_remove_preserves_order_and_empty_apply_clears_owner(
    memory_session,
) -> None:
    control = memory_session.control
    page = _aliases_page(_open(memory_session))
    table = _table(page)
    _button(page, "add").click()
    _cell(table, 2, 0).setText("spare")
    _cell(table, 2, 1).setText("Q3")
    table.clearSelection()
    table.selectRow(1)
    _button(page, "remove").click()
    assert control.get_alias_rows() == (
        AliasRow("qubit", "Q1"),
        AliasRow("resonator", "$qubit.resonator"),
    )
    _button(page, "apply").click()
    assert control.get_alias_rows() == (
        AliasRow("qubit", "Q1"),
        AliasRow("spare", "Q3"),
    )
    table.selectAll()
    _button(page, "remove").click()
    _button(page, "apply").click()
    assert control.get_alias_rows() == ()


def test_apply_saves_to_the_opened_entry(entry_session) -> None:
    entry = entry_session.entry
    assert entry is not None
    page = _aliases_page(_open(entry_session))
    _cell(_table(page), 0, 1).setText("Q2")
    _button(page, "apply").click()
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
    value.setText("Q2")
    # A directory where aliases.yaml belongs makes the atomic replace fail.
    blocker = entry.result_path / "aliases.yaml"
    blocker.mkdir()
    _button(page, "apply").click()
    assert control.get_alias_rows() == (AliasRow("qubit", "Q1"),)
    assert _error(page)
    assert value.text() == "Q2"

    blocker.rmdir()
    _button(page, "apply").click()
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


def test_resolution_stays_applied_until_successful_apply(resolved_session) -> None:
    control = resolved_session.control
    page = _aliases_page(_open(resolved_session))
    table = _table(page)
    _cell(table, 0, 1).setText("$resonator")
    _button(page, "apply").click()
    assert _cell(table, 0, 2).text() == "Q1"
    _cell(table, 0, 1).setText("R1")
    assert _cell(table, 0, 2).text() == "Q1"
    _button(page, "apply").click()
    assert control.get_alias_rows()[0].value == "R1"
    assert _cell(table, 0, 2).text() == "R1"
    linked = _cell(table, 1, 2)
    assert linked.text() == control.read_context().resolved_aliases()[1].error
    assert linked.toolTip() == linked.text()


@pytest.mark.parametrize("fixture_name", ["memory_session", "entry_session"])
def test_resolution_without_entry_or_selected_point_is_empty(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    table = _table(_aliases_page(_open(request.getfixturevalue(fixture_name))))
    assert table.rowCount() > 0
    for row in range(table.rowCount()):
        result = _cell(table, row, 2)
        assert result.text() == ""
        assert not result.flags() & Qt.ItemFlag.ItemIsEditable


def test_added_resolution_cell_is_read_only_and_updates_on_apply(
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
    _button(page, "apply").click()
    assert _cell(table, 4, 2).text() == "R1"


def test_dirty_alias_close_cancel_keeps_the_draft_and_save_applies(
    entry_session,
) -> None:
    entry = entry_session.entry
    assert entry is not None
    presenter = RecordingDialogPresenter()
    dialog = _open(entry_session, presenter)
    page = _aliases_page(dialog)
    _cell(_table(page), 0, 1).setText("Q2")

    presenter.queue_save_discard_cancel("cancel")
    dialog.close()
    assert [call.kind for call in presenter.calls] == ["save_discard_cancel"]
    assert dialog.isVisible()
    assert _cell(_table(page), 0, 1).text() == "Q2"
    assert entry_session.control.get_alias_rows() == (AliasRow("qubit", "Q1"),)

    presenter.queue_save_discard_cancel("save")
    dialog.close()
    assert not dialog.isVisible()
    assert entry_session.control.get_alias_rows() == (AliasRow("qubit", "Q2"),)
    assert _fresh_rows(entry) == (AliasRow("qubit", "Q2"),)


def test_dirty_alias_close_discard_leaves_the_installed_table(memory_session) -> None:
    control = memory_session.control
    installed = control.get_alias_rows()
    presenter = RecordingDialogPresenter()
    dialog = _open(memory_session, presenter)
    page = _aliases_page(dialog)
    _cell(_table(page), 0, 1).setText("Q7")
    presenter.queue_save_discard_cancel("discard")
    dialog.close()
    assert not dialog.isVisible()
    assert control.get_alias_rows() == installed


def test_a_rejected_save_on_close_keeps_inspect_open_on_the_draft(
    memory_session,
) -> None:
    control = memory_session.control
    installed = control.get_alias_rows()
    presenter = RecordingDialogPresenter()
    dialog = _open(memory_session, presenter)
    page = _aliases_page(dialog)
    _cell(_table(page), 0, 1).setText("$resonator")
    tabs = dialog.findChild(QTabWidget)
    assert tabs is not None
    tabs.setCurrentIndex(0)

    presenter.queue_save_discard_cancel("save")
    dialog.close()
    assert dialog.isVisible()
    assert control.get_alias_rows() == installed
    assert tabs.currentWidget() is page
    assert _error(page)
    assert _cell(_table(page), 0, 1).text() == "$resonator"


def test_clean_aliases_close_without_asking(memory_session) -> None:
    presenter = RecordingDialogPresenter()
    dialog = _open(memory_session, presenter)
    page = _aliases_page(dialog)
    _cell(_table(page), 0, 1).setText("Q2")
    _button(page, "apply").click()
    dialog.close()
    assert not dialog.isVisible()
    assert presenter.calls == []


def test_read_only_locks_alias_writes_but_keeps_the_table_readable(
    memory_session,
) -> None:
    control = memory_session.control
    dialog = _open(memory_session)
    page = _aliases_page(dialog)
    table = _table(page)
    dialog.set_read_only(read_only=True)
    for name in ("add", "remove", "apply"):
        assert not _button(page, name).isEnabled(), name
    assert table.isEnabled()
    assert table.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers
    assert _shown_rows(table) == [("qubit", "Q1"), ("resonator", "$qubit.resonator")]
    table.selectRow(1)
    assert table.selectedItems()

    dialog.set_read_only(read_only=False)
    for name in ("add", "remove", "apply"):
        assert _button(page, name).isEnabled(), name
    assert table.editTriggers() != QAbstractItemView.EditTrigger.NoEditTriggers
    _cell(table, 0, 1).setText("Q2")
    _button(page, "apply").click()
    assert control.get_alias_rows()[0] == AliasRow("qubit", "Q2")


# Named presets in catalog order; "zeta" precedes "alpha" so order is not sorted.
PRESETS: AliasPresets = {
    "default": (AliasRow("qubit", "Q9"),),
    "zeta": (
        AliasRow("probe", "Q2"),
        AliasRow("sense", "$probe.resonator"),
        AliasRow("spare", ""),
    ),
    "alpha": (),
}


def _preset_button(page: QWidget) -> QPushButton:
    return _button(page, "load_preset")


def _preset_names(page: QWidget) -> list[str]:
    menu = _preset_button(page).menu()
    assert menu is not None, "Load preset has no menu"
    return [action.text() for action in menu.actions()]


def _load_preset(page: QWidget, name: str) -> None:
    """Choose name from the Load preset menu, as a user clicking it does."""
    menu = _preset_button(page).menu()
    assert menu is not None, "Load preset has no menu"
    action = next((a for a in menu.actions() if a.text() == name), None)
    assert action is not None, f"Load preset lacks {name!r}"
    action.trigger()


def _rows(rows: tuple[AliasRow, ...]) -> list[tuple[str, str]]:
    return [(row.alias, row.value) for row in rows]


def test_load_preset_menu_offers_catalog_names_in_order(memory_session) -> None:
    page = _aliases_page(_open_with_presets(memory_session, PRESETS))
    assert _preset_button(page).isEnabled()
    assert _preset_names(page) == ["default", "zeta", "alpha"]


def test_without_a_catalog_no_preset_can_be_loaded(memory_session) -> None:
    page = _aliases_page(_open_with_presets(memory_session, None))
    button = _preset_button(page)
    assert not button.isEnabled()
    menu = button.menu()
    assert menu is None or not menu.actions()


def test_loading_a_preset_replaces_the_whole_draft_without_writing(
    entry_session,
) -> None:
    entry = entry_session.entry
    assert entry is not None
    control = entry_session.control
    page = _aliases_page(_open_with_presets(entry_session, PRESETS))
    table = _table(page)
    _cell(table, 0, 1).setText("Q2")
    _button(page, "add").click()
    _cell(table, 1, 0).setText("extra")
    _cell(table, 1, 1).setText("Q3")

    _load_preset(page, "zeta")
    assert _shown_rows(table) == _rows(PRESETS["zeta"])
    assert control.get_alias_rows() == (AliasRow("qubit", "Q1"),)
    assert _fresh_rows(entry) == ()

    _load_preset(page, "alpha")
    assert table.rowCount() == 0
    assert control.get_alias_rows() == (AliasRow("qubit", "Q1"),)
    assert _fresh_rows(entry) == ()


def test_apply_saves_the_loaded_preset_and_closing_keeps_it(entry_session) -> None:
    entry = entry_session.entry
    assert entry is not None
    control = entry_session.control
    presenter = RecordingDialogPresenter()
    dialog = _open_with_presets(entry_session, PRESETS, presenter)
    page = _aliases_page(dialog)
    _load_preset(page, "zeta")
    _button(page, "apply").click()
    assert control.get_alias_rows() == PRESETS["zeta"]
    assert _fresh_rows(entry) == PRESETS["zeta"]

    dialog.close()
    assert not dialog.isVisible()
    assert presenter.calls == []
    assert control.get_alias_rows() == PRESETS["zeta"]


def test_failed_apply_keeps_owner_file_and_preset_draft(entry_session) -> None:
    entry = entry_session.entry
    assert entry is not None
    control = entry_session.control
    page = _aliases_page(_open_with_presets(entry_session, PRESETS))
    _load_preset(page, "zeta")
    # A directory where aliases.yaml belongs makes the atomic replace fail.
    blocker = entry.result_path / "aliases.yaml"
    blocker.mkdir()
    _button(page, "apply").click()
    assert _error(page)
    assert control.get_alias_rows() == (AliasRow("qubit", "Q1"),)
    assert _shown_rows(_table(page)) == _rows(PRESETS["zeta"])
    blocker.rmdir()
    assert _fresh_rows(entry) == ()


def test_unsaved_preset_draft_close_follows_save_discard_cancel(
    entry_session,
) -> None:
    entry = entry_session.entry
    assert entry is not None
    control = entry_session.control
    presenter = RecordingDialogPresenter()
    dialog = _open_with_presets(entry_session, PRESETS, presenter)
    page = _aliases_page(dialog)
    _load_preset(page, "zeta")
    tabs = dialog.findChild(QTabWidget)
    assert tabs is not None
    tabs.setCurrentIndex(0)

    presenter.queue_save_discard_cancel("cancel")
    dialog.close()
    assert dialog.isVisible()
    assert _shown_rows(_table(page)) == _rows(PRESETS["zeta"])

    blocker = entry.result_path / "aliases.yaml"
    blocker.mkdir()
    presenter.queue_save_discard_cancel("save")
    dialog.close()
    assert dialog.isVisible()
    assert tabs.currentWidget() is page
    assert _error(page)
    assert _shown_rows(_table(page)) == _rows(PRESETS["zeta"])
    assert control.get_alias_rows() == (AliasRow("qubit", "Q1"),)
    blocker.rmdir()

    presenter.queue_save_discard_cancel("discard")
    dialog.close()
    assert not dialog.isVisible()
    assert [call.kind for call in presenter.calls] == ["save_discard_cancel"] * 3
    assert control.get_alias_rows() == (AliasRow("qubit", "Q1"),)
    assert _fresh_rows(entry) == ()


def test_read_only_blocks_loading_presets_but_keeps_rows_readable(
    memory_session,
) -> None:
    dialog = _open_with_presets(memory_session, PRESETS)
    page = _aliases_page(dialog)
    dialog.set_read_only(read_only=True)
    assert not _preset_button(page).isEnabled()
    assert _shown_rows(_table(page)) == [
        ("qubit", "Q1"),
        ("resonator", "$qubit.resonator"),
    ]

    dialog.set_read_only(read_only=False)
    assert _preset_button(page).isEnabled()


def test_alias_resolution_error_uses_table_theme(resolved_session, window_palette):
    page = _aliases_page(_open(resolved_session))
    table = _table(page)
    table.setPalette(window_palette)
    _button(page, "apply").click()
    broken = _cell(table, 3, 2)
    assert (
        broken.text()
        == resolved_session.control.read_context().resolved_aliases()[3].error
    )
    assert broken.foreground().color() == error_colour(table)
