"""The measure composition's named alias presets reach the main window's Inspect."""

from __future__ import annotations

import sys
import threading
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pytest
from qtpy.QtWidgets import (
    QApplication,
    QPushButton,
    QTableWidget,
    QTabWidget,
    QWidget,
)
from zcu_tools.gui.app.measure.app import MeasureGuiBehavior
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote.dialogs import DialogName
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.session.alias_presets import AliasPresets
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.ui.inspect_dialog import InspectDialog

from tests.gui.app.measure._reload_fakes import Loader

PRESETS: AliasPresets = {
    "default": (AliasRow("qubit", ""), AliasRow("resonator", "$qubit.resonator")),
    "probe": (AliasRow("probe", "Q2"), AliasRow("spare", "")),
}


@dataclass(frozen=True)
class _Composed:
    controller: Controller
    window: MainWindow


@pytest.fixture
def composed(qapp: QApplication, tmp_path: Path) -> Generator[_Composed]:
    """Assemble the shipped measure composition without remote control."""
    hooks = sys.excepthook, threading.excepthook
    controller: Controller | None = None
    window: MainWindow | None = None
    try:
        behavior = MeasureGuiBehavior(
            lambda: (Registry(), Loader()),
            alias_presets=PRESETS,
            clean=True,
            project_root=str(tmp_path),
        )
        assembly = behavior.assemble(None)
        assert isinstance(assembly.controller, Controller)
        controller = assembly.controller
        assert isinstance(assembly.window, MainWindow)
        window = assembly.window
        yield _Composed(controller, window)
    finally:
        if controller is not None:
            controller.close()
        if window is not None:
            for dialog in window.findChildren(InspectDialog):
                # hide() skips closeEvent, so a loaded draft asks nothing here.
                dialog.hide()
                dialog.deleteLater()
            window.deleteLater()
        sys.excepthook, threading.excepthook = hooks
        qapp.processEvents()


def _inspect_aliases(window: MainWindow) -> QWidget:
    """Open Inspect from the window and make its Aliases page current."""
    window.open_dialog(DialogName.INSPECT)
    inspect = window.findChild(InspectDialog)
    assert inspect is not None
    tabs = inspect.findChild(QTabWidget)
    assert tabs is not None
    titles = [tabs.tabText(index) for index in range(tabs.count())]
    tabs.setCurrentIndex(titles.index("Aliases"))
    page = tabs.currentWidget()
    assert page is not None
    return page


def _load_preset_button(page: QWidget) -> QPushButton:
    button = page.findChild(QPushButton, "inspect.aliases.load_preset")
    assert button is not None, "Inspect's Aliases page lacks Load preset"
    return button


def test_composition_presets_load_into_the_main_window_inspect(
    composed: _Composed,
) -> None:
    page = _inspect_aliases(composed.window)
    menu = _load_preset_button(page).menu()
    assert menu is not None
    assert [action.text() for action in menu.actions()] == ["default", "probe"]
    next(action for action in menu.actions() if action.text() == "probe").trigger()

    table = page.findChild(QTableWidget, "inspect.aliases.table")
    assert table is not None
    shown = []
    for row in range(table.rowCount()):
        alias, value = table.item(row, 0), table.item(row, 1)
        assert alias is not None and value is not None
        shown.append(AliasRow(alias.text(), value.text()))
    assert tuple(shown) == PRESETS["probe"]
    assert composed.controller.context_control.get_alias_rows() == PRESETS["default"]


def test_remote_viewing_inspect_cannot_load_presets(composed: _Composed) -> None:
    composed.window.set_writable(writable=False)
    page = _inspect_aliases(composed.window)
    assert not _load_preset_button(page).isEnabled()
