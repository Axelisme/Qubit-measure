"""Opening the autofluxdep-gui Setup dialog: app launch and toolbar share one path."""

from __future__ import annotations

from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication, QPushButton
from zcu_tools.gui.app.autofluxdep.app import AutoFluxDepGuiBehavior, build_core
from zcu_tools.gui.app.autofluxdep.ui.main_window import MainWindow
from zcu_tools.gui.runtime import GuiAssembly
from zcu_tools.gui.session.ui.setup_dialog import SetupDialog


@pytest.fixture
def launched(qapp: QApplication, tmp_path: Path):
    ctrl = build_core(project_root=str(tmp_path))
    win = MainWindow(ctrl)
    behavior = AutoFluxDepGuiBehavior(project_root=str(tmp_path))
    yield ctrl, win, behavior
    ctrl.quiesce_background()
    win.close()
    win.deleteLater()


def _visible_setup_dialogs(win: MainWindow) -> list[SetupDialog]:
    return [dialog for dialog in win.findChildren(SetupDialog) if dialog.isVisible()]


def _click_toolbar_setup(win: MainWindow) -> None:
    button = next(
        button for button in win.findChildren(QPushButton) if button.text() == "Setup…"
    )
    button.click()


def _show_app(launched, qapp: QApplication) -> list[SetupDialog]:
    ctrl, win, behavior = launched
    behavior.after_show(GuiAssembly(controller=ctrl, window=win, control_adapter=None))
    qapp.processEvents()
    return _visible_setup_dialogs(win)


def test_after_show_opens_a_visible_setup_dialog(launched, qapp) -> None:
    dialogs = _show_app(launched, qapp)

    assert len(dialogs) == 1
    dialogs[0].reject()


def test_opening_setup_neither_applies_a_project_nor_connects(launched, qapp) -> None:
    ctrl, _win, _behavior = launched
    assert ctrl.state.project is None

    (dialog,) = _show_app(launched, qapp)

    assert ctrl.state.project is None
    assert ctrl.setup_control.get_soccfg() is None
    dialog.reject()


def test_toolbar_focuses_the_dialog_opened_by_after_show(launched, qapp) -> None:
    _ctrl, win, _behavior = launched
    (first,) = _show_app(launched, qapp)

    _click_toolbar_setup(win)
    qapp.processEvents()

    assert _visible_setup_dialogs(win) == [first]
    first.reject()


def test_setup_reopens_from_the_toolbar_after_it_is_closed(launched, qapp) -> None:
    _ctrl, win, _behavior = launched
    (first,) = _show_app(launched, qapp)
    first.reject()
    qapp.processEvents()
    assert _visible_setup_dialogs(win) == []

    _click_toolbar_setup(win)
    qapp.processEvents()

    (reopened,) = _visible_setup_dialogs(win)
    assert reopened is not first
    reopened.reject()
