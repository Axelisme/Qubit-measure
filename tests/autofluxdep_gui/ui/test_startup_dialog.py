"""Tests for the autofluxdep-gui startup dialog auto-open.

Mirrors the behaviour in ``zcu_tools.gui.app.main.app._show_startup_dialog``:
on first launch the setup dialog is opened non-modally with ``startup_mode=True``
so the user is immediately prompted to configure the project/connection.
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import patch

import pytest
from zcu_tools.gui.app.autofluxdep.app import _show_startup_dialog, build_core
from zcu_tools.gui.app.autofluxdep.ui.main_window import MainWindow


@pytest.fixture
def app(qapp):
    ctrl = build_core()
    win = MainWindow(ctrl)
    yield ctrl, win
    ctrl._background_svc.quiesce()
    win.close()
    win.deleteLater()


def test_show_startup_dialog_opens_setup_dialog(app):
    """_show_startup_dialog opens SetupDialog with startup_mode=True non-modally.

    The dialog must be opened via ``dlg.open()`` (non-modal), not ``dlg.exec()``
    (modal), so the Qt event loop keeps pumping during the dialog's lifetime.
    """
    _ctrl, win = app

    opened_dialogs: list[object] = []

    class _Signal:
        def __init__(self) -> None:
            self._callbacks: list[Callable[..., None]] = []

        def connect(self, callback: Callable[..., None]) -> None:
            self._callbacks.append(callback)

        def emit(self, *args: object) -> None:
            for callback in list(self._callbacks):
                callback(*args)

    class _TrackedSetupDialog:
        """Spy that records ``open()`` calls without blocking the event loop."""

        def __init__(self, ctrl, parent, startup_mode):
            self._startup_mode = startup_mode
            self._opened = False
            self.finished = _Signal()
            self.destroyed = _Signal()
            opened_dialogs.append(self)

        def setAttribute(self, attr) -> None:
            pass

        def open(self) -> None:
            self._opened = True

    # SetupDialog is a deferred import inside _show_startup_dialog; patch the
    # source module so the local ``from … import SetupDialog`` picks up the spy.
    with patch(
        "zcu_tools.gui.session.ui.setup_dialog.SetupDialog",
        new=_TrackedSetupDialog,
    ):
        _show_startup_dialog(parent=win)

    assert len(opened_dialogs) == 1, "Expected exactly one SetupDialog to be created"
    dlg = opened_dialogs[0]
    assert isinstance(dlg, _TrackedSetupDialog)
    assert dlg._startup_mode is True, "Dialog must be opened with startup_mode=True"
    assert dlg._opened is True, "Dialog must be opened non-modally via open()"
    assert win._setup_dialog is dlg

    dlg.finished.emit(0)

    assert win._setup_dialog is None
