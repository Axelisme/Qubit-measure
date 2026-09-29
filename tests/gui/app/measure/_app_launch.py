"""Launch the real measure-gui behavior (window, remote adapter, MCP client) for flow tests."""

from __future__ import annotations

import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

from qtpy.QtWidgets import QApplication, QPushButton
from zcu_tools.gui.app.measure.app import MeasureGuiBehavior
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.role_catalog import RoleCatalog
from zcu_tools.gui.app.measure.services.persistence_types import AppPersistedState
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.ui.setup_dialog import SetupDialog

from tests.gui.app.measure._reload_fakes import Loader
from tests.gui.app.measure.remote._helpers import mcp_client


@dataclass(frozen=True)
class LaunchedMeasureApp:
    controller: Controller
    window: MainWindow
    remote: RemoteControlAdapter
    invoke: Callable[[str, dict[str, Any]], dict[str, Any]]


def visible_setup_dialogs(window: MainWindow) -> list[SetupDialog]:
    return [dialog for dialog in window.findChildren(SetupDialog) if dialog.isVisible()]


def click_toolbar_setup(window: MainWindow) -> None:
    button = next(
        button
        for button in window.findChildren(QPushButton)
        if button.text() == "Setup…"
    )
    button.click()


@contextmanager
def launched_measure_app(
    qapp: QApplication,
    tmp_path: Path,
    *,
    restore: AppPersistedState | None = None,
) -> Generator[LaunchedMeasureApp]:
    """Run the app lifecycle: assemble, restore, show, after_show, then attach MCP.

    ``restore`` stands in for the saved settings the persistence caretaker would
    load in ``before_show``; it goes through the Controller's memento restore.
    """
    previous_hook = sys.excepthook
    background = BackgroundRunner()
    window: MainWindow | None = None
    remote: RemoteControlAdapter | None = None
    bridge = None
    try:
        behavior = MeasureGuiBehavior(
            lambda: (Registry(), RoleCatalog(), Loader()),
            clean=True,
            project_root=str(tmp_path),
        )
        # Keep the real worker runner owned by this fixture for safe Qt teardown.
        with patch(
            "zcu_tools.gui.app.measure.services.app_services.BackgroundRunner",
            return_value=background,
        ) as runner_factory:
            assembly = behavior.assemble(ControlOptions(port=0))
        runner_factory.assert_called_once_with()
        behavior.before_show(assembly)
        assert isinstance(assembly.controller, Controller)
        assert isinstance(assembly.window, MainWindow)
        assert isinstance(assembly.control_adapter, RemoteControlAdapter)
        controller, window, remote = (
            assembly.controller,
            assembly.window,
            assembly.control_adapter,
        )
        if restore is not None:
            controller.restore_persisted_state(restore)
        window.show()
        port = remote.start()
        behavior.after_show(assembly)
        qapp.processEvents()
        bridge, invoke = mcp_client(port, tmp_path)
        invoke("connect", {"port": port})
        yield LaunchedMeasureApp(controller, window, remote, invoke)
    finally:
        if bridge is not None:
            bridge.disconnect()
        if remote is not None:
            remote.stop()
        background.quiesce()
        if window is not None:
            window.deleteLater()
        sys.excepthook = previous_hook
        qapp.processEvents()
