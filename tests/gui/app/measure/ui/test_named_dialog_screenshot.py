"""Screenshots of measure-gui named dialogs through the public registry."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from qtpy.QtWidgets import QApplication, QPushButton, QWidget
from zcu_tools.gui.app.measure.app import MeasureGuiBehavior
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.remote.dialogs import DialogName
from zcu_tools.gui.app.measure.role_catalog import RoleCatalog
from zcu_tools.gui.app.measure.ui.main_dialog_registry import MainDialogRegistry
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.ui.setup_dialog import SetupDialog
from zcu_tools.gui.widgets import DialogRefStore

from tests.gui.app.measure._reload_fakes import Loader
from tests.gui.app.measure.remote._helpers import mcp_client


def test_arb_waveform_named_dialog_captures_only_while_open(qapp) -> None:
    parent = QWidget()
    ctrl = MagicMock()
    ctrl.list_arb_waveform_infos.return_value = []
    ctrl.list_arb_waveforms.return_value = []
    registry = MainDialogRegistry(ctrl, parent=parent, dialog_refs=DialogRefStore())

    with pytest.raises(RuntimeError, match="not currently open"):
        registry.take_screenshot(DialogName.ARB_WAVEFORM)

    try:
        registry.open(DialogName.ARB_WAVEFORM)
        qapp.processEvents()
        assert DialogName.ARB_WAVEFORM in registry.visible_names()
        assert registry.take_screenshot(DialogName.ARB_WAVEFORM).startswith(b"\x89PNG")
    finally:
        registry.close(DialogName.ARB_WAVEFORM)
        qapp.processEvents()

    with pytest.raises(RuntimeError, match="not currently open"):
        registry.take_screenshot(DialogName.ARB_WAVEFORM)


def _visible_setup_dialogs(window: MainWindow) -> list[SetupDialog]:
    return [dialog for dialog in window.findChildren(SetupDialog) if dialog.isVisible()]


@pytest.mark.uses_wall_clock
@pytest.mark.parametrize("opening", ["launch", "toolbar_focus", "toolbar_reopen"])
def test_setup_screenshot_uses_visible_gui_dialog_through_mcp(
    qapp: QApplication, tmp_path: Path, opening: str
) -> None:
    previous_hook = sys.excepthook
    background = BackgroundRunner()
    window: MainWindow | None = None
    remote: RemoteControlAdapter | None = None
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
        window, remote = assembly.window, assembly.control_adapter
        window.show()
        port = remote.start()
        behavior.after_show(assembly)
        qapp.processEvents()
        launched = _visible_setup_dialogs(window)
        assert len(launched) == 1
        if opening != "launch":
            if opening == "toolbar_reopen":
                window.close_dialog(DialogName.SETUP)
                qapp.processEvents()
            setup = next(
                button
                for button in window.findChildren(QPushButton)
                if button.text() == "Setup…"
            )
            setup.click()
            qapp.processEvents()
            if opening == "toolbar_focus":
                assert _visible_setup_dialogs(window) == launched
        assert len(_visible_setup_dialogs(window)) == 1
        expected = [DialogName.SETUP]
        assert window.list_open_dialogs() == expected

        bridge, invoke = mcp_client(port, tmp_path)
        try:
            invoke("connect", {"port": port})
            path = Path(invoke("screenshot", {"target": "setup"})["path"])
            assert path.is_file() and path.read_bytes().startswith(b"\x89PNG")
            assert window.list_open_dialogs() == expected
        finally:
            bridge.disconnect()
    finally:
        if remote is not None:
            remote.stop()
        background.quiesce()
        if window is not None:
            window.deleteLater()
        sys.excepthook = previous_hook
        qapp.processEvents()
