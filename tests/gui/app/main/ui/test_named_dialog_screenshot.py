"""Screenshots of measure-gui named dialogs through the public registry."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QApplication, QPushButton, QWidget
from zcu_tools.gui.app.main.app import MeasureGuiBehavior
from zcu_tools.gui.app.main.controller import Controller
from zcu_tools.gui.app.main.registry import Registry
from zcu_tools.gui.app.main.role_catalog import RoleCatalog
from zcu_tools.gui.app.main.services.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.main.services.remote.dialogs import DialogName
from zcu_tools.gui.app.main.ui.main_dialog_registry import MainDialogRegistry
from zcu_tools.gui.app.main.ui.main_window import MainWindow
from zcu_tools.gui.widgets import DialogRefStore

from tests.gui.app.main._reload_fakes import Loader
from tests.gui.app.main.services.remote._helpers import mcp_client


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


@pytest.mark.uses_wall_clock
@pytest.mark.parametrize("opening", ["startup", "toolbar"])
def test_setup_screenshot_uses_visible_gui_dialog_through_mcp(
    qapp: QApplication, tmp_path: Path, opening: str
) -> None:
    previous_hook = sys.excepthook
    ctrl: Controller | None = None
    window: MainWindow | None = None
    remote: RemoteControlAdapter | None = None
    try:
        behavior = MeasureGuiBehavior(
            lambda: (Registry(), RoleCatalog(), Loader()),
            clean=True,
            project_root=str(tmp_path),
        )
        assembly = behavior.assemble(ControlOptions(port=0))
        behavior.before_show(assembly)
        assert isinstance(assembly.controller, Controller)
        assert isinstance(assembly.window, MainWindow)
        assert isinstance(assembly.control_adapter, RemoteControlAdapter)
        ctrl, window, remote = (
            assembly.controller,
            assembly.window,
            assembly.control_adapter,
        )
        window.show()
        port = remote.start()
        behavior.after_show(assembly)
        qapp.processEvents()
        if opening == "toolbar":
            window.close_dialog(DialogName.STARTUP)
            qapp.processEvents()
            setup = next(
                button
                for button in window.findChildren(QPushButton)
                if button.text() == "Setup…"
            )
            setup.click()
            qapp.processEvents()
        expected = [DialogName.STARTUP if opening == "startup" else DialogName.SETUP]
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
        if ctrl is not None and window is not None:
            ctrl._background_svc.quiesce()  # pyright: ignore[reportPrivateUsage] - fixture teardown
            window.deleteLater()
        sys.excepthook = previous_hook
        qapp.processEvents()
