"""Screenshots of measure-gui named dialogs through the public registry."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QApplication, QWidget
from zcu_tools.gui.app.measure.remote.dialogs import DialogName
from zcu_tools.gui.app.measure.ui.main_dialog_registry import MainDialogRegistry
from zcu_tools.gui.widgets import DialogRefStore

from tests.gui.app.measure._app_launch import (
    click_toolbar_setup,
    launched_measure_app,
    visible_setup_dialogs,
)


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
@pytest.mark.parametrize("opening", ["launch", "toolbar_focus", "toolbar_reopen"])
def test_setup_screenshot_uses_visible_gui_dialog_through_mcp(
    qapp: QApplication, tmp_path: Path, opening: str
) -> None:
    with launched_measure_app(qapp, tmp_path) as app:
        launched = visible_setup_dialogs(app.window)
        assert len(launched) == 1
        if opening != "launch":
            if opening == "toolbar_reopen":
                app.window.close_dialog(DialogName.SETUP)
                qapp.processEvents()
            click_toolbar_setup(app.window)
            qapp.processEvents()
            if opening == "toolbar_focus":
                assert visible_setup_dialogs(app.window) == launched
        assert len(visible_setup_dialogs(app.window)) == 1
        expected = [DialogName.SETUP]
        assert app.window.list_open_dialogs() == expected

        path = Path(app.invoke("screenshot", {"target": "setup"})["path"])
        assert path.is_file() and path.read_bytes().startswith(b"\x89PNG")
        assert app.window.list_open_dialogs() == expected
