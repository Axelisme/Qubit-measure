"""Screenshots of measure-gui named dialogs through the public registry."""

from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QWidget
from zcu_tools.gui.app.main.services.remote.dialogs import DialogName
from zcu_tools.gui.app.main.ui.main_dialog_registry import MainDialogRegistry
from zcu_tools.gui.widgets import DialogRefStore


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
