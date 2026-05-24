"""Smoke tests for InspectDialog."""

from __future__ import annotations

from unittest.mock import MagicMock

from qtpy.QtWidgets import QTableWidgetItem  # type: ignore[attr-defined]
from zcu_tools.gui.ui.inspect_dialog import InspectDialog


def test_inspect_dialog_init_and_refresh(qapp):
    ctrl = MagicMock()
    bus = MagicMock()

    # Setup mock returns
    mock_md = MagicMock()
    mock_md.items.return_value = {"scalar_int": 42, "scalar_float": 3.14, "nested": {"key": "value"}}.items()
    ctrl.get_current_md.return_value = mock_md

    mock_ml = MagicMock()
    mock_ml.modules = {"my_module": {"type": "readout"}}
    mock_ml.waveforms = {"my_waveform": {"type": "square"}}
    ctrl.get_current_ml.return_value = mock_ml

    dialog = InspectDialog(ctrl, bus)

    # Check subscriptions
    assert bus.subscribe.call_count >= 2

    # Check if MD dict was populated
    # The layout is flat, so there should be 3 rows (scalar_int, scalar_float, nested)
    assert dialog._md_table.rowCount() == 3

    # Check if ML was populated (tree widget)
    # The top level items should be "modules" and "waveforms"
    assert dialog._ml_tree.topLevelItemCount() == 2


def test_inspect_dialog_md_edit(qapp):
    ctrl = MagicMock()
    bus = MagicMock()

    mock_md = MagicMock()
    mock_md.items.return_value = {"key1": 10}.items()
    ctrl.get_current_md.return_value = mock_md
    dialog = InspectDialog(ctrl, bus)

    # Select row to populate edit box
    dialog._md_table.selectRow(0)
    dialog._on_md_row_clicked(0, 0)

    dialog._edit_value.setText("20")
    dialog._set_btn.click()

    ctrl.set_md_attr.assert_called_with("key1", 20)
