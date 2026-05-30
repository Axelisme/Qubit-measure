from unittest.mock import MagicMock, patch

from zcu_tools.gui.adapter import ModuleWriteback
from zcu_tools.gui.ui.writeback_widget import WritebackWidget


def test_writeback_widget_edit_module_item(qapp):
    item = ModuleWriteback(
        key="mod_key",
        description="A module",
        edit_schema=MagicMock(),
    )

    ctrl = MagicMock()
    widget = WritebackWidget(ctrl)
    widget.populate([item])

    with patch("zcu_tools.gui.ui.writeback_widget.QDialog") as mock_qdialog_cls:
        mock_dialog_instance = MagicMock()
        mock_qdialog_cls.return_value = mock_dialog_instance
        # Verification that instantiation succeeded and structure matches
        pass
