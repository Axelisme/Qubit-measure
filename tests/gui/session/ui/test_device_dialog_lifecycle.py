"""DeviceDialog terminal cleanup and nonterminal hide/show lifecycle."""

from __future__ import annotations

from qtpy.QtWidgets import QLineEdit
from zcu_tools.gui.session.ui.eval_field import EvalNumericField

from tests.gui.session.ui._device_dialog_support import (
    connected_snapshot,
    device_entry,
    device_list,
    device_panel,
    make_device_control,
    make_device_dialog,
)


def test_device_dialog_done_retires_all_cached_panel_callbacks(qapp):
    from zcu_tools.device.fake import FakeDeviceInfo
    from zcu_tools.device.sgs100a import RohdeSchwarzSGS100AInfo
    from zcu_tools.device.yoko import YOKOGS200Info

    infos = [
        FakeDeviceInfo(address="none"),
        RohdeSchwarzSGS100AInfo(address="none"),
        YOKOGS200Info(address="none", mode="voltage"),
    ]
    snapshots = {info.type: connected_snapshot(info.type, info) for info in infos}
    ctrl = make_device_control()
    ctrl.list_devices.return_value = [
        device_entry(info.type, info.type) for info in infos
    ]
    ctrl.get_device_snapshot.side_effect = snapshots.get
    dialog = make_device_dialog(ctrl)
    fields: list[EvalNumericField] = []
    changes: list[None] = []
    try:
        dialog.show()
        for row in range(len(infos)):
            device_list(dialog).setCurrentRow(row)
            for field in device_panel(dialog).findChildren(EvalNumericField):
                field.raw_changed.connect(lambda: changes.append(None))
                field.load_expression("1")
                fields.append(field)
        assert fields

        dialog.hide()
        dialog.show()
        for field in fields:
            edit = field.findChild(QLineEdit, "eval.expression")
            assert edit is not None
            ctrl.read_context.reset_mock()
            changes.clear()
            edit.setText("2")
            ctrl.read_context.assert_called_once_with()
            assert changes == [None]

        dialog.done(0)
        dialog.done(0)  # Terminal cleanup is safe to repeat before Qt deletion.
        ctrl.read_context.reset_mock()
        changes.clear()
        for field in fields:
            edit = field.findChild(QLineEdit, "eval.expression")
            assert edit is not None
            edit.setText("3")
        ctrl.read_context.assert_not_called()
        assert changes == []
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()
