"""Smoke tests for DeviceDialog."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event

import pytest
from PyQt6.QtCore import QObject, pyqtSignal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QComboBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QWidget,
)
from zcu_tools.device import BaseDeviceInfo
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.device_models import (
    ConnectDeviceRequest,
    DeviceEntry,
    DeviceSnapshot,
    DisconnectDeviceRequest,
    SetupDeviceRequest,
)
from zcu_tools.gui.session.events import (
    DeviceChangedPayload,
    DeviceSetupFinishedPayload,
)
from zcu_tools.gui.session.state import DeviceStatus
from zcu_tools.gui.session.ui.device_dialog import DeviceDialog
from zcu_tools.gui.session.ui.eval_field import EvalNumericField
from zcu_tools.gui.widgets import error_colour

from tests.gui.session.ui._device_dialog_support import (
    apply_button,
    confirm_device_action,
    connected_snapshot,
    device_entry,
    device_list,
    device_panel,
    labelled_field,
    make_device_control,
    make_device_dialog,
    make_parameter_control,
)


def _button(dialog: DeviceDialog, text: str) -> QPushButton:
    return next(b for b in dialog.findChildren(QPushButton) if b.text() == text)


class _PollTimer(QObject):
    """Control the external repeating timer, retaining real Qt timeout delivery."""

    timeout = pyqtSignal()

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self.running = False
        self.interval = 0

    def setInterval(self, milliseconds: int) -> None:
        """Accept the production interval without scheduling wall-clock work."""
        assert milliseconds > 0
        self.interval = milliseconds

    def isActive(self) -> bool:
        return self.running

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def tick(self) -> None:
        """Deliver one tick only while the dialog has enabled polling."""
        if self.running:
            self.timeout.emit()


@pytest.fixture
def poll_timers(monkeypatch: pytest.MonkeyPatch):
    timers: list[_PollTimer] = []

    def create(parent: QObject) -> _PollTimer:
        timer = _PollTimer(parent)
        timers.append(timer)
        return timer

    monkeypatch.setattr("zcu_tools.gui.session.ui.device_dialog.QTimer", create)
    try:
        yield timers
    finally:
        for timer in timers:
            timer.stop()


def _setting_up_snapshot(name: str, info: BaseDeviceInfo) -> DeviceSnapshot:
    """A snapshot whose status is SETTING_UP — Phase C's SSOT for 'this device
    is mid-setup' (the dialog derives the live-setup set from snapshot status,
    not from the device service's active-op enumerators)."""
    return DeviceSnapshot(
        name=name,
        type_name=getattr(info, "type", "FakeDevice"),
        address=getattr(info, "address", ""),
        status=DeviceStatus.SETTING_UP,
        info=info,
    )


def test_device_dialog_init(qapp):
    ctrl = make_device_control()
    ctrl.list_devices.return_value = [device_entry("fakedevice")]

    from zcu_tools.device.fake import FakeDeviceInfo

    info = FakeDeviceInfo(address="none")
    ctrl.get_device_snapshot.return_value = connected_snapshot("fakedevice", info)

    dialog = make_device_dialog(ctrl)

    # list_devices should be called during init (Phase C also reads it in
    # _setup_device_names, so just assert it was used, not the exact count).
    ctrl.list_devices.assert_called()
    assert device_list(dialog).count() == 1

    device_list(dialog).setCurrentRow(0)

    # Selecting the device displays its value.
    assert (
        labelled_field(device_panel(dialog), "Value:", EvalNumericField).read_raw()
        == 0.0
    )


def test_device_dialog_add_device_dispatches_request(qapp):
    ctrl = make_device_control()

    dialog = make_device_dialog(ctrl)
    labelled_field(dialog, "Type:", QComboBox).setCurrentText("YOKOGS200")
    labelled_field(dialog, "Address:", QLineEdit).setText("TCPIP::127.0.0.1::INSTR")
    labelled_field(dialog, "Name:", QLineEdit).setText("fakedevice")

    _button(dialog, "Add Device").click()

    ctrl.start_connect_device.assert_called_once()
    (req,) = ctrl.start_connect_device.call_args.args
    assert isinstance(req, ConnectDeviceRequest)
    assert req.type_name == "YOKOGS200"
    assert req.name == "fakedevice"
    assert req.address == "TCPIP::127.0.0.1::INSTR"
    ctrl.start_setup_device.assert_not_called()


def test_device_dialog_drop_device(qapp):
    from zcu_tools.device.yoko import YOKOGS200Info

    ctrl = make_device_control()
    ctrl.list_devices.return_value = [device_entry("yoko", "YOKOGS200")]

    info = YOKOGS200Info(address="GPIB::1", mode="voltage")
    ctrl.get_device_snapshot.return_value = connected_snapshot("yoko", info)

    dialog = make_device_dialog(ctrl)

    assert device_list(dialog).count() == 1
    device_list(dialog).setCurrentRow(0)

    # Drop disconnects but keeps startup memory
    _button(dialog, "Drop").click()
    confirm_device_action("Drop")
    ctrl.start_disconnect_device.assert_called_once()
    (req,) = ctrl.start_disconnect_device.call_args.args
    assert isinstance(req, DisconnectDeviceRequest)
    assert req.name == "yoko"


def test_device_dialog_forget_memory_device_dispatches_single_transaction(qapp):
    ctrl = make_device_control()
    ctrl.list_devices.return_value = [device_entry("remembered", connected=False)]
    ctrl.is_memory_device.return_value = True
    ctrl.get_device_snapshot.return_value = DeviceSnapshot(
        name="remembered",
        type_name="FakeDevice",
        address="addr",
        status=DeviceStatus.MEMORY_ONLY,
    )

    dialog = make_device_dialog(ctrl)
    device_list(dialog).setCurrentRow(0)
    _button(dialog, "Forget").click()
    confirm_device_action("Forget")

    ctrl.forget_device.assert_called_once_with("remembered")


def test_memory_device_reconnect_uses_selected_name_without_setup(qapp):
    ctrl = make_device_control()
    ctrl.list_devices.return_value = [device_entry("remembered", connected=False)]
    ctrl.is_memory_device.return_value = True
    ctrl.get_device_snapshot.return_value = DeviceSnapshot(
        name="remembered",
        type_name="FakeDevice",
        address="addr",
        status=DeviceStatus.MEMORY_ONLY,
    )
    dialog = make_device_dialog(ctrl)
    device_list = dialog.findChild(QListWidget)
    assert device_list is not None
    device_list.setCurrentRow(0)
    reconnect = next(
        button
        for button in dialog.findChildren(QPushButton)
        if button.text() == "Reconnect"
    )
    reconnect.click()
    ctrl.start_reconnect_device.assert_called_once_with("remembered")
    ctrl.start_setup_device.assert_not_called()
    dialog.close()
    dialog.deleteLater()
    qapp.processEvents()


def test_device_dialog_refresh_reloads_selected_device_info(qapp):
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = make_device_control()
    ctrl.list_devices.return_value = [device_entry("fd")]
    # By-name + mutable so the assertion is robust to *how many* times the dialog
    # reads the snapshot (Phase C's _setup_device_names adds extra reads); the
    # value moves only when the test explicitly bumps it.
    box = {"value": 1.0}
    ctrl.get_device_snapshot.side_effect = lambda _n: connected_snapshot(
        "fd", FakeDeviceInfo(address="none", value=box["value"])
    )
    dialog = make_device_dialog(ctrl)
    device_list(dialog).setCurrentRow(0)
    panel = device_panel(dialog)
    assert labelled_field(panel, "Value:", EvalNumericField).read_raw() == 1.0

    box["value"] = 2.0
    _button(dialog, "Refresh").click()

    assert labelled_field(panel, "Value:", EvalNumericField).read_raw() == 2.0
    ctrl.get_device_info.assert_called_once_with("fd")


def test_device_changed_repaints_selected_panel(qapp):
    """A DEVICE_CHANGED for the selected device repaints the right panel with the
    fresh cached info (so a value that moved underneath us shows immediately)."""
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = make_device_control()
    ctrl.list_devices.return_value = [device_entry("fd")]
    # By-name + mutable (robust to extra snapshot reads, see above).
    box = {"value": 1.0}
    ctrl.get_device_snapshot.side_effect = lambda _n: connected_snapshot(
        "fd", FakeDeviceInfo(address="none", value=box["value"])
    )

    dialog = make_device_dialog(ctrl)
    device_list(dialog).setCurrentRow(0)
    panel = device_panel(dialog)
    assert labelled_field(panel, "Value:", EvalNumericField).read_raw() == 1.0

    box["value"] = 2.0
    ctrl.get_bus.return_value.emit(DeviceChangedPayload(name="fd"))

    assert labelled_field(panel, "Value:", EvalNumericField).read_raw() == 2.0


def test_device_changed_for_other_device_keeps_selection(qapp):
    """A DEVICE_CHANGED for a non-selected device must not steal the selection
    away from the device the user is currently inspecting."""
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = make_device_control()
    ctrl.list_devices.return_value = [device_entry("A"), device_entry("B")]
    ctrl.get_device_snapshot.side_effect = lambda n: connected_snapshot(
        n, FakeDeviceInfo(address="none")
    )

    dialog = make_device_dialog(ctrl)
    device_list(dialog).setCurrentRow(0)
    item = device_list(dialog).currentItem()
    assert item is not None and item.text().removeprefix("● ") == "A (FakeDevice)"

    ctrl.get_bus.return_value.emit(DeviceChangedPayload(name="B"))

    item = device_list(dialog).currentItem()
    assert item is not None and item.text().removeprefix("● ") == "A (FakeDevice)"


def test_device_changed_surfaces_device_when_none_selected(qapp):
    """With nothing selected, a DEVICE_CHANGED surfaces the changed device (the
    'first device just connected' case)."""
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = make_device_control()
    ctrl.list_devices.return_value = [device_entry("fd")]
    ctrl.get_device_snapshot.side_effect = lambda n: connected_snapshot(
        n, FakeDeviceInfo(address="none")
    )

    dialog = make_device_dialog(ctrl)
    device_list(dialog).setCurrentRow(-1)

    ctrl.get_bus.return_value.emit(DeviceChangedPayload(name="fd"))

    item = device_list(dialog).currentItem()
    assert item is not None
    assert item.text().removeprefix("● ") == "fd (FakeDevice)"


def test_poll_tick_polls_selected_device(qapp, poll_timers):
    """Each tick asks the controller to off-main poll the *selected* device."""
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = make_device_control()
    ctrl.list_devices.return_value = [device_entry("A"), device_entry("B")]
    ctrl.get_device_snapshot.side_effect = lambda n: connected_snapshot(
        n, FakeDeviceInfo(address="none")
    )

    dialog = make_device_dialog(ctrl)
    timer = poll_timers[0]
    try:
        device_list(dialog).setCurrentRow(1)  # select "B"
        timer.tick()
        ctrl.try_poll_device_info.assert_not_called()
        dialog.show()
        timer.tick()
        ctrl.try_poll_device_info.assert_called_once_with("B")
        ctrl.try_poll_device_info.reset_mock()
        device_list(dialog).setCurrentRow(0)
        timer.tick()
        ctrl.try_poll_device_info.assert_called_once_with("A")
        ctrl.try_poll_device_info.reset_mock()
        device_list(dialog).setCurrentRow(-1)
        timer.tick()
        ctrl.try_poll_device_info.assert_not_called()
        device_list(dialog).setCurrentRow(1)
        dialog.hide()
        timer.tick()
        ctrl.try_poll_device_info.assert_not_called()
        dialog.show()
        timer.tick()
        ctrl.try_poll_device_info.assert_called_once_with("B")
    finally:
        dialog.close()
    ctrl.try_poll_device_info.reset_mock()
    timer.tick()
    ctrl.try_poll_device_info.assert_not_called()


def test_poll_tick_polls_selected_setting_up_device(qapp, poll_timers):
    """Auto poll still asks for a best-effort refresh while the selected device
    is setting up; the synchronous Refresh button can remain disabled."""
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = make_device_control()
    ctrl.list_devices.return_value = [
        DeviceEntry(
            name="fd",
            type_name="FakeDevice",
            status=DeviceStatus.SETTING_UP.value,
        )
    ]
    ctrl.get_device_snapshot.side_effect = lambda n: _setting_up_snapshot(
        n, FakeDeviceInfo(address="none")
    )

    dialog = make_device_dialog(ctrl)
    try:
        device_list(dialog).setCurrentRow(0)
        assert _button(dialog, "Refresh").isEnabled() is False

        dialog.show()
        timer = poll_timers[0]
        timer.tick()

        ctrl.try_poll_device_info.assert_called_once_with("fd")
    finally:
        dialog.close()


def test_device_dialog_apply_changes(qapp):
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = make_device_control()
    ctrl.list_devices.return_value = [device_entry("fd")]

    info = FakeDeviceInfo(address="none")
    ctrl.get_device_snapshot.return_value = connected_snapshot("fd", info)
    ctrl.get_device_info.return_value = info
    dialog = make_device_dialog(ctrl)
    device_list(dialog).setCurrentRow(0)

    assert labelled_field(device_panel(dialog), "Value:", EvalNumericField).isEnabled()

    apply_button(dialog).click()

    ctrl.start_setup_device.assert_called_once()
    (req,) = ctrl.start_setup_device.call_args.args
    assert isinstance(req, SetupDeviceRequest)
    assert req.name == "fd"
    assert isinstance(req.info, FakeDeviceInfo)


def test_device_dialog_read_only_blocks_mutations_but_keeps_refresh(qapp):
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = make_device_control()
    ctrl.list_devices.return_value = [device_entry("fd")]
    info = FakeDeviceInfo(address="none")
    ctrl.get_device_snapshot.return_value = connected_snapshot("fd", info)
    ctrl.get_device_info.return_value = info

    dialog = make_device_dialog(ctrl)
    device_list(dialog).setCurrentRow(0)
    panel = device_panel(dialog)

    dialog.set_read_only(read_only=True)

    assert not _button(dialog, "Add Device").isEnabled()
    assert not _button(dialog, "Drop").isEnabled()
    assert _button(dialog, "Refresh").isEnabled()
    assert not apply_button(dialog).isEnabled()
    assert not labelled_field(panel, "Output:", QComboBox).isEnabled()
    assert not labelled_field(panel, "Value:", EvalNumericField).isEnabled()

    _button(dialog, "Add Device").click()
    _button(dialog, "Drop").click()
    apply_button(dialog).click()

    ctrl.start_connect_device.assert_not_called()
    ctrl.start_disconnect_device.assert_not_called()
    ctrl.forget_device.assert_not_called()
    ctrl.start_reconnect_device.assert_not_called()
    ctrl.cancel_device_operation.assert_not_called()
    ctrl.start_setup_device.assert_not_called()

    _button(dialog, "Refresh").click()

    ctrl.get_device_info.assert_called_once_with("fd")

    dialog.set_read_only(read_only=False)

    assert _button(dialog, "Add Device").isEnabled()
    assert _button(dialog, "Drop").isEnabled()
    assert _button(dialog, "Refresh").isEnabled()
    assert apply_button(dialog).isEnabled()
    assert labelled_field(panel, "Output:", QComboBox).isEnabled()
    assert labelled_field(panel, "Value:", EvalNumericField).isEnabled()


def test_device_dialog_restores_background_setup_and_stops_it(qapp):
    ctrl = make_device_control()
    ctrl.list_devices.return_value = [device_entry("fd")]
    from zcu_tools.device.fake import FakeDeviceInfo

    info = FakeDeviceInfo(address="none")
    # Phase C: a device mid-setup is expressed by its snapshot status, the SSOT
    # the dialog reads — start with fd SETTING_UP, flip it to CONNECTED on finish.
    state = {"status": DeviceStatus.SETTING_UP}
    ctrl.get_device_snapshot.side_effect = lambda _n: DeviceSnapshot(
        name="fd",
        type_name="FakeDevice",
        address="none",
        status=state["status"],
        info=info,
    )

    dialog = make_device_dialog(ctrl)
    # Opened mid-setup with nothing pre-selected → the setup owner is surfaced.
    item = device_list(dialog).currentItem()
    assert item is not None
    assert item.text().removeprefix("● ") == "fd (FakeDevice)"
    assert apply_button(dialog).text() == "Stop"
    # Phase C: the list stays enabled (user can switch devices).
    assert device_list(dialog).isEnabled() is True
    # Refresh is disabled for a device that is itself setting up.
    assert _button(dialog, "Refresh").isEnabled() is False
    # The dialog attached to progress by the setting-up device's name, so live
    # bars render even though it opened mid-setup.
    ctrl.attach_progress.assert_called_once()
    assert ctrl.attach_progress.call_args.args[0] == "fd"

    apply_button(dialog).click()
    ctrl.cancel_device_operation.assert_called_once_with("fd")

    # Setup reaches a terminal state → the device returns to CONNECTED and the
    # finish event repaints buttons back to normal.
    state["status"] = DeviceStatus.CONNECTED
    ctrl.get_bus.return_value.emit(
        DeviceSetupFinishedPayload(name="fd", outcome="cancelled"),
    )
    assert apply_button(dialog).text() == "Apply Changes"
    assert device_list(dialog).isEnabled() is True
    assert _button(dialog, "Refresh").isEnabled() is True


# ---------------------------------------------------------------------------
# Phase C concurrent-setup tests (per-device lock scoped to each device's own
# setup; different devices set up in parallel)
# ---------------------------------------------------------------------------


def _two_device_ctrl(setting_up: set[str]):
    """Build a ctrl with fd_a/fd_b where the names in ``setting_up`` report
    snapshot status SETTING_UP and the rest report CONNECTED. ``setting_up`` is
    the live, mutable set the test flips to drive setup start/finish — the dialog
    derives its setup view from snapshot status (Phase C SSOT)."""
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = make_device_control()
    ctrl.list_devices.return_value = [device_entry("fd_a"), device_entry("fd_b")]

    def snapshot(n: str) -> DeviceSnapshot:
        info = FakeDeviceInfo(address="none")
        if n in setting_up:
            return _setting_up_snapshot(n, info)
        return connected_snapshot(n, info)

    ctrl.get_device_snapshot.side_effect = snapshot
    return ctrl


def _setup_two_device_dialog(qapp):
    """Dialog with fd_a setting up and fd_b connected; selection on the
    non-owner fd_b. Returns (dialog, ctrl, setting_up_set)."""
    setting_up = {"fd_a"}
    ctrl = _two_device_ctrl(setting_up)
    dialog = make_device_dialog(ctrl)
    device_list(dialog).setCurrentRow(1)  # "fd_b"
    return dialog, ctrl, setting_up


def test_setup_running_list_remains_enabled(qapp):
    """During a setup the device list stays enabled so the user can switch
    to view (and even set up) another device."""
    dialog, _, _ = _setup_two_device_dialog(qapp)
    assert device_list(dialog).isEnabled() is True


def test_setup_running_can_switch_to_other_device(qapp):
    """Switching to another device shows that device's panel."""
    dialog, _, _ = _setup_two_device_dialog(qapp)

    item = device_list(dialog).currentItem()
    assert item is not None
    assert item.text().removeprefix("● ") == "fd_b (FakeDevice)"
    assert labelled_field(device_panel(dialog), "Value:", EvalNumericField).isEnabled()


def test_setting_up_device_shows_stop_button(qapp):
    """A device that is itself setting up shows an enabled Stop button."""
    dialog, _, _ = _setup_two_device_dialog(qapp)

    device_list(dialog).setCurrentRow(0)  # "fd_a" (setting up)
    assert apply_button(dialog).text() == "Stop"
    assert apply_button(dialog).isEnabled() is True


def test_other_device_apply_is_enabled_during_setup(qapp):
    """Phase C: a device that is NOT setting up keeps an enabled Apply even while
    another device is mid-setup — this is the concurrency the phase unlocks."""
    dialog, _, _ = _setup_two_device_dialog(qapp)

    # fd_b is connected and not setting up.
    assert apply_button(dialog).text() == "Apply Changes"
    assert apply_button(dialog).isEnabled() is True


def test_apply_on_other_device_starts_concurrent_setup(qapp):
    """Clicking Apply on a device while another is setting up dispatches a second
    setup (concurrent), rather than being blocked."""
    from zcu_tools.device.fake import FakeDeviceInfo

    dialog, ctrl, _ = _setup_two_device_dialog(qapp)
    ctrl.get_device_info.return_value = FakeDeviceInfo(address="none")

    # fd_b selected (not setting up) → Apply starts its setup even though fd_a is.
    apply_button(dialog).click()

    ctrl.start_setup_device.assert_called_once()
    (req,) = ctrl.start_setup_device.call_args.args
    assert isinstance(req, SetupDeviceRequest)
    assert req.name == "fd_b"
    ctrl.cancel_device_operation.assert_not_called()


def test_switch_between_devices_recomputes_buttons(qapp):
    """Switching between a setting-up device and an idle one flips the button
    between Stop and Apply."""
    dialog, _, _ = _setup_two_device_dialog(qapp)

    device_list(dialog).setCurrentRow(0)  # fd_a (setting up)
    assert apply_button(dialog).text() == "Stop"
    assert apply_button(dialog).isEnabled() is True

    device_list(dialog).setCurrentRow(1)  # fd_b (idle)
    assert apply_button(dialog).text() == "Apply Changes"
    assert apply_button(dialog).isEnabled() is True


def test_two_concurrent_setups_each_show_stop_and_cancel_independently(qapp):
    """With both devices setting up, each selected device shows its own Stop and
    cancels only itself."""
    setting_up = {"fd_a", "fd_b"}
    ctrl = _two_device_ctrl(setting_up)
    dialog = make_device_dialog(ctrl)

    device_list(dialog).setCurrentRow(0)  # fd_a
    assert apply_button(dialog).text() == "Stop"
    apply_button(dialog).click()
    ctrl.cancel_device_operation.assert_called_once_with("fd_a")

    device_list(dialog).setCurrentRow(1)  # fd_b
    assert apply_button(dialog).text() == "Stop"
    apply_button(dialog).click()
    assert ctrl.cancel_device_operation.call_args_list[-1].args == ("fd_b",)


def test_two_concurrent_setups_subscribe_progress_per_owner(qapp):
    """Both setting-up devices attach progress through the control port."""
    setting_up = {"fd_a", "fd_b"}
    ctrl = _two_device_ctrl(setting_up)
    dialog = make_device_dialog(ctrl)

    owners = {call.args[0] for call in ctrl.attach_progress.call_args_list}
    assert owners == {"fd_a", "fd_b"}
    dialog.close()


def test_all_setups_finished_restores_buttons(qapp):
    """When the last setup finishes, selection and buttons return to idle."""
    dialog, ctrl, setting_up = _setup_two_device_dialog(qapp)

    setting_up.discard("fd_a")
    ctrl.get_bus.return_value.emit(
        DeviceSetupFinishedPayload(name="fd_a", outcome="finished"),
    )
    # fd_b still selected, Apply enabled, list enabled.
    item = device_list(dialog).currentItem()
    assert item is not None and item.text().removeprefix("● ") == "fd_b (FakeDevice)"
    assert apply_button(dialog).text() == "Apply Changes"
    assert apply_button(dialog).isEnabled() is True
    assert device_list(dialog).isEnabled() is True


def test_drop_disabled_for_setting_up_device_only(qapp):
    """Phase C: Drop is disabled for a device that is itself setting up, but
    enabled for an idle device even while another is mid-setup."""
    dialog, _, _ = _setup_two_device_dialog(qapp)

    device_list(dialog).setCurrentRow(1)  # fd_b idle → drop enabled
    assert _button(dialog, "Drop").isEnabled() is True

    device_list(dialog).setCurrentRow(0)  # fd_a setting up → drop disabled
    assert _button(dialog, "Drop").isEnabled() is False


def test_add_box_enabled_during_setup(qapp):
    """Phase C: the Add-box stays enabled during a setup (a new-name connect is
    resource-disjoint from any in-flight mutation)."""
    dialog, _, _ = _setup_two_device_dialog(qapp)

    assert _button(dialog, "Add Device").isEnabled() is True
    assert labelled_field(dialog, "Type:", QComboBox).isEnabled() is True
    assert labelled_field(dialog, "Name:", QLineEdit).isEnabled() is True
    assert labelled_field(dialog, "Address:", QLineEdit).isEnabled() is True


def test_refresh_enabled_for_idle_device_during_other_setup(qapp):
    """Refresh is allowed for an idle device even while another device sets up."""
    dialog, _, _ = _setup_two_device_dialog(qapp)

    device_list(dialog).setCurrentRow(1)  # fd_b idle
    assert _button(dialog, "Refresh").isEnabled() is True


def test_refresh_disabled_for_setting_up_device(qapp):
    """Refresh is disabled for a device that is itself setting up."""
    dialog, _, _ = _setup_two_device_dialog(qapp)

    device_list(dialog).setCurrentRow(0)  # fd_a setting up
    assert _button(dialog, "Refresh").isEnabled() is False


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


def test_device_dialog_close_keeps_setup_running_and_unsubscribes(qapp):
    ctrl = make_device_control()
    ctrl.list_devices.return_value = [device_entry("fd")]
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl.get_device_snapshot.return_value = _setting_up_snapshot(
        "fd", FakeDeviceInfo(address="none")
    )
    dialog = make_device_dialog(ctrl)

    dialog.accept()

    ctrl.cancel_device_operation.assert_not_called()
    assert ctrl._event_disposers
    for dispose in ctrl._event_disposers:
        dispose.assert_called_once()


def test_empty_device_list_shows_hint_and_drop_sits_under_list(qapp):
    ctrl = make_device_control()
    dialog = make_device_dialog(ctrl)
    dialog.show()
    qapp.processEvents()
    try:
        hint = dialog.findChild(QLabel, "devices_empty_hint")
        assert hint is not None and hint.isVisibleTo(dialog)
        listing = dialog.findChild(QListWidget)
        drop = next(b for b in dialog.findChildren(QPushButton) if b.text() == "Drop")
        assert listing is not None
        assert not drop.isEnabled()
        list_box = listing.geometry()
        drop_pos = drop.mapTo(dialog, drop.rect().topLeft())
        list_pos = listing.mapTo(dialog, listing.rect().bottomLeft())
        assert drop_pos.y() > list_pos.y()
        assert drop_pos.x() < list_pos.x() + list_box.width()
    finally:
        dialog.close()


def test_device_hint_hides_when_a_device_is_listed(qapp):
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = make_device_control()
    ctrl.list_devices.return_value = [device_entry("fakedevice")]
    ctrl.get_device_snapshot.return_value = connected_snapshot(
        "fakedevice", FakeDeviceInfo(address="none")
    )
    dialog = make_device_dialog(ctrl)
    dialog.show()
    qapp.processEvents()
    try:
        hint = dialog.findChild(QLabel, "devices_empty_hint")
        assert hint is not None and not hint.isVisibleTo(dialog)
    finally:
        dialog.close()


@pytest.mark.parametrize("failure", ["provider", "expression", "bounds", "snapshot"])
def test_device_error_uses_theme_foreground(qapp, window_palette, failure):
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = make_parameter_control()
    ctrl.list_devices.return_value = [device_entry("fd")]
    info = FakeDeviceInfo(address="none")
    ctrl.get_device_snapshot.return_value = connected_snapshot("fd", info)
    ctrl.get_device_info.return_value = info
    dialog = make_device_dialog(ctrl)
    try:
        dialog.setPalette(window_palette)
        if failure == "snapshot":
            ctrl.get_device_snapshot.return_value = DeviceSnapshot(
                name="fd",
                type_name="FakeDevice",
                address="none",
                status=DeviceStatus.CONNECTED,
                info=info,
                error="snapshot failed",
            )
        device_list(dialog).setCurrentRow(0)
        if failure == "snapshot":
            ctrl.get_bus().emit(DeviceChangedPayload(name="fd"))
        else:
            if failure == "provider":
                ctrl.read_context.side_effect = RuntimeError("capture failed")
            expression = "1e100" if failure == "bounds" else "unknown_name"
            labelled_field(
                device_panel(dialog), "Value:", EvalNumericField
            ).load_expression(expression)
            apply_button(dialog).click()
        label = next(
            label
            for label in dialog.findChildren(QLabel)
            if label.text().startswith("fd:") or label.text() == "snapshot failed"
        )
        label.ensurePolished()
        assert label.palette().color(label.foregroundRole()) == error_colour(label)
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize(
    "status, colour_name",
    [
        (DeviceStatus.CONNECTED, "success_colour"),
        (DeviceStatus.CONNECTING, "secondary_colour"),
    ],
)
def test_device_status_replaces_error_foreground(
    qapp, window_palette, status, colour_name
):
    from dataclasses import replace

    from zcu_tools.device.fake import FakeDeviceInfo
    from zcu_tools.gui.widgets import secondary_colour, success_colour

    colours: dict[str, Callable[[QWidget], QColor]] = {
        "success_colour": success_colour,
        "secondary_colour": secondary_colour,
    }
    colour = colours[colour_name]
    ctrl = make_device_control()
    ctrl.list_devices.return_value = [device_entry("fd")]
    snapshot = connected_snapshot("fd", FakeDeviceInfo(address="none"))
    ctrl.get_device_snapshot.return_value = snapshot
    dialog = make_device_dialog(ctrl)
    try:
        dialog.setPalette(window_palette)
        ctrl.get_device_snapshot.return_value = replace(
            snapshot, error="snapshot failed"
        )
        ctrl.get_bus().emit(DeviceChangedPayload(name="fd"))
        label = next(
            label
            for label in dialog.findChildren(QLabel)
            if label.text() == "snapshot failed"
        )
        ctrl.get_device_snapshot.return_value = replace(snapshot, status=status)
        ctrl.get_bus().emit(DeviceChangedPayload(name="fd"))
        label.ensurePolished()
        assert label.palette().color(label.foregroundRole()) == colour(dialog)
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


@contextmanager
def _activity_dialog(qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from zcu_tools.device import DeviceManager, FakeDevice, FakeDeviceInfo
    from zcu_tools.device.base import device_operation
    from zcu_tools.gui.app.measure.controller import Controller
    from zcu_tools.gui.app.measure.registry import Registry
    from zcu_tools.gui.app.measure.state import State
    from zcu_tools.gui.session.types import SessionEnv

    from tests.gui._user_input import ComponentDriver

    entered, release = Event(), Event()

    class HoldingDevice(FakeDevice):
        info_model = FakeDeviceInfo

        @device_operation
        def hold(self) -> None:
            entered.set()
            # The test releases this operation in finally, even on assertion failure.
            assert release.wait(30)

    clock = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    flux, unused = HoldingDevice(fast_mode=True), FakeDevice(fast_mode=True)
    manager = DeviceManager()
    manager.register_device("flux", flux)
    manager.register_device("unused", unused)
    ctrl = Controller(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        Registry(),
        None,
        EventBus(),
        project_root=str(tmp_path),
        device_registry=manager,
    )
    dialog = DeviceDialog(ctrl.device_control, parameter_provider=lambda: None)
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        dialog.show()
        qapp.processEvents()
        yield ComponentDriver(dialog), flux, clock, entered, release, pool
    finally:
        release.set()
        pool.shutdown(wait=True)
        ctrl.close(timeout_ms=30_000)
        dialog.close()
        dialog.deleteLater()
        manager.close_all_devices()


def test_driver_activity_marks_only_operated_rows_and_holds_between_short_writes(
    qapp, poll_timers, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _activity_dialog(qapp, monkeypatch, tmp_path) as (
        ui,
        flux,
        clock,
        entered,
        release,
        pool,
    ):
        ui.select_list_row("unused (FakeDevice)")
        ui.screenshot(tmp_path / "01-idle.png")
        future = pool.submit(flux.hold)
        assert entered.wait(30), "submitted fake operation must enter before release"
        for timer in poll_timers:
            timer.tick()
        rows = ui.list_rows()
        assert [row.text for row in rows] == [
            "● flux (FakeDevice)",
            "unused (FakeDevice)",
        ]
        assert rows[0].tooltip == "Operating or recently active"
        assert rows[1].selected
        assert ui.observe("Apply Changes").enabled, (
            "activity is observation, not admission"
        )
        ui.screenshot(tmp_path / "02-operation.png")
        release.set()
        future.result(timeout=30)
        for index, now in enumerate((100.4, 100.8, 101.2)):
            clock[0] = now
            flux.set_value(0.1)
            for timer in poll_timers:
                if timer.interval < 1000:
                    timer.tick()
            assert ui.list_rows()[0].text == "● flux (FakeDevice)"
            assert ui.list_rows()[1].selected
            ui.screenshot(tmp_path / f"03-short-write-{index}.png")
        clock[0] = 102.0
        for timer in poll_timers:
            if timer.interval < 1000:
                timer.tick()
        assert ui.list_rows()[0].text == "● flux (FakeDevice)"
        ui.screenshot(tmp_path / "04-hold.png")
        clock[0] = 102.4
        for timer in poll_timers:
            if timer.interval < 1000:
                timer.tick()
        assert [row.text for row in ui.list_rows()] == [
            "flux (FakeDevice)",
            "unused (FakeDevice)",
        ]
        assert ui.list_rows()[1].selected
        ui.screenshot(tmp_path / "05-idle-again.png")
        ui.click("Close")
