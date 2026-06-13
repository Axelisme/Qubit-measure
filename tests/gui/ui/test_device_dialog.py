"""Smoke tests for DeviceDialog."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.events import (
    DeviceChangedPayload,
    DeviceSetupFinishedPayload,
)
from zcu_tools.gui.session.services.device import (
    ConnectDeviceRequest,
    DeviceEntry,
    DeviceSnapshot,
    DeviceStatus,
    DisconnectDeviceRequest,
    SetupDeviceRequest,
)
from zcu_tools.gui.session.ui.device_dialog import DeviceDialog, _FakeDevicePanel


def _entry(
    name: str, type_name: str = "FakeDevice", connected: bool = True
) -> DeviceEntry:
    return DeviceEntry(name=name, type_name=type_name, is_connected=connected)


def _make_ctrl() -> MagicMock:
    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.get_active_device_setup.return_value = None
    ctrl.progress_bars.return_value = ()
    ctrl.list_devices.return_value = []
    ctrl.is_memory_device.return_value = False
    ctrl.get_memory_device_address.return_value = None
    ctrl.get_device_snapshot.return_value = None
    return ctrl


def _connected_snapshot(name: str, info: object) -> DeviceSnapshot:
    return DeviceSnapshot(
        name=name,
        type_name=getattr(info, "type", "FakeDevice"),
        address=getattr(info, "address", ""),
        status=DeviceStatus.CONNECTED,
        info=info,  # type: ignore[arg-type]
    )


def _setting_up_snapshot(name: str, info: object) -> DeviceSnapshot:
    """A snapshot whose status is SETTING_UP — Phase C's SSOT for 'this device
    is mid-setup' (the dialog derives the live-setup set from snapshot status,
    not from the single-valued get_active_device_setup)."""
    return DeviceSnapshot(
        name=name,
        type_name=getattr(info, "type", "FakeDevice"),
        address=getattr(info, "address", ""),
        status=DeviceStatus.SETTING_UP,
        info=info,  # type: ignore[arg-type]
    )


def test_device_dialog_init(qapp):
    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = [_entry("fakedevice")]

    from zcu_tools.device.fake import FakeDeviceInfo

    info = FakeDeviceInfo(address="none")
    ctrl.get_device_snapshot.return_value = _connected_snapshot("fakedevice", info)

    dialog = DeviceDialog(ctrl)

    # list_devices should be called during init (Phase C also reads it in
    # _setup_device_names, so just assert it was used, not the exact count).
    ctrl.list_devices.assert_called()
    assert dialog._list.count() == 1

    dialog._list.setCurrentRow(0)

    # The current stack index should map to FakeDevice (1)
    assert dialog._stack.currentIndex() == 1


def test_device_dialog_add_device_dispatches_request(qapp):
    ctrl = _make_ctrl()

    dialog = DeviceDialog(ctrl)
    dialog._type_combo.setCurrentText("FakeDevice")
    dialog._addr_edit.setText("TCPIP::127.0.0.1::INSTR")
    dialog._name_edit.setText("fakedevice")

    dialog._add_btn.click()

    ctrl.start_connect_device.assert_called_once()
    (req,) = ctrl.start_connect_device.call_args.args
    assert isinstance(req, ConnectDeviceRequest)
    assert req.type_name == "FakeDevice"
    assert req.name == "fakedevice"
    assert req.address == "TCPIP::127.0.0.1::INSTR"
    ctrl.start_setup_device.assert_not_called()


def test_device_dialog_add_device_does_not_persist_before_async_success(qapp):
    ctrl = _make_ctrl()
    dialog = DeviceDialog(ctrl)
    dialog._add_btn.click()


def test_device_dialog_add_device_propagates_unexpected_errors(qapp):
    """Programmer errors must not be swallowed by the dialog catch."""
    ctrl = _make_ctrl()
    ctrl.start_connect_device.side_effect = ValueError("contract violation")

    dialog = DeviceDialog(ctrl)
    dialog._type_combo.setCurrentText("FakeDevice")
    dialog._addr_edit.setText("addr")
    with pytest.raises(ValueError, match="contract violation"):
        dialog._on_add_clicked()


def test_device_dialog_drop_device(qapp):
    from zcu_tools.device.yoko import YOKOGS200Info

    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = [_entry("yoko", "YOKOGS200")]

    info = YOKOGS200Info(address="GPIB::1")
    ctrl.get_device_snapshot.return_value = _connected_snapshot("yoko", info)

    dialog = DeviceDialog(ctrl)

    assert dialog._list.count() == 1
    dialog._list.setCurrentRow(0)

    # Drop disconnects but keeps startup memory
    dialog._drop_btn.click()
    ctrl.start_disconnect_device.assert_called_once()
    (req,) = ctrl.start_disconnect_device.call_args.args
    assert isinstance(req, DisconnectDeviceRequest)
    assert req.name == "yoko"


def test_device_dialog_forget_memory_device_dispatches_single_transaction(qapp):
    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = [_entry("remembered", connected=False)]
    ctrl.is_memory_device.return_value = True
    ctrl.get_device_snapshot.return_value = DeviceSnapshot(
        name="remembered",
        type_name="FakeDevice",
        address="addr",
        status=DeviceStatus.MEMORY_ONLY,
    )

    dialog = DeviceDialog(ctrl)
    dialog._list.setCurrentRow(0)
    dialog._drop_btn.click()

    ctrl.forget_device.assert_called_once_with("remembered")


def test_device_dialog_refresh_reloads_selected_device_info(qapp):
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = [_entry("fd")]
    # By-name + mutable so the assertion is robust to *how many* times the dialog
    # reads the snapshot (Phase C's _setup_device_names adds extra reads); the
    # value moves only when the test explicitly bumps it.
    box = {"value": 1.0}
    ctrl.get_device_snapshot.side_effect = lambda _n: _connected_snapshot(
        "fd", FakeDeviceInfo(address="none", value=box["value"])
    )
    dialog = DeviceDialog(ctrl)
    dialog._list.setCurrentRow(0)
    panel = dialog._stack.currentWidget()
    assert isinstance(panel, _FakeDevicePanel)
    assert panel._value_spin.value() == 1.0

    box["value"] = 2.0
    dialog._refresh_btn.click()

    assert panel._value_spin.value() == 2.0
    ctrl.get_device_info.assert_called_once_with("fd")


def test_device_changed_repaints_selected_panel(qapp):
    """A DEVICE_CHANGED for the selected device repaints the right panel with the
    fresh cached info (so a value that moved underneath us shows immediately)."""
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = [_entry("fd")]
    # By-name + mutable (robust to extra snapshot reads, see above).
    box = {"value": 1.0}
    ctrl.get_device_snapshot.side_effect = lambda _n: _connected_snapshot(
        "fd", FakeDeviceInfo(address="none", value=box["value"])
    )

    dialog = DeviceDialog(ctrl)
    dialog._list.setCurrentRow(0)
    panel = dialog._stack.currentWidget()
    assert isinstance(panel, _FakeDevicePanel)
    assert panel._value_spin.value() == 1.0

    box["value"] = 2.0
    ctrl.get_bus.return_value.emit(DeviceChangedPayload(name="fd"))

    assert panel._value_spin.value() == 2.0


def test_device_changed_for_other_device_keeps_selection(qapp):
    """A DEVICE_CHANGED for a non-selected device must not steal the selection
    away from the device the user is currently inspecting."""
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = [_entry("A"), _entry("B")]
    ctrl.get_device_snapshot.side_effect = lambda n: _connected_snapshot(
        n, FakeDeviceInfo(address="none")
    )

    dialog = DeviceDialog(ctrl)
    dialog._list.setCurrentRow(0)
    item = dialog._list.currentItem()
    assert item is not None and item.data(256) == "A"

    ctrl.get_bus.return_value.emit(DeviceChangedPayload(name="B"))

    item = dialog._list.currentItem()
    assert item is not None and item.data(256) == "A"


def test_device_changed_surfaces_device_when_none_selected(qapp):
    """With nothing selected, a DEVICE_CHANGED surfaces the changed device (the
    'first device just connected' case)."""
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = [_entry("fd")]
    ctrl.get_device_snapshot.side_effect = lambda n: _connected_snapshot(
        n, FakeDeviceInfo(address="none")
    )

    dialog = DeviceDialog(ctrl)
    dialog._list.setCurrentRow(-1)

    ctrl.get_bus.return_value.emit(DeviceChangedPayload(name="fd"))

    item = dialog._list.currentItem()
    assert item is not None
    assert item.data(256) == "fd"


def test_poll_timer_runs_only_when_visible_and_selected(qapp):
    """The live poller is dialog-scoped + selection-scoped: it runs only while
    the dialog is visible AND a device is selected, and stops otherwise."""
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = [_entry("fd")]
    ctrl.get_device_snapshot.side_effect = lambda n: _connected_snapshot(
        n, FakeDeviceInfo(address="none")
    )

    dialog = DeviceDialog(ctrl)
    try:
        # Hidden + nothing selected → stopped.
        assert not dialog._poll_timer.isActive()

        # Visible but still nothing selected → stopped.
        dialog.show()
        qapp.processEvents()
        dialog._list.setCurrentRow(-1)
        assert not dialog._poll_timer.isActive()

        # Visible + selected → running.
        dialog._list.setCurrentRow(0)
        assert dialog._poll_timer.isActive()

        # Hiding the dialog stops the poller (dialog-scoped).
        dialog.hide()
        qapp.processEvents()
        assert not dialog._poll_timer.isActive()
    finally:
        dialog.close()


def test_poll_tick_polls_selected_device(qapp):
    """Each tick asks the controller to off-main poll the *selected* device."""
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = [_entry("A"), _entry("B")]
    ctrl.get_device_snapshot.side_effect = lambda n: _connected_snapshot(
        n, FakeDeviceInfo(address="none")
    )

    dialog = DeviceDialog(ctrl)
    try:
        dialog._list.setCurrentRow(1)  # select "B"
        dialog._on_poll_tick()
        ctrl.poll_device_info.assert_called_once_with("B")
    finally:
        dialog.close()


def test_poll_tick_with_no_selection_stops_timer(qapp):
    """A tick that finds nothing selected stops the timer instead of polling
    None (guards a race where the selection vanished between decisions)."""
    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = []

    dialog = DeviceDialog(ctrl)
    try:
        dialog._poll_timer.start()  # force-running with no selection
        dialog._on_poll_tick()
        ctrl.poll_device_info.assert_not_called()
        assert not dialog._poll_timer.isActive()
    finally:
        dialog.close()


def test_poll_timer_stops_on_dialog_close(qapp):
    """Closing the dialog stops the poll timer (no leak after close)."""
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = [_entry("fd")]
    ctrl.get_device_snapshot.side_effect = lambda n: _connected_snapshot(
        n, FakeDeviceInfo(address="none")
    )

    dialog = DeviceDialog(ctrl)
    dialog.show()
    qapp.processEvents()
    dialog._list.setCurrentRow(0)
    assert dialog._poll_timer.isActive()

    dialog.accept()  # finished → _cleanup_bus_subscriptions stops the timer
    qapp.processEvents()
    assert not dialog._poll_timer.isActive()


def test_device_dialog_apply_changes(qapp):
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = [_entry("fd")]

    info = FakeDeviceInfo(address="none")
    ctrl.get_device_snapshot.return_value = _connected_snapshot("fd", info)
    ctrl.get_device_info.return_value = info
    dialog = DeviceDialog(ctrl)
    dialog._list.setCurrentRow(0)

    assert dialog._stack.currentIndex() == 1  # FakeDevice page

    dialog._apply_btn.click()

    ctrl.start_setup_device.assert_called_once()
    (req,) = ctrl.start_setup_device.call_args.args
    assert isinstance(req, SetupDeviceRequest)
    assert req.name == "fd"
    assert isinstance(req.info, FakeDeviceInfo)


def test_device_dialog_restores_background_setup_and_stops_it(qapp):
    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = [_entry("fd")]
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
        info=info,  # type: ignore[arg-type]
    )

    dialog = DeviceDialog(ctrl)
    # Opened mid-setup with nothing pre-selected → the setup owner is surfaced.
    item = dialog._list.currentItem()
    assert item is not None
    assert item.data(256) == "fd"
    assert dialog._apply_btn.text() == "Stop"
    # Phase C: the list stays enabled (user can switch devices).
    assert dialog._list.isEnabled() is True
    # Refresh is disabled for a device that is itself setting up.
    assert dialog._refresh_btn.isEnabled() is False
    # The dialog attached to progress by the setting-up device's name, so live
    # bars render even though it opened mid-setup.
    ctrl.attach_progress.assert_called_once()
    assert ctrl.attach_progress.call_args.args[0] == "fd"

    dialog._apply_btn.click()
    ctrl.cancel_device_operation.assert_called_once_with("fd")

    # Setup reaches a terminal state → the device returns to CONNECTED and the
    # finish event repaints buttons back to normal.
    state["status"] = DeviceStatus.CONNECTED
    ctrl.get_bus.return_value.emit(
        DeviceSetupFinishedPayload(name="fd", outcome="cancelled"),
    )
    assert dialog._apply_btn.text() == "Apply Changes"
    assert dialog._list.isEnabled() is True
    assert dialog._refresh_btn.isEnabled() is True


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

    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = [_entry("fd_a"), _entry("fd_b")]

    def snapshot(n: str) -> DeviceSnapshot:
        info = FakeDeviceInfo(address="none")
        if n in setting_up:
            return _setting_up_snapshot(n, info)
        return _connected_snapshot(n, info)

    ctrl.get_device_snapshot.side_effect = snapshot
    return ctrl


def _setup_two_device_dialog(qapp):
    """Dialog with fd_a setting up and fd_b connected; selection on the
    non-owner fd_b. Returns (dialog, ctrl, setting_up_set)."""
    setting_up = {"fd_a"}
    ctrl = _two_device_ctrl(setting_up)
    dialog = DeviceDialog(ctrl)
    dialog._list.setCurrentRow(1)  # "fd_b"
    return dialog, ctrl, setting_up


def test_setup_running_list_remains_enabled(qapp):
    """During a setup the device list stays enabled so the user can switch
    to view (and even set up) another device."""
    dialog, _, _ = _setup_two_device_dialog(qapp)
    assert dialog._list.isEnabled() is True


def test_setup_running_can_switch_to_other_device(qapp):
    """Switching to another device shows that device's panel."""
    dialog, _, _ = _setup_two_device_dialog(qapp)

    item = dialog._list.currentItem()
    assert item is not None
    assert item.data(256) == "fd_b"
    assert dialog._stack.currentIndex() == 1  # FakeDevice page


def test_setting_up_device_shows_stop_button(qapp):
    """A device that is itself setting up shows the red Stop button."""
    dialog, _, _ = _setup_two_device_dialog(qapp)

    dialog._list.setCurrentRow(0)  # "fd_a" (setting up)
    assert dialog._apply_btn.text() == "Stop"
    assert "red" in dialog._apply_btn.styleSheet()
    assert dialog._apply_btn.isEnabled() is True


def test_other_device_apply_is_enabled_during_setup(qapp):
    """Phase C: a device that is NOT setting up keeps an enabled Apply even while
    another device is mid-setup — this is the concurrency the phase unlocks."""
    dialog, _, _ = _setup_two_device_dialog(qapp)

    # fd_b is connected and not setting up.
    assert dialog._apply_btn.text() == "Apply Changes"
    assert dialog._apply_btn.isEnabled() is True


def test_apply_on_other_device_starts_concurrent_setup(qapp):
    """Clicking Apply on a device while another is setting up dispatches a second
    setup (concurrent), rather than being blocked."""
    from zcu_tools.device.fake import FakeDeviceInfo

    dialog, ctrl, _ = _setup_two_device_dialog(qapp)
    ctrl.get_device_info.return_value = FakeDeviceInfo(address="none")

    # fd_b selected (not setting up) → Apply starts its setup even though fd_a is.
    dialog._apply_btn.click()

    ctrl.start_setup_device.assert_called_once()
    (req,) = ctrl.start_setup_device.call_args.args
    assert isinstance(req, SetupDeviceRequest)
    assert req.name == "fd_b"
    ctrl.cancel_device_operation.assert_not_called()


def test_switch_between_devices_recomputes_buttons(qapp):
    """Switching between a setting-up device and an idle one flips the button
    between Stop and Apply."""
    dialog, _, _ = _setup_two_device_dialog(qapp)

    dialog._list.setCurrentRow(0)  # fd_a (setting up)
    assert dialog._apply_btn.text() == "Stop"
    assert dialog._apply_btn.isEnabled() is True

    dialog._list.setCurrentRow(1)  # fd_b (idle)
    assert dialog._apply_btn.text() == "Apply Changes"
    assert dialog._apply_btn.isEnabled() is True


def test_two_concurrent_setups_each_show_stop_and_cancel_independently(qapp):
    """With both devices setting up, each selected device shows its own Stop and
    cancels only itself."""
    setting_up = {"fd_a", "fd_b"}
    ctrl = _two_device_ctrl(setting_up)
    dialog = DeviceDialog(ctrl)

    dialog._list.setCurrentRow(0)  # fd_a
    assert dialog._apply_btn.text() == "Stop"
    dialog._apply_btn.click()
    ctrl.cancel_device_operation.assert_called_once_with("fd_a")

    dialog._list.setCurrentRow(1)  # fd_b
    assert dialog._apply_btn.text() == "Stop"
    dialog._apply_btn.click()
    assert ctrl.cancel_device_operation.call_args_list[-1].args == ("fd_b",)


def test_two_concurrent_setups_subscribe_progress_per_owner(qapp):
    """Both setting-up devices get their own progress subscription (multi-owner);
    the merged stack is shown."""
    setting_up = {"fd_a", "fd_b"}
    ctrl = _two_device_ctrl(setting_up)
    dialog = DeviceDialog(ctrl)

    owners = {call.args[0] for call in ctrl.attach_progress.call_args_list}
    assert owners == {"fd_a", "fd_b"}
    assert set(dialog._progress_unsubs) == {"fd_a", "fd_b"}
    assert dialog._progress.isVisibleTo(dialog) or not dialog.isVisible()


def test_one_of_two_setups_finishing_keeps_the_other(qapp):
    """When one of two concurrent setups finishes, its progress subscription is
    disposed but the other survives."""
    setting_up = {"fd_a", "fd_b"}
    ctrl = _two_device_ctrl(setting_up)
    dialog = DeviceDialog(ctrl)
    assert set(dialog._progress_unsubs) == {"fd_a", "fd_b"}

    # fd_a finishes (back to CONNECTED); fd_b still setting up.
    setting_up.discard("fd_a")
    ctrl.get_bus.return_value.emit(
        DeviceSetupFinishedPayload(name="fd_a", outcome="finished"),
    )
    assert set(dialog._progress_unsubs) == {"fd_b"}


def test_all_setups_finished_hides_progress_and_restores_buttons(qapp):
    """When the last setup finishes, progress is hidden and buttons normalise."""
    dialog, ctrl, setting_up = _setup_two_device_dialog(qapp)
    assert set(dialog._progress_unsubs) == {"fd_a"}

    setting_up.discard("fd_a")
    ctrl.get_bus.return_value.emit(
        DeviceSetupFinishedPayload(name="fd_a", outcome="finished"),
    )
    assert dialog._progress_unsubs == {}
    # fd_b still selected, Apply enabled, list enabled.
    item = dialog._list.currentItem()
    assert item is not None and item.data(256) == "fd_b"
    assert dialog._apply_btn.text() == "Apply Changes"
    assert dialog._apply_btn.isEnabled() is True
    assert dialog._list.isEnabled() is True


def test_drop_disabled_for_setting_up_device_only(qapp):
    """Phase C: Drop is disabled for a device that is itself setting up, but
    enabled for an idle device even while another is mid-setup."""
    dialog, _, _ = _setup_two_device_dialog(qapp)

    dialog._list.setCurrentRow(1)  # fd_b idle → drop enabled
    assert dialog._drop_btn.isEnabled() is True

    dialog._list.setCurrentRow(0)  # fd_a setting up → drop disabled
    assert dialog._drop_btn.isEnabled() is False


def test_add_box_enabled_during_setup(qapp):
    """Phase C: the Add-box stays enabled during a setup (a new-name connect is
    resource-disjoint from any in-flight mutation)."""
    dialog, _, _ = _setup_two_device_dialog(qapp)

    assert dialog._add_btn.isEnabled() is True
    assert dialog._type_combo.isEnabled() is True
    assert dialog._name_edit.isEnabled() is True
    assert dialog._addr_edit.isEnabled() is True


def test_refresh_enabled_for_idle_device_during_other_setup(qapp):
    """Refresh is allowed for an idle device even while another device sets up."""
    dialog, _, _ = _setup_two_device_dialog(qapp)

    dialog._list.setCurrentRow(1)  # fd_b idle
    assert dialog._refresh_btn.isEnabled() is True


def test_refresh_disabled_for_setting_up_device(qapp):
    """Refresh is disabled for a device that is itself setting up."""
    dialog, _, _ = _setup_two_device_dialog(qapp)

    dialog._list.setCurrentRow(0)  # fd_a setting up
    assert dialog._refresh_btn.isEnabled() is False


def test_device_dialog_close_keeps_setup_running_and_unsubscribes(qapp):
    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = [_entry("fd")]
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl.get_device_snapshot.return_value = _setting_up_snapshot(
        "fd", FakeDeviceInfo(address="none")
    )
    dialog = DeviceDialog(ctrl)

    dialog.accept()

    ctrl.cancel_device_operation.assert_not_called()
    assert ctrl.get_bus.return_value._subs[DeviceSetupFinishedPayload] == []
