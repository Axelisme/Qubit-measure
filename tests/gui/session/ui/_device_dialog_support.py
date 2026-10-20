"""Shared DeviceDialog lookups and per-case control builders."""

from __future__ import annotations

from unittest.mock import MagicMock

from qtpy.QtWidgets import (
    QFormLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QStackedWidget,
    QWidget,
)
from zcu_tools.device import BaseDeviceInfo
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.device_models import DeviceEntry, DeviceSnapshot
from zcu_tools.gui.session.events import (
    DeviceChangedPayload,
    DeviceSetupFinishedPayload,
    DeviceSetupStartedPayload,
)
from zcu_tools.gui.session.state import DeviceStatus
from zcu_tools.gui.session.ui.device_dialog import DeviceDialog

from tests.gui.session.ui._parameter_fakes import make_parameters


def device_list(dialog: DeviceDialog) -> QListWidget:
    """Return the dialog device list; assert if it is missing."""
    listing = dialog.findChild(QListWidget)
    assert listing is not None
    return listing


def device_panel(dialog: DeviceDialog) -> QWidget:
    """Return the selected device panel; assert if the stack or panel is missing."""
    stack = dialog.findChild(QStackedWidget)
    assert stack is not None
    panel = stack.currentWidget()
    assert panel is not None
    return panel


def apply_button(dialog: DeviceDialog) -> QPushButton:
    """Return Apply Changes, Stop or Reconnect; raise StopIteration if absent."""
    return next(
        b
        for b in dialog.findChildren(QPushButton)
        if b.text() in {"Apply Changes", "Stop", "Reconnect"}
    )


def labelled_field[T: QWidget](root: QWidget, label: str, kind: type[T]) -> T:
    """Return a field of kind with the exact form label, or raise AssertionError."""
    for field in root.findChildren(kind):
        parent = field.parentWidget()
        assert parent is not None
        layout = parent.layout()
        if isinstance(layout, QFormLayout):
            caption = layout.labelForField(field)
            if isinstance(caption, QLabel) and caption.text() == label:
                return field
    raise AssertionError(f"No field labelled {label!r}")


def device_entry(
    name: str, type_name: str = "FakeDevice", connected: bool = True
) -> DeviceEntry:
    """Build a connected or memory-only listing entry for the given name and type."""
    # DeviceEntry now carries the fine-grained status string (FC7) instead of a
    # coarse is_connected bool: a connected device maps to "connected", a
    # remembered-but-not-live one to "memory_only".
    status = (
        DeviceStatus.CONNECTED.value if connected else DeviceStatus.MEMORY_ONLY.value
    )
    return DeviceEntry(name=name, type_name=type_name, status=status)


def make_device_control() -> MagicMock:
    """Build a fresh control fake with its own event bus and subscription disposers."""
    ctrl = MagicMock()
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl._event_disposers = []

    def _subscribe(payload_type: type, handler):
        handle = bus.subscribe(payload_type, handler)
        dispose = MagicMock(side_effect=handle.unsubscribe)
        ctrl._event_disposers.append(dispose)
        return dispose

    ctrl.on_device_changed.side_effect = lambda handler: _subscribe(
        DeviceChangedPayload, handler
    )
    ctrl.on_device_setup_started.side_effect = lambda handler: _subscribe(
        DeviceSetupStartedPayload, handler
    )
    ctrl.on_device_setup_finished.side_effect = lambda handler: _subscribe(
        DeviceSetupFinishedPayload, handler
    )
    ctrl.progress_bars.return_value = ()
    ctrl.list_devices.return_value = []
    ctrl.is_memory_device.return_value = False
    ctrl.get_device_snapshot.return_value = None
    ctrl.activity_snapshot.return_value = {}
    ctrl.try_poll_device_info.return_value = True
    ctrl.read_context.return_value = None
    return ctrl


def make_device_dialog(ctrl: MagicMock) -> DeviceDialog:
    """Build a dialog using ctrl and its separate read_context parameter provider."""
    return DeviceDialog(ctrl, parameter_provider=ctrl.read_context)


def connected_snapshot(name: str, info: BaseDeviceInfo) -> DeviceSnapshot:
    """Build a CONNECTED snapshot for name using the supplied device info."""
    return DeviceSnapshot(
        name=name,
        type_name=getattr(info, "type", "FakeDevice"),
        address=getattr(info, "address", ""),
        status=DeviceStatus.CONNECTED,
        info=info,
    )


def make_parameter_control(flx_int: float = 0.5) -> MagicMock:
    """Build a device control fake with a separate fixed parameter provider."""
    ctrl = make_device_control()
    ctrl.read_context.return_value = make_parameters({"flx_int": flx_int})
    return ctrl
