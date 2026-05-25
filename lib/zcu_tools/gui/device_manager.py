from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional, Protocol, cast, runtime_checkable

logger = logging.getLogger(__name__)

from qtpy.QtCore import QObject, QThread, Signal  # type: ignore[attr-defined]


@runtime_checkable
class DeviceProtocol(Protocol):
    def setup(
        self,
        cfg: Any,
        *,
        progress: bool = True,
        stop_event: Optional[threading.Event] = None,
    ) -> None: ...
    def get_info(self) -> object: ...


@runtime_checkable
class ValueDeviceProtocol(DeviceProtocol, Protocol):
    def get_value(self) -> object: ...
    def set_value(self, value: object) -> object: ...


class _DeviceSetupWorker(QThread):
    """Runs dev.setup(info) on a background thread with optional pbar factory and cancellation."""

    finished: Signal = Signal(str)  # name
    failed: Signal = Signal(str, str)  # name, error_message
    cancelled: Signal = Signal(str)  # name

    def __init__(
        self,
        dev: DeviceProtocol,
        name: str,
        info: Any,
        pbar_factory: Optional[Callable[..., Any]],
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._dev = dev
        self._name = name
        self._info = info
        self._pbar_factory = pbar_factory
        self._stop_event = threading.Event()

    def cancel(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        from zcu_tools.progress_bar import use_pbar_factory

        try:
            if self._pbar_factory is not None:
                with use_pbar_factory(self._pbar_factory):
                    self._dev.setup(self._info, stop_event=self._stop_event)
            else:
                self._dev.setup(self._info, stop_event=self._stop_event)

            if self._stop_event.is_set():
                self.cancelled.emit(self._name)
            else:
                self.finished.emit(self._name)
        except Exception as exc:
            self.failed.emit(self._name, str(exc))


class DeviceManager(QObject):
    """Thin wrapper around GlobalDeviceManager for GUI use."""

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)

    def register_device(self, name: str, device: DeviceProtocol) -> None:
        """Register an already-constructed device instance."""
        from zcu_tools.device import GlobalDeviceManager

        logger.info("register_device: name=%r type=%s", name, type(device).__name__)
        GlobalDeviceManager.register_device(name, device)

    def drop_device(self, name: str) -> None:
        from zcu_tools.device import GlobalDeviceManager

        logger.info("drop_device: name=%r", name)
        GlobalDeviceManager.drop_device(name)

    def list_devices(self) -> dict[str, str]:
        """Return {name: device_type_string} for display."""
        from zcu_tools.device import GlobalDeviceManager

        devices = GlobalDeviceManager.get_all_devices()
        return {name: type(dev).__name__ for name, dev in devices.items()}

    def get_device_value(self, name: str) -> object:
        """Read the current value from a device via get_value()."""
        from zcu_tools.device import GlobalDeviceManager

        dev = cast(ValueDeviceProtocol, GlobalDeviceManager.get_device(name))
        return dev.get_value()

    def set_device_value(self, name: str, value: Any) -> object:
        """Set a device value via set_value(); return the actual value applied."""
        from zcu_tools.device import GlobalDeviceManager

        dev = cast(ValueDeviceProtocol, GlobalDeviceManager.get_device(name))
        return dev.set_value(value)

    def get_device_info(self, name: str) -> object | None:
        """Return the DeviceInfo object for a registered device, or None if not found."""
        from zcu_tools.device import GlobalDeviceManager

        try:
            dev = GlobalDeviceManager.get_device(name)
        except ValueError:
            return None
        return dev.get_info()

    def setup_device(
        self,
        name: str,
        info: Any,
        pbar_factory: Optional[Callable[..., Any]] = None,
    ) -> _DeviceSetupWorker:
        """Start an async worker that applies info to the device; return the worker."""
        from zcu_tools.device import GlobalDeviceManager

        dev = GlobalDeviceManager.get_device(name)
        worker = _DeviceSetupWorker(dev, name, info, pbar_factory, parent=self)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        worker.start()
        return worker

    def get_all_info(self) -> dict[str, Any]:
        """Delegate to GlobalDeviceManager.get_all_info()."""
        from zcu_tools.device import GlobalDeviceManager

        return GlobalDeviceManager.get_all_info()
