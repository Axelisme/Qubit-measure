from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from qtpy.QtCore import QThread, Signal  # type: ignore[attr-defined]

from zcu_tools.device import GlobalDeviceManager
from zcu_tools.progress_bar import use_pbar_factory


class _DeviceSetupWorker(QThread):
    """Runs dev.setup(info) on a background thread with optional pbar factory and cancellation."""

    finished: Signal = Signal(str)  # name
    failed: Signal = Signal(str, str)  # name, error_message
    cancelled: Signal = Signal(str)  # name

    def __init__(
        self,
        dev: Any,
        name: str,
        info: Any,
        pbar_factory: Optional[Callable[..., Any]],
        parent: Any = None,
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


class DeviceManager:
    """Thin wrapper around GlobalDeviceManager for GUI use."""

    def register_device(self, name: str, device: Any) -> None:
        """Register an already-constructed device instance."""
        GlobalDeviceManager.register_device(name, device)

    def drop_device(self, name: str) -> None:
        GlobalDeviceManager.drop_device(name)

    def list_devices(self) -> dict[str, str]:
        """Return {name: device_type_string} for display."""
        devices = GlobalDeviceManager.get_all_devices()
        return {name: type(dev).__name__ for name, dev in devices.items()}

    def get_device_value(self, name: str) -> Any:
        """Read the current value from a device via get_value()."""
        dev = GlobalDeviceManager.get_device(name)
        return dev.get_value()  # type: ignore[attr-defined]

    def set_device_value(self, name: str, value: Any) -> Any:
        """Set a device value via set_value(); return the actual value applied."""
        dev = GlobalDeviceManager.get_device(name)
        return dev.set_value(value)  # type: ignore[attr-defined]

    def get_device_info(self, name: str) -> Any:
        """Return the DeviceInfo object for a registered device."""
        dev = GlobalDeviceManager.get_device(name)
        return dev.get_info()  # type: ignore[attr-defined]

    def setup_device(
        self,
        name: str,
        info: Any,
        pbar_factory: Optional[Callable[..., Any]] = None,
    ) -> _DeviceSetupWorker:
        """Start an async worker that applies info to the device; return the worker."""
        dev = GlobalDeviceManager.get_device(name)
        worker = _DeviceSetupWorker(dev, name, info, pbar_factory)
        worker.start()
        return worker

    def get_all_info(self) -> dict[str, Any]:
        """Delegate to GlobalDeviceManager.get_all_info()."""
        return GlobalDeviceManager.get_all_info()
