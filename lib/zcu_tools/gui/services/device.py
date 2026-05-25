from __future__ import annotations

import logging
import threading
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Protocol,
    cast,
    runtime_checkable,
)

logger = logging.getLogger(__name__)

from qtpy.QtCore import QObject, QThread, Signal  # type: ignore[attr-defined]

from zcu_tools.gui.event_bus import DeviceChangedPayload, GuiEvent

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import EventBus
    from zcu_tools.gui.state import State


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


class DeviceService(QObject):
    """Device registration, setup, and value access for the GUI."""

    def __init__(
        self, state: "State", bus: "EventBus", parent: Optional[QObject] = None
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._bus = bus

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_device(self, name: str, device: DeviceProtocol) -> None:
        logger.info("register_device: name=%r type=%s", name, type(device).__name__)
        if self._state.is_run_active():
            raise RuntimeError("Cannot register device while a run is active")
        from zcu_tools.device import GlobalDeviceManager

        GlobalDeviceManager.register_device(name, device)
        self._bus.emit(GuiEvent.DEVICE_CHANGED, DeviceChangedPayload())

    def drop_device(self, name: str) -> None:
        logger.info("drop_device: name=%r", name)
        if self._state.is_run_active():
            raise RuntimeError("Cannot drop device while a run is active")
        from zcu_tools.device import GlobalDeviceManager

        GlobalDeviceManager.drop_device(name)
        self._bus.emit(GuiEvent.DEVICE_CHANGED, DeviceChangedPayload())

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list_devices(self) -> dict[str, str]:
        from zcu_tools.device import GlobalDeviceManager

        devices = GlobalDeviceManager.get_all_devices()
        return {name: type(dev).__name__ for name, dev in devices.items()}

    def get_device_info(self, name: str) -> object | None:
        from zcu_tools.device import GlobalDeviceManager

        try:
            dev = GlobalDeviceManager.get_device(name)
        except ValueError:
            return None
        return dev.get_info()

    def get_device_value(self, name: str) -> object:
        from zcu_tools.device import GlobalDeviceManager

        dev = cast(ValueDeviceProtocol, GlobalDeviceManager.get_device(name))
        return dev.get_value()

    def set_device_value(self, name: str, value: Any) -> object:
        if self._state.is_run_active():
            raise RuntimeError("Cannot set device value while a run is active")
        from zcu_tools.device import GlobalDeviceManager

        dev = cast(ValueDeviceProtocol, GlobalDeviceManager.get_device(name))
        return dev.set_value(value)

    # ------------------------------------------------------------------
    # Setup (async)
    # ------------------------------------------------------------------

    def setup_device(
        self,
        name: str,
        info: Any,
        pbar_factory: Optional[Callable[..., Any]] = None,
    ) -> _DeviceSetupWorker:
        if self._state.is_run_active():
            raise RuntimeError("Cannot setup device while a run is active")
        from zcu_tools.device import GlobalDeviceManager

        dev = GlobalDeviceManager.get_device(name)
        worker = _DeviceSetupWorker(dev, name, info, pbar_factory, parent=self)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        worker.start()
        return worker
