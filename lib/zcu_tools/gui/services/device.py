from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
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

from zcu_tools.device.base import BaseDeviceInfo
from zcu_tools.gui.event_bus import (
    DeviceChangedPayload,
    DeviceSetupChangedPayload,
    GuiEvent,
)

from .device_progress import (
    DeviceSetupProgressFactory,
    DeviceSetupProgressModel,
    ProgressEntrySnapshot,
)

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
    def get_info(self) -> BaseDeviceInfo: ...


@runtime_checkable
class ValueDeviceProtocol(DeviceProtocol, Protocol):
    def get_value(self) -> object: ...
    def set_value(self, value: object) -> object: ...


@dataclass(frozen=True)
class DeviceSetupSnapshot:
    device_name: str
    progress: tuple[ProgressEntrySnapshot, ...]


class _DeviceSetupWorker(QThread):
    """Runs dev.setup(info) on a background thread with optional pbar factory and cancellation."""

    setup_finished: Signal = Signal(str)  # name
    failed: Signal = Signal(str, str)  # name, error_message
    cancelled: Signal = Signal(str)  # name

    def __init__(
        self,
        dev: DeviceProtocol,
        name: str,
        info: BaseDeviceInfo,
        pbar_factory: Optional[Callable[..., Any]],
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._dev = dev
        self._name = name
        self._info = info
        self._pbar_factory = pbar_factory
        self._stop_event = threading.Event()
        self._completed = False
        self._error: Optional[str] = None
        self.finished.connect(self._emit_outcome)

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

            self._completed = True
        except Exception as exc:
            self._error = str(exc)

    def _emit_outcome(self) -> None:
        if self._error is not None:
            self.failed.emit(self._name, self._error)
        elif not self._completed:
            raise RuntimeError("Device setup worker stopped without an outcome")
        elif self._stop_event.is_set():
            self.cancelled.emit(self._name)
        else:
            self.setup_finished.emit(self._name)


class DeviceService(QObject):
    """Device registration, setup, and value access for the GUI."""

    setup_finished: Signal = Signal(str)
    setup_failed: Signal = Signal(str, str)
    setup_cancelled: Signal = Signal(str)

    def __init__(
        self, state: "State", bus: "EventBus", parent: Optional[QObject] = None
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._bus = bus
        self._progress = DeviceSetupProgressModel(parent=self)
        self._progress.changed.connect(self._emit_setup_changed)
        self._active_worker: Optional[_DeviceSetupWorker] = None

    def _require_device_mutation_available(self, action: str) -> None:
        if self._state.is_run_active():
            raise RuntimeError(f"Cannot {action} while a run is active")
        if self._state.is_device_setup_active():
            raise RuntimeError(f"Cannot {action} while device setup is active")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_device(self, name: str, device: DeviceProtocol) -> None:
        logger.info("register_device: name=%r type=%s", name, type(device).__name__)
        self._require_device_mutation_available("register device")
        from zcu_tools.device import GlobalDeviceManager

        GlobalDeviceManager.register_device(name, device)
        self._bus.emit(GuiEvent.DEVICE_CHANGED, DeviceChangedPayload())

    def drop_device(self, name: str) -> None:
        logger.info("drop_device: name=%r", name)
        self._require_device_mutation_available("drop device")
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

    def get_device_info(self, name: str) -> BaseDeviceInfo | None:
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
        self._require_device_mutation_available("set device value")
        from zcu_tools.device import GlobalDeviceManager

        dev = cast(ValueDeviceProtocol, GlobalDeviceManager.get_device(name))
        return dev.set_value(value)

    # ------------------------------------------------------------------
    # Setup (async)
    # ------------------------------------------------------------------

    def setup_device(
        self,
        name: str,
        info: BaseDeviceInfo,
    ) -> None:
        from zcu_tools.device import GlobalDeviceManager

        self._require_device_mutation_available("setup device")
        dev = GlobalDeviceManager.get_device(name)
        self._state.begin_device_setup(name)
        worker = _DeviceSetupWorker(
            dev,
            name,
            info,
            DeviceSetupProgressFactory(self._progress),
            parent=self,
        )
        self._active_worker = worker
        worker.setup_finished.connect(self._on_setup_finished)
        worker.failed.connect(self._on_setup_failed)
        worker.cancelled.connect(self._on_setup_cancelled)
        worker.finished.connect(worker.deleteLater)
        self._emit_setup_changed()
        worker.start()

    def get_active_setup(self) -> Optional[DeviceSetupSnapshot]:
        name = self._state.active_device_setup_name
        if name is None:
            return None
        return DeviceSetupSnapshot(device_name=name, progress=self._progress.snapshot())

    def cancel_setup(self) -> None:
        if self._active_worker is None:
            raise RuntimeError("No device setup is active")
        self._active_worker.cancel()

    def _emit_setup_changed(self) -> None:
        self._bus.emit(
            GuiEvent.DEVICE_SETUP_CHANGED,
            DeviceSetupChangedPayload(active_setup=self.get_active_setup()),
        )

    def _clear_active_setup(self, name: str) -> None:
        self._state.end_device_setup(name)
        self._active_worker = None
        had_progress = bool(self._progress.snapshot())
        self._progress.clear()
        if not had_progress:
            self._emit_setup_changed()

    def _on_setup_finished(self, name: str) -> None:
        self._clear_active_setup(name)
        self.setup_finished.emit(name)

    def _on_setup_failed(self, name: str, error: str) -> None:
        self._clear_active_setup(name)
        self.setup_failed.emit(name, error)

    def _on_setup_cancelled(self, name: str) -> None:
        self._clear_active_setup(name)
        self.setup_cancelled.emit(name)
