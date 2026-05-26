from __future__ import annotations

import importlib
import logging
import threading
from dataclasses import dataclass, replace
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Protocol,
    cast,
    runtime_checkable,
)

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
from .operation_gate import (
    OperationConflictError,
    OperationGate,
    OperationKind,
    OperationLease,
)

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import EventBus

logger = logging.getLogger(__name__)


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
    def close(self) -> None: ...


@runtime_checkable
class ValueDeviceProtocol(DeviceProtocol, Protocol):
    def get_value(self) -> object: ...
    def set_value(self, value: object) -> object: ...


@dataclass(frozen=True)
class ConnectDeviceRequest:
    type_name: str
    name: str
    address: str
    remember: bool = True


@dataclass(frozen=True)
class DisconnectDeviceRequest:
    name: str
    remember: bool = True


@dataclass(frozen=True)
class SetupDeviceRequest:
    name: str
    info: BaseDeviceInfo


@dataclass(frozen=True)
class SetDeviceValueRequest:
    name: str
    value: object


@dataclass(frozen=True)
class DeviceMemoryInfo:
    type_name: str
    name: str
    address: str


class DeviceStatus(Enum):
    MEMORY_ONLY = "memory_only"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    SETTING_UP = "setting_up"
    SETTING_VALUE = "setting_value"


@dataclass(frozen=True)
class DeviceSnapshot:
    name: str
    type_name: str
    address: str
    status: DeviceStatus
    info: BaseDeviceInfo | None = None
    progress: tuple[ProgressEntrySnapshot, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class DeviceEntry:
    name: str
    type_name: str
    is_connected: bool


@dataclass(frozen=True)
class DeviceSetupSnapshot:
    device_name: str
    progress: tuple[ProgressEntrySnapshot, ...]


class DeviceRegistrationError(RuntimeError):
    """Expected driver construction or registration failure."""


_DEVICE_TYPE_REGISTRY: dict[str, tuple[str, bool]] = {
    "FakeDevice": ("zcu_tools.device.fake.FakeDevice", False),
    "YOKOGS200": ("zcu_tools.device.yoko.YOKOGS200", True),
    "RohdeSchwarzSGS100A": ("zcu_tools.device.sgs100a.RohdeSchwarzSGS100A", True),
}

_DEVICE_DEFAULT_UNITS: dict[str, str] = {
    "FakeDevice": "none",
    "YOKOGS200": "A",
}


def list_supported_device_types() -> list[str]:
    return list(_DEVICE_TYPE_REGISTRY.keys())


def _default_driver_factory(type_name: str, address: str) -> DeviceProtocol:
    if type_name not in _DEVICE_TYPE_REGISTRY:
        raise DeviceRegistrationError(f"Unknown device type: {type_name!r}")
    class_path, requires_address = _DEVICE_TYPE_REGISTRY[type_name]
    module_path, cls_name = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)
    if requires_address:
        import pyvisa  # type: ignore[import-untyped]

        return cast(DeviceProtocol, cls(address, pyvisa.ResourceManager()))
    return cast(DeviceProtocol, cls())


class _DeviceCommandWorker(QThread):
    succeeded: Signal = Signal(object)
    failed: Signal = Signal(object)

    def __init__(
        self, command: Callable[[], object], parent: Optional[QObject] = None
    ) -> None:
        super().__init__(parent)
        self._command = command
        self._result: object | None = None
        self._error: Exception | None = None
        self.finished.connect(self._emit_outcome)

    def run(self) -> None:
        try:
            self._result = self._command()
        except Exception as exc:
            self._error = exc

    def _emit_outcome(self) -> None:
        if self._error is not None:
            self.failed.emit(self._error)
            return
        self.succeeded.emit(self._result)


class _DeviceSetupWorker(QThread):
    setup_finished: Signal = Signal(str, object)
    failed: Signal = Signal(str, str)
    cancelled: Signal = Signal(str)

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
        self._result: BaseDeviceInfo | None = None
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
            self._result = self._dev.get_info()
            self._completed = True
        except Exception as exc:
            self._error = str(exc)

    def _emit_outcome(self) -> None:
        if self._error is not None:
            self.failed.emit(self._name, self._error)
        elif not self._completed or self._result is None:
            raise RuntimeError("Device setup worker stopped without an outcome")
        elif self._stop_event.is_set():
            self.cancelled.emit(self._name)
        else:
            self.setup_finished.emit(self._name, self._result)


class DeviceService(QObject):
    """Own device mutation workers and cached render snapshots."""

    device_connected: Signal = Signal(object)
    device_disconnected: Signal = Signal(object)
    value_set: Signal = Signal(str)
    operation_failed: Signal = Signal(str, str)
    setup_finished: Signal = Signal(str)
    setup_failed: Signal = Signal(str, str)
    setup_cancelled: Signal = Signal(str)

    def __init__(
        self,
        bus: "EventBus",
        gate: OperationGate | None = None,
        parent: Optional[QObject] = None,
        driver_factory: Optional[Callable[[str, str], DeviceProtocol]] = None,
    ) -> None:
        super().__init__(parent)
        self._bus = bus
        self._gate = gate or OperationGate()
        self._driver_factory = driver_factory or _default_driver_factory
        self._snapshots: dict[str, DeviceSnapshot] = {}
        self._progress = DeviceSetupProgressModel(parent=self)
        self._progress.changed.connect(self._emit_setup_changed)
        self._active_lease: OperationLease | None = None
        self._active_name: str | None = None
        self._active_prior: DeviceSnapshot | None = None
        self._command_worker: _DeviceCommandWorker | None = None
        self._setup_worker: _DeviceSetupWorker | None = None

    def start_connect_device(self, req: ConnectDeviceRequest) -> None:
        current = self._snapshots.get(req.name)
        if current is not None and current.status is not DeviceStatus.MEMORY_ONLY:
            raise RuntimeError(f"Device {req.name!r} is already connected or busy")
        initial = current or DeviceSnapshot(
            name=req.name,
            type_name=req.type_name,
            address=req.address,
            status=DeviceStatus.MEMORY_ONLY,
        )
        self._begin_operation(
            OperationKind.DEVICE_CONNECT,
            req.name,
            replace(initial, status=DeviceStatus.CONNECTING, error=None),
        )
        worker = _DeviceCommandWorker(lambda: self._connect(req), parent=self)
        self._command_worker = worker
        worker.succeeded.connect(lambda info: self._on_connect_succeeded(req, info))
        worker.failed.connect(lambda error: self._on_operation_failed(req.name, error))
        worker.finished.connect(worker.deleteLater)
        self._start_command_worker(worker, req.name)

    def start_reconnect_device(self, name: str) -> None:
        snapshot = self._require_snapshot(name)
        if snapshot.status is not DeviceStatus.MEMORY_ONLY:
            raise RuntimeError(f"Device {name!r} is not a memory-only device")
        self.start_connect_device(
            ConnectDeviceRequest(
                type_name=snapshot.type_name,
                name=snapshot.name,
                address=snapshot.address,
                remember=True,
            )
        )

    def start_disconnect_device(self, req: DisconnectDeviceRequest) -> None:
        current = self._require_connected_snapshot(req.name)
        self._begin_operation(
            OperationKind.DEVICE_DISCONNECT,
            req.name,
            replace(current, status=DeviceStatus.DISCONNECTING, error=None),
        )
        worker = _DeviceCommandWorker(lambda: self._disconnect(req.name), parent=self)
        self._command_worker = worker
        worker.succeeded.connect(lambda _unused: self._on_disconnect_succeeded(req))
        worker.failed.connect(lambda error: self._on_operation_failed(req.name, error))
        worker.finished.connect(worker.deleteLater)
        self._start_command_worker(worker, req.name)

    def start_set_device_value(self, req: SetDeviceValueRequest) -> None:
        current = self._require_connected_snapshot(req.name)
        self._begin_operation(
            OperationKind.DEVICE_SET_VALUE,
            req.name,
            replace(current, status=DeviceStatus.SETTING_VALUE, error=None),
        )
        worker = _DeviceCommandWorker(lambda: self._set_value(req), parent=self)
        self._command_worker = worker
        worker.succeeded.connect(
            lambda info: self._on_value_set_succeeded(req.name, info)
        )
        worker.failed.connect(lambda error: self._on_operation_failed(req.name, error))
        worker.finished.connect(worker.deleteLater)
        self._start_command_worker(worker, req.name)

    def start_setup_device(self, req: SetupDeviceRequest) -> None:
        from zcu_tools.device import GlobalDeviceManager

        current = self._require_connected_snapshot(req.name)
        dev = cast(DeviceProtocol, GlobalDeviceManager.get_device(req.name))
        self._begin_operation(
            OperationKind.DEVICE_SETUP,
            req.name,
            replace(current, status=DeviceStatus.SETTING_UP, error=None, progress=()),
        )
        worker = _DeviceSetupWorker(
            dev,
            req.name,
            req.info,
            DeviceSetupProgressFactory(self._progress),
            parent=self,
        )
        self._setup_worker = worker
        worker.setup_finished.connect(self._on_setup_finished)
        worker.failed.connect(self._on_setup_failed)
        worker.cancelled.connect(self._on_setup_cancelled)
        worker.finished.connect(worker.deleteLater)
        try:
            self._emit_setup_changed()
            worker.start()
        except Exception:
            self._abort_unstarted_operation(req.name)
            raise

    def cancel_device_operation(self, name: str) -> None:
        if self._active_name != name or self._setup_worker is None:
            raise RuntimeError(f"No cancellable device setup is active for {name!r}")
        self._setup_worker.cancel()

    def register_remembered_devices(self, entries: list[DeviceMemoryInfo]) -> None:
        for entry in entries:
            current = self._snapshots.get(entry.name)
            if current is not None and current.status is not DeviceStatus.MEMORY_ONLY:
                logger.warning("Ignoring remembered live device %r", entry.name)
                continue
            self._snapshots[entry.name] = DeviceSnapshot(
                name=entry.name,
                type_name=entry.type_name,
                address=entry.address,
                status=DeviceStatus.MEMORY_ONLY,
            )

    def forget_device(self, name: str) -> None:
        snapshot = self._require_snapshot(name)
        if snapshot.status is not DeviceStatus.MEMORY_ONLY:
            raise RuntimeError(f"Device {name!r} is not a memory-only device")
        del self._snapshots[name]
        self._emit_device_changed(name)

    def list_device_snapshots(self) -> tuple[DeviceSnapshot, ...]:
        return tuple(self._snapshots[name] for name in sorted(self._snapshots))

    def get_device_snapshot(self, name: str) -> DeviceSnapshot | None:
        return self._snapshots.get(name)

    def get_active_device_operation(self) -> DeviceSnapshot | None:
        if self._active_name is None:
            return None
        return self._snapshots.get(self._active_name)

    def get_active_setup(self) -> Optional[DeviceSetupSnapshot]:
        snapshot = self.get_active_device_operation()
        if snapshot is None or snapshot.status is not DeviceStatus.SETTING_UP:
            return None
        return DeviceSetupSnapshot(
            device_name=snapshot.name,
            progress=snapshot.progress,
        )

    def list_devices(self) -> list[DeviceEntry]:
        return [
            DeviceEntry(
                name=snapshot.name,
                type_name=snapshot.type_name,
                is_connected=snapshot.status
                in {
                    DeviceStatus.CONNECTED,
                    DeviceStatus.DISCONNECTING,
                    DeviceStatus.SETTING_UP,
                    DeviceStatus.SETTING_VALUE,
                },
            )
            for snapshot in self.list_device_snapshots()
        ]

    def list_device_names(self) -> list[str]:
        return [
            snapshot.name
            for snapshot in self.list_device_snapshots()
            if snapshot.status is DeviceStatus.CONNECTED
        ]

    def is_memory_device(self, name: str) -> bool:
        snapshot = self._snapshots.get(name)
        return snapshot is not None and snapshot.status is DeviceStatus.MEMORY_ONLY

    def get_memory_device_address(self, name: str) -> Optional[str]:
        snapshot = self._snapshots.get(name)
        if snapshot is None or snapshot.status is not DeviceStatus.MEMORY_ONLY:
            return None
        return snapshot.address

    def get_device_unit(self, name: str) -> str:
        snapshot = self._snapshots.get(name)
        if snapshot is None:
            return "none"
        if snapshot.type_name == "YOKOGS200" and snapshot.info is not None:
            return "V" if getattr(snapshot.info, "mode", None) == "voltage" else "A"
        return _DEVICE_DEFAULT_UNITS.get(snapshot.type_name, "none")

    def get_device_info(self, name: str) -> BaseDeviceInfo | None:
        from zcu_tools.device.manager import GlobalDeviceManager

        self._reject_mutating_read(name)
        snapshot = self._snapshots.get(name)
        if snapshot is None or snapshot.status is DeviceStatus.MEMORY_ONLY:
            return None
        try:
            info = GlobalDeviceManager.get_info(name)
        except ValueError:
            return None
        self._snapshots[name] = replace(snapshot, info=info, error=None)
        return info

    def get_device_value(self, name: str) -> object:
        info = self.get_device_info(name)
        return None if info is None else getattr(info, "value", None)

    def get_device_value_for_new_context(self, name: str) -> Optional[float]:
        info = self.get_device_info(name)
        if info is None:
            return None
        raw = getattr(info, "value", None)
        return None if raw is None else float(raw)

    def _connect(self, req: ConnectDeviceRequest) -> BaseDeviceInfo:
        from zcu_tools.device import GlobalDeviceManager

        if req.name in GlobalDeviceManager.get_all_devices():
            raise DeviceRegistrationError(f"Device {req.name!r} is already registered")
        device: DeviceProtocol | None = None
        registered = False
        try:
            device = self._driver_factory(req.type_name, req.address)
            GlobalDeviceManager.register_device(req.name, device)
            registered = True
            return device.get_info()
        except DeviceRegistrationError:
            self._cleanup_failed_connection(req.name, device, registered)
            raise
        except Exception as exc:
            self._cleanup_failed_connection(req.name, device, registered)
            raise DeviceRegistrationError(
                f"Failed to connect {req.type_name} {req.name!r}: {exc}"
            ) from exc

    @staticmethod
    def _cleanup_failed_connection(
        name: str, device: DeviceProtocol | None, registered: bool
    ) -> None:
        from zcu_tools.device import GlobalDeviceManager

        if registered:
            GlobalDeviceManager.drop_device(name, ignore_error=True)
        if device is not None:
            try:
                device.close()
            except Exception:
                logger.exception(
                    "Failed to close device %r after connect failure", name
                )

    @staticmethod
    def _disconnect(name: str) -> object:
        from zcu_tools.device.manager import GlobalDeviceManager

        device = cast(DeviceProtocol, GlobalDeviceManager.get_device(name))
        device.close()
        GlobalDeviceManager.drop_device(name)
        return None

    @staticmethod
    def _set_value(req: SetDeviceValueRequest) -> BaseDeviceInfo:
        from zcu_tools.device import GlobalDeviceManager

        device = GlobalDeviceManager.get_device(req.name)
        if not isinstance(device, ValueDeviceProtocol):
            raise RuntimeError(
                f"Device {req.name!r} ({type(device).__name__}) does not support set_value."
            )
        device.set_value(req.value)
        return device.get_info()

    def _begin_operation(
        self, kind: OperationKind, name: str, pending: DeviceSnapshot
    ) -> None:
        lease = self._gate.acquire(kind, owner_id=name, resource_id=name)
        prior = self._snapshots.get(name)
        self._active_lease = lease
        self._active_name = name
        self._active_prior = prior
        self._snapshots[name] = pending
        try:
            self._emit_device_changed(name)
        except Exception:
            if prior is None:
                del self._snapshots[name]
            else:
                self._snapshots[name] = prior
            self._active_name = None
            self._active_lease = None
            self._active_prior = None
            self._gate.release(lease)
            raise

    def _finish_operation(self, name: str) -> None:
        if self._active_name != name or self._active_lease is None:
            raise RuntimeError(f"Device operation for {name!r} has no active lease")
        lease = self._active_lease
        self._active_name = None
        self._active_lease = None
        self._active_prior = None
        self._command_worker = None
        self._setup_worker = None
        self._gate.release(lease)

    def _start_command_worker(self, worker: _DeviceCommandWorker, name: str) -> None:
        try:
            worker.start()
        except Exception:
            self._abort_unstarted_operation(name)
            raise

    def _abort_unstarted_operation(self, name: str) -> None:
        prior = self._active_prior
        if prior is None:
            self._snapshots.pop(name, None)
        else:
            self._snapshots[name] = prior
        self._finish_operation(name)
        self._emit_device_changed(name)

    def _on_connect_succeeded(self, req: ConnectDeviceRequest, info: object) -> None:
        self._snapshots[req.name] = DeviceSnapshot(
            name=req.name,
            type_name=req.type_name,
            address=req.address,
            status=DeviceStatus.CONNECTED,
            info=cast(BaseDeviceInfo, info),
        )
        self._finish_operation(req.name)
        self._emit_device_changed(req.name)
        self.device_connected.emit(req)

    def _on_disconnect_succeeded(self, req: DisconnectDeviceRequest) -> None:
        current = self._require_snapshot(req.name)
        if req.remember:
            self._snapshots[req.name] = replace(
                current,
                status=DeviceStatus.MEMORY_ONLY,
                info=None,
                progress=(),
                error=None,
            )
        else:
            del self._snapshots[req.name]
        self._finish_operation(req.name)
        self._emit_device_changed(req.name)
        self.device_disconnected.emit(req)

    def _on_value_set_succeeded(self, name: str, info: object) -> None:
        current = self._require_snapshot(name)
        self._snapshots[name] = replace(
            current,
            status=DeviceStatus.CONNECTED,
            info=cast(BaseDeviceInfo, info),
            error=None,
        )
        self._finish_operation(name)
        self._emit_device_changed(name)
        self.value_set.emit(name)

    def _on_setup_finished(self, name: str, info: object) -> None:
        current = self._require_snapshot(name)
        self._snapshots[name] = replace(
            current,
            status=DeviceStatus.CONNECTED,
            info=cast(BaseDeviceInfo, info),
            progress=(),
            error=None,
        )
        self._finish_operation(name)
        self._clear_progress()
        self._emit_device_changed(name)
        self._emit_setup_changed()
        self.setup_finished.emit(name)

    def _on_setup_failed(self, name: str, error: str) -> None:
        self._restore_prior_snapshot(name, error)
        self._finish_operation(name)
        self._clear_progress()
        self._emit_device_changed(name)
        self._emit_setup_changed()
        self.setup_failed.emit(name, error)

    def _on_setup_cancelled(self, name: str) -> None:
        self._restore_prior_snapshot(name, None)
        self._finish_operation(name)
        self._clear_progress()
        self._emit_device_changed(name)
        self._emit_setup_changed()
        self.setup_cancelled.emit(name)

    def _on_operation_failed(self, name: str, error: object) -> None:
        message = str(error)
        lease = self._active_lease
        if (
            lease is not None
            and lease.kind is OperationKind.DEVICE_CONNECT
            and self._active_prior is None
        ):
            del self._snapshots[name]
        else:
            self._restore_prior_snapshot(name, message)
        self._finish_operation(name)
        self._emit_device_changed(name)
        self.operation_failed.emit(name, message)

    def _restore_prior_snapshot(self, name: str, error: str | None) -> None:
        prior = self._active_prior
        if prior is None:
            pending = self._require_snapshot(name)
            prior = replace(
                pending,
                status=DeviceStatus.MEMORY_ONLY,
                info=None,
                progress=(),
            )
        self._snapshots[name] = replace(prior, error=error, progress=())

    def _emit_device_changed(self, name: str) -> None:
        self._bus.emit(GuiEvent.DEVICE_CHANGED, DeviceChangedPayload(name=name))

    def _emit_setup_changed(self) -> None:
        if self._active_name is not None:
            snapshot = self._snapshots.get(self._active_name)
            if snapshot is not None and snapshot.status is DeviceStatus.SETTING_UP:
                self._snapshots[self._active_name] = replace(
                    snapshot, progress=self._progress.snapshot()
                )
                self._emit_device_changed(self._active_name)
        self._bus.emit(
            GuiEvent.DEVICE_SETUP_CHANGED,
            DeviceSetupChangedPayload(active_setup=self.get_active_setup()),
        )

    def _clear_progress(self) -> None:
        had_progress = bool(self._progress.snapshot())
        self._progress.clear()
        if not had_progress:
            self._emit_setup_changed()

    def _reject_mutating_read(self, name: str) -> None:
        if self._gate.is_device_mutating(name):
            raise OperationConflictError(
                f"Cannot read device {name!r} while it is being mutated"
            )

    def _require_snapshot(self, name: str) -> DeviceSnapshot:
        snapshot = self._snapshots.get(name)
        if snapshot is None:
            raise RuntimeError(f"Device {name!r} is not known")
        return snapshot

    def _require_connected_snapshot(self, name: str) -> DeviceSnapshot:
        snapshot = self._require_snapshot(name)
        if snapshot.status is not DeviceStatus.CONNECTED:
            raise RuntimeError(f"Device {name!r} is not connected")
        return snapshot
