from __future__ import annotations

import importlib
import logging
import threading
from dataclasses import dataclass, replace
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
from zcu_tools.gui.state import DeviceState, DeviceStatus

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
    OperationOutcome,
)

# DeviceMemoryInfo lives in the contract layer (ports) — the element type of
# RememberedDevicePort. Re-exported here for back-compat with imports of
# ``from .device import DeviceMemoryInfo``.
from .ports import DeviceMemoryInfo  # noqa: F401

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import EventBus
    from zcu_tools.gui.state import State

    from .ports import DriverFactoryPort

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
class DeviceSnapshot:
    """Read-time projection of a device for View / remote readers.

    Assembled by ``DeviceService._project`` from the State-owned ``DeviceState``
    (name/type/address/status/info/error) plus the live setup ``progress``
    (owned by ``DeviceSetupProgressModel``, spliced only for the active setup).
    It is never stored — State is the device-state SSOT.
    """

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
        state: "State",
        gate: OperationGate | None = None,
        parent: Optional[QObject] = None,
        driver_factory: Optional["DriverFactoryPort"] = None,
    ) -> None:
        super().__init__(parent)
        self._bus = bus
        self._state = state
        self._gate = gate or OperationGate()
        self._driver_factory = driver_factory or _default_driver_factory
        # Device state lives in State (the SSOT). This service holds only the
        # live driver (in GlobalDeviceManager), the worker threads, the setup
        # progress model and the in-flight operation transient below.
        self._progress = DeviceSetupProgressModel(parent=self)
        self.progress_model = self._progress  # public read-only alias for UI attachment
        self._active_lease: OperationLease | None = None
        self._active_name: str | None = None
        # Rollback buffer for the in-flight transition (worker-bound, not
        # serializable, so it stays here rather than in State).
        self._active_prior: DeviceState | None = None
        self._command_worker: _DeviceCommandWorker | None = None
        self._setup_worker: _DeviceSetupWorker | None = None

    def start_connect_device(self, req: ConnectDeviceRequest) -> None:
        current = self._state.get_device(req.name)
        if current is not None and current.is_live():
            raise RuntimeError(f"Device {req.name!r} is already connected or busy")
        initial = current or DeviceState(
            name=req.name,
            type_name=req.type_name,
            address=req.address,
            status=DeviceStatus.MEMORY_ONLY,
            remember=req.remember,
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
        dev = self._require_device(name)
        if not dev.is_memory_only():
            raise RuntimeError(f"Device {name!r} is not a memory-only device")
        self.start_connect_device(
            ConnectDeviceRequest(
                type_name=dev.type_name,
                name=dev.name,
                address=dev.address,
                remember=True,
            )
        )

    def start_disconnect_device(self, req: DisconnectDeviceRequest) -> None:
        current = self._require_connected_device(req.name)
        self._begin_operation(
            OperationKind.DEVICE_DISCONNECT,
            req.name,
            replace(current, status=DeviceStatus.DISCONNECTING, error=None),
        )
        worker = _DeviceCommandWorker(lambda: self._disconnect(req.name), parent=self)
        self._command_worker = worker
        worker.succeeded.connect(lambda _: self._on_disconnect_succeeded(req))
        worker.failed.connect(lambda error: self._on_operation_failed(req.name, error))
        worker.finished.connect(worker.deleteLater)
        self._start_command_worker(worker, req.name)

    def start_set_device_value(self, req: SetDeviceValueRequest) -> None:
        current = self._require_connected_device(req.name)
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

    def start_setup_device(self, req: SetupDeviceRequest) -> int:
        from zcu_tools.device import GlobalDeviceManager

        current = self._require_connected_device(req.name)
        driver = cast(DeviceProtocol, GlobalDeviceManager.get_device(req.name))
        self._begin_operation(
            OperationKind.DEVICE_SETUP,
            req.name,
            replace(current, status=DeviceStatus.SETTING_UP, error=None),
        )
        assert self._active_lease is not None  # set by _begin_operation
        token = self._active_lease.token
        worker = _DeviceSetupWorker(
            driver,
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
            worker.start()
            self._bus.emit(
                GuiEvent.DEVICE_SETUP_CHANGED,
                DeviceSetupChangedPayload(active_setup=self.get_active_setup()),
            )
        except Exception:
            self._abort_unstarted_operation(req.name)
            raise
        return token

    def cancel_device_operation(self, name: str) -> None:
        if self._active_name != name or self._setup_worker is None:
            raise RuntimeError(f"No cancellable device setup is active for {name!r}")
        self._setup_worker.cancel()

    def register_remembered_devices(self, entries: list[DeviceMemoryInfo]) -> None:
        for entry in entries:
            current = self._state.get_device(entry.name)
            if current is not None and current.is_live():
                logger.warning("Ignoring remembered live device %r", entry.name)
                continue
            self._state.put_device(
                DeviceState(
                    name=entry.name,
                    type_name=entry.type_name,
                    address=entry.address,
                    status=DeviceStatus.MEMORY_ONLY,
                    remember=True,
                )
            )

    def forget_device(self, name: str) -> None:
        dev = self._require_device(name)
        if not dev.is_memory_only():
            raise RuntimeError(f"Device {name!r} is not a memory-only device")
        self._state.remove_device(name)
        self._emit_device_changed(name)

    def _project(self, dev: DeviceState) -> DeviceSnapshot:
        """Assemble the read-time projection of a device-state entry.

        Splices live setup ``progress`` only for the device that is the active
        setup; every other field comes straight from State.
        """
        progress: tuple[ProgressEntrySnapshot, ...] = ()
        if dev.status is DeviceStatus.SETTING_UP and dev.name == self._active_name:
            progress = self._progress.snapshot()
        return DeviceSnapshot(
            name=dev.name,
            type_name=dev.type_name,
            address=dev.address,
            status=dev.status,
            info=dev.info,
            progress=progress,
            error=dev.error,
        )

    def list_device_snapshots(self) -> tuple[DeviceSnapshot, ...]:
        return tuple(self._project(dev) for dev in self._state.list_devices())

    def get_device_snapshot(self, name: str) -> DeviceSnapshot | None:
        dev = self._state.get_device(name)
        return None if dev is None else self._project(dev)

    def get_active_device_operation(self) -> DeviceSnapshot | None:
        if self._active_name is None:
            return None
        dev = self._state.get_device(self._active_name)
        return None if dev is None else self._project(dev)

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
        dev = self._state.get_device(name)
        return dev is not None and dev.is_memory_only()

    def get_memory_device_address(self, name: str) -> Optional[str]:
        dev = self._state.get_device(name)
        if dev is None or not dev.is_memory_only():
            return None
        return dev.address

    def get_device_unit(self, name: str) -> str:
        dev = self._state.get_device(name)
        if dev is None:
            return "none"
        if dev.type_name == "YOKOGS200" and dev.info is not None:
            return "V" if getattr(dev.info, "mode", None) == "voltage" else "A"
        return _DEVICE_DEFAULT_UNITS.get(dev.type_name, "none")

    def get_device_info(self, name: str) -> BaseDeviceInfo | None:
        from zcu_tools.device.manager import GlobalDeviceManager

        self._reject_mutating_read(name)
        dev = self._state.get_device(name)
        if dev is None or dev.is_memory_only():
            return None
        try:
            info = GlobalDeviceManager.get_info(name)
        except ValueError:
            return None
        # Read-time cache refresh: silent when unchanged (no bump), but if the
        # driver value moved underneath us this is a genuine state change — State
        # bumps device:<name> and we emit DEVICE_CHANGED so readers re-query.
        if self._state.refresh_device_info_cache(name, info):
            self._emit_device_changed(name)
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
        self, kind: OperationKind, name: str, pending: DeviceState
    ) -> None:
        lease = self._gate.acquire(kind, owner_id=name, resource_id=name)
        prior = self._state.get_device(name)
        self._active_lease = lease
        self._active_name = name
        self._active_prior = prior
        self._state.put_device(pending)
        try:
            self._emit_device_changed(name)
        except Exception:
            if prior is None:
                self._state.remove_device(name)
            else:
                self._state.put_device(prior)
            self._active_name = None
            self._active_lease = None
            self._active_prior = None
            self._gate.release(
                lease, OperationOutcome("failed", "operation failed to begin")
            )
            raise

    def _finish_operation(self, name: str, outcome: OperationOutcome) -> None:
        if self._active_name != name or self._active_lease is None:
            raise RuntimeError(f"Device operation for {name!r} has no active lease")
        lease = self._active_lease
        self._active_name = None
        self._active_lease = None
        self._active_prior = None
        self._command_worker = None
        self._setup_worker = None
        self._gate.release(lease, outcome)

    def _start_command_worker(self, worker: _DeviceCommandWorker, name: str) -> None:
        try:
            worker.start()
        except Exception:
            self._abort_unstarted_operation(name)
            raise

    def _abort_unstarted_operation(self, name: str) -> None:
        prior = self._active_prior
        if prior is None:
            if self._state.has_device(name):
                self._state.remove_device(name)
        else:
            self._state.put_device(prior)
        self._finish_operation(
            name, OperationOutcome("failed", "operation failed to start")
        )
        self._emit_device_changed(name)

    def _on_connect_succeeded(self, req: ConnectDeviceRequest, info: object) -> None:
        self._state.put_device(
            DeviceState(
                name=req.name,
                type_name=req.type_name,
                address=req.address,
                status=DeviceStatus.CONNECTED,
                remember=req.remember,
                info=cast(BaseDeviceInfo, info),
            )
        )
        self._finish_operation(req.name, OperationOutcome("finished"))
        self._emit_device_changed(req.name)
        self.device_connected.emit(req)

    def _on_disconnect_succeeded(self, req: DisconnectDeviceRequest) -> None:
        current = self._require_device(req.name)
        if req.remember:
            self._state.put_device(
                replace(
                    current,
                    status=DeviceStatus.MEMORY_ONLY,
                    info=None,
                    error=None,
                    remember=True,
                )
            )
        else:
            self._state.remove_device(req.name)
        self._finish_operation(req.name, OperationOutcome("finished"))
        self._emit_device_changed(req.name)
        self.device_disconnected.emit(req)

    def _on_value_set_succeeded(self, name: str, info: object) -> None:
        current = self._require_device(name)
        self._state.put_device(
            replace(
                current,
                status=DeviceStatus.CONNECTED,
                info=cast(BaseDeviceInfo, info),
                error=None,
            )
        )
        self._finish_operation(name, OperationOutcome("finished"))
        self._emit_device_changed(name)
        self.value_set.emit(name)

    def _on_setup_finished(self, name: str, info: object) -> None:
        current = self._require_device(name)
        self._state.put_device(
            replace(
                current,
                status=DeviceStatus.CONNECTED,
                info=cast(BaseDeviceInfo, info),
                error=None,
            )
        )
        # release() settles the operation handle (sets the token Event and stores
        # the outcome) — may wake an off-main operation.await waiter.
        self._finish_operation(name, OperationOutcome("finished"))
        self._clear_progress()
        self._emit_device_changed(name)
        self._bus.emit(
            GuiEvent.DEVICE_SETUP_CHANGED,
            DeviceSetupChangedPayload(active_setup=None),
        )
        self.setup_finished.emit(name)

    def _on_setup_failed(self, name: str, error: str) -> None:
        self._restore_prior_device(name, error)
        self._finish_operation(name, OperationOutcome("failed", error))
        self._clear_progress()
        self._emit_device_changed(name)
        self._bus.emit(
            GuiEvent.DEVICE_SETUP_CHANGED,
            DeviceSetupChangedPayload(active_setup=None),
        )
        self.setup_failed.emit(name, error)

    def _on_setup_cancelled(self, name: str) -> None:
        self._restore_prior_device(name, None)
        self._finish_operation(
            name, OperationOutcome("cancelled", f"device {name!r} setup was cancelled")
        )
        self._clear_progress()
        self._emit_device_changed(name)
        self._bus.emit(
            GuiEvent.DEVICE_SETUP_CHANGED,
            DeviceSetupChangedPayload(active_setup=None),
        )
        self.setup_cancelled.emit(name)

    def _on_operation_failed(self, name: str, error: object) -> None:
        message = str(error)
        lease = self._active_lease
        if (
            lease is not None
            and lease.kind is OperationKind.DEVICE_CONNECT
            and self._active_prior is None
        ):
            self._state.remove_device(name)
        else:
            self._restore_prior_device(name, message)
        self._finish_operation(name, OperationOutcome("failed", message))
        self._emit_device_changed(name)
        self.operation_failed.emit(name, message)

    def _restore_prior_device(self, name: str, error: str | None) -> None:
        prior = self._active_prior
        if prior is None:
            pending = self._require_device(name)
            prior = replace(
                pending,
                status=DeviceStatus.MEMORY_ONLY,
                info=None,
            )
        self._state.put_device(replace(prior, error=error))

    def _emit_device_changed(self, name: str) -> None:
        # Pure signal: every caller has already written device state through a
        # State mutator (which bumps device:<name> on the main thread). This is
        # the notification half only — no state write, no version bump here.
        self._bus.emit(GuiEvent.DEVICE_CHANGED, DeviceChangedPayload(name=name))

    def _clear_progress(self) -> None:
        self._progress.clear()

    def _reject_mutating_read(self, name: str) -> None:
        if self._gate.is_device_mutating(name):
            raise OperationConflictError(
                f"Cannot read device {name!r} while it is being mutated"
            )

    def _require_device(self, name: str) -> DeviceState:
        dev = self._state.get_device(name)
        if dev is None:
            raise RuntimeError(f"Device {name!r} is not known")
        return dev

    def _require_connected_device(self, name: str) -> DeviceState:
        dev = self._require_device(name)
        if not dev.is_connected():
            raise RuntimeError(f"Device {name!r} is not connected")
        return dev
