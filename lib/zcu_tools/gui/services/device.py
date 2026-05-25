from __future__ import annotations

import importlib
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


@dataclass(frozen=True)
class RegisterDeviceRequest:
    """Immutable request to construct + register a device by type name and address."""

    type_name: str
    name: str
    address: str


class DeviceRegistrationError(RuntimeError):
    """Expected failure raised by DeviceService.register_device for user-facing errors.

    Covers driver constructor failures (connection refused, timeout, OS errors,
    pyvisa errors, etc.). Distinct from contract violations which raise plain
    RuntimeError via the operation guard.
    """


# (class_path, requires_address) per device type. Service-owned; not for view use.
_DEVICE_TYPE_REGISTRY: dict[str, tuple[str, bool]] = {
    "FakeDevice": ("zcu_tools.device.fake.FakeDevice", False),
    "YOKOGS200": ("zcu_tools.device.yoko.YOKOGS200", True),
    "RohdeSchwarzSGS100A": ("zcu_tools.device.sgs100a.RohdeSchwarzSGS100A", True),
}

# Default unit per device type. YOKOGS200 overrides via runtime mode query.
_DEVICE_DEFAULT_UNITS: dict[str, str] = {
    "FakeDevice": "none",
    "YOKOGS200": "A",
}


def list_supported_device_types() -> list[str]:
    """Return the supported device type names in insertion order."""
    return list(_DEVICE_TYPE_REGISTRY.keys())


def _default_driver_factory(type_name: str, address: str) -> DeviceProtocol:
    """Resolve and instantiate a device driver by registered type name.

    Acquires pyvisa ResourceManager when needed. Tests may monkeypatch this
    symbol on the module to inject fakes without going through pyvisa.
    """
    if type_name not in _DEVICE_TYPE_REGISTRY:
        raise DeviceRegistrationError(f"Unknown device type: {type_name!r}")
    class_path, requires_address = _DEVICE_TYPE_REGISTRY[type_name]
    module_path, cls_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    if requires_address:
        import pyvisa  # type: ignore[import-untyped]

        rm = pyvisa.ResourceManager()
        return cast(DeviceProtocol, cls(address, rm))
    return cast(DeviceProtocol, cls())


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
        self,
        state: "State",
        bus: "EventBus",
        parent: Optional[QObject] = None,
        driver_factory: Optional[Callable[[str, str], DeviceProtocol]] = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._bus = bus
        self._progress = DeviceSetupProgressModel(parent=self)
        self._progress.changed.connect(self._emit_setup_changed)
        self._active_worker: Optional[_DeviceSetupWorker] = None
        # Driver factory is injected here so tests can substitute a fake without
        # going through pyvisa or importlib. Production code calls register_device
        # with a RegisterDeviceRequest; the factory turns (type_name, address)
        # into a DeviceProtocol instance.
        self._driver_factory: Callable[[str, str], DeviceProtocol] = (
            driver_factory or _default_driver_factory
        )

    def _require_device_mutation_available(self, action: str) -> None:
        if self._state.is_run_active():
            raise RuntimeError(f"Cannot {action} while a run is active")
        if self._state.is_device_setup_active():
            raise RuntimeError(f"Cannot {action} while device setup is active")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_device(self, req: RegisterDeviceRequest) -> None:
        """Construct + register a device atomically under the mutation guard.

        Raises DeviceRegistrationError for user-facing constructor failures
        (network/IO/pyvisa); raises plain RuntimeError when the mutation guard
        rejects the operation.
        """
        logger.info(
            "register_device: name=%r type=%r addr=%r",
            req.name,
            req.type_name,
            req.address,
        )
        self._require_device_mutation_available("register device")
        from zcu_tools.device import GlobalDeviceManager

        try:
            device = self._driver_factory(req.type_name, req.address)
        except DeviceRegistrationError:
            raise
        except (ConnectionRefusedError, TimeoutError, OSError) as exc:
            raise DeviceRegistrationError(
                f"Failed to connect to {req.type_name} at {req.address!r}: {exc}"
            ) from exc
        except Exception as exc:
            raise DeviceRegistrationError(
                f"Failed to construct {req.type_name}: {exc}"
            ) from exc

        try:
            GlobalDeviceManager.register_device(req.name, device)
        except Exception:
            # Best effort: release resources held by the half-constructed driver.
            close = getattr(device, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.exception(
                        "Failed to close device %r after register failure", req.name
                    )
            raise
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

    def list_device_names(self) -> list[str]:
        """Return sorted registered device names. Single read boundary for views/models."""
        from zcu_tools.device import GlobalDeviceManager

        return sorted(GlobalDeviceManager.get_all_devices().keys())

    def get_device_unit(self, name: str) -> str:
        """Return the unit string for a registered device.

        Encapsulates the YOKOGS200 mode-dependent voltage/current branch so
        Views never read the global manager or driver state directly.
        """
        from zcu_tools.device import GlobalDeviceManager

        try:
            dev = GlobalDeviceManager.get_device(name)
        except ValueError:
            return "none"
        dev_type = type(dev).__name__
        if dev_type == "YOKOGS200":
            from zcu_tools.device.yoko import YOKOGS200

            if isinstance(dev, YOKOGS200):
                mode = dev.get_mode()
                return "V" if mode == "voltage" else "A"
        return _DEVICE_DEFAULT_UNITS.get(dev_type, "none")

    def get_device_value_for_new_context(self, name: str) -> Optional[float]:
        """Read the device's current value as a float, for new-context creation.

        Returns None when the device is not registered. Any read failure raises
        as a contract violation (RuntimeError) because the caller will already
        have validated the device exists via list_device_names() and the GUI
        guard.
        """
        from zcu_tools.device import GlobalDeviceManager

        try:
            dev = cast(ValueDeviceProtocol, GlobalDeviceManager.get_device(name))
        except ValueError:
            return None
        return float(dev.get_value())  # type: ignore[arg-type]

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
