from __future__ import annotations

import importlib
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    cast,
    runtime_checkable,
)

from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from zcu_tools.device.base import BaseDeviceInfo
from zcu_tools.gui.session.events import (
    DeviceChangedPayload,
    DeviceSetupFinishedPayload,
    DeviceSetupStartedPayload,
)
from zcu_tools.gui.session.operation_handles import OperationHandles, OperationOutcome
from zcu_tools.gui.session.ports import (
    BackgroundExecutor,
    DeviceMemoryInfo,
    DriverFactoryPort,
    ExclusionGate,
    OffMainScopes,
    OperationConflictError,
    OperationKind,
    ProgressHub,
)
from zcu_tools.gui.session.state import DeviceState, DeviceStatus

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.state import SessionState

logger = logging.getLogger(__name__)


@runtime_checkable
class DeviceProtocol(Protocol):
    """Structural contract a hardware driver must satisfy to be a GUI device.

    pyright only checks the *shape* (these methods exist with these signatures).
    The semantic invariants an implementer must also honour — and which a bug
    silently violates — are:

    - ``setup`` runs **off the main thread** (via ``BackgroundService``), never
      the Qt main thread. It MUST poll ``stop_event`` during any long operation:
      cancellation works by ``stop_event.set()``, and the worker reports
      "cancelled" only if it returns with the event set. A ``setup`` that ignores
      ``stop_event`` makes cancel a no-op (the worker blocks until natural
      completion). ``progress`` may drive a pbar via the ambient pbar factory.
    - ``get_info`` returns a fresh value snapshot. It is called both on the
      worker (right after ``setup``) and on the main thread (idle live-read,
      guarded by OperationGate against a concurrent mutation). It must not mutate
      device state.
    - ``close`` releases the underlying resource and SHOULD be idempotent: it is
      called on disconnect, and on any rollback where a driver was constructed
      but never became the live registry entry.
    """

    def setup(
        self,
        cfg: Any,
        *,
        progress: bool = True,
        stop_event: threading.Event | None = None,
    ) -> None: ...
    def get_info(self) -> BaseDeviceInfo: ...
    def close(self) -> None: ...


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
class DeviceSnapshot:
    """Read-time projection of a device for View / remote readers.

    Assembled by ``DeviceService._project`` from the State-owned ``DeviceState``
    (name/type/address/status/info/error). Live setup progress is not spliced
    here — it is polled separately via ``operation.progress`` (by operation_id,
    ProgressService). It is never stored — State is the device-state SSOT.
    """

    name: str
    type_name: str
    address: str
    status: DeviceStatus
    info: BaseDeviceInfo | None = None
    error: str | None = None


@dataclass(frozen=True)
class DeviceEntry:
    name: str
    type_name: str
    is_connected: bool


@dataclass(frozen=True)
class DeviceSetupSnapshot:
    # Live progress is polled via operation.progress (by operation_id, ADR-0013
    # device↔run alignment), not carried here — this names *which* device is
    # setting up.
    device_name: str


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


def _mode_dependent_unit(dev: DeviceState) -> str | None:
    """Return "V" or "A" for a YOKOGS200 whose mode is cached in ``dev.info``.

    Returns ``None`` for any other device type or when ``dev.info`` is ``None``
    (device not yet connected / info not cached), signalling the caller to fall
    back to :data:`_DEVICE_DEFAULT_UNITS`.
    """
    if dev.type_name == "YOKOGS200" and dev.info is not None:
        return "V" if getattr(dev.info, "mode", None) == "voltage" else "A"
    return None


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


class DeviceService(QObject):
    """Own device mutation workers and cached render snapshots."""

    device_connected: Signal = Signal(object)
    device_disconnected: Signal = Signal(object)
    operation_failed: Signal = Signal(str, str)
    setup_finished: Signal = Signal(str)
    setup_failed: Signal = Signal(str, str)
    setup_cancelled: Signal = Signal(str)

    def __init__(
        self,
        bus: BaseEventBus,
        state: SessionState,
        gate: ExclusionGate,
        bg: BackgroundExecutor,
        progress: ProgressHub,
        *,
        handles: OperationHandles | None = None,
        driver_factory: DriverFactoryPort | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._bus = bus
        self._state = state
        # Device composes both leaves (ADR-0019): Exclusion (device mutation vs
        # run / another mutation of the same device) + a Handle (operation_id +
        # await + cancel for setup). The app injects the exclusion gate, the
        # off-main executor and the progress hub via their session ports — this
        # session service never constructs app/Qt infrastructure itself.
        self._gate = gate
        self._handles = handles or OperationHandles()
        # Device execution is the OffMain-thread strategy (ADR-0019): connect /
        # disconnect carry no scopes; setup carries the progress scope + a
        # directly-polled stop_event (not an ActiveTask scope).
        self._bg = bg
        self._driver_factory = driver_factory or _default_driver_factory
        self._progress: ProgressHub = progress
        # Device state lives in State (the SSOT). This service holds only the
        # live driver (in GlobalDeviceManager), the worker threads, and the
        # in-flight operation transient below. Setup progress lives in the
        # shared ProgressService, keyed by this operation's token (owner = name).
        self._active_token: int | None = None
        self._active_kind: OperationKind | None = None
        self._active_name: str | None = None
        # Rollback buffer for the in-flight transition (worker-bound, not
        # serializable, so it stays here rather than in State).
        self._active_prior: DeviceState | None = None

    def start_connect_device(self, req: ConnectDeviceRequest) -> int:
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
        assert self._active_token is not None  # set by _begin_operation
        token = self._active_token
        self._submit_command(
            req.name,
            lambda: self._connect(req),
            on_done=lambda info: self._on_connect_succeeded(req, info),
        )
        return token

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

    def start_disconnect_device(self, req: DisconnectDeviceRequest) -> int:
        current = self._require_connected_device(req.name)
        self._begin_operation(
            OperationKind.DEVICE_DISCONNECT,
            req.name,
            replace(current, status=DeviceStatus.DISCONNECTING, error=None),
        )
        assert self._active_token is not None  # set by _begin_operation
        token = self._active_token
        self._submit_command(
            req.name,
            lambda: self._disconnect(req.name),
            on_done=lambda _result: self._on_disconnect_succeeded(req),
        )
        return token

    def start_setup_device(self, req: SetupDeviceRequest) -> int:
        from zcu_tools.device import GlobalDeviceManager

        current = self._require_connected_device(req.name)
        driver = cast(DeviceProtocol, GlobalDeviceManager.get_device(req.name))
        # Single owner of the cancellation flag: passed to both the gate (set on
        # cancel) and the worker (polls it / self-judges 'cancelled').
        stop_event = threading.Event()
        self._begin_operation(
            OperationKind.DEVICE_SETUP,
            req.name,
            replace(current, status=DeviceStatus.SETTING_UP, error=None),
            stop_event=stop_event,
        )
        assert self._active_token is not None  # set by _begin_operation
        token = self._active_token
        # Setup is the OffMain-thread strategy with the progress scope (no figure
        # routing). The stop_event is NOT an ActiveTask scope — the driver's
        # setup() polls it directly — so it is captured by the work closure, not
        # OffMainScopes. Cancellation is interpreted in _on_setup_done (we own the
        # stop_event); bg only reports done/failed.
        scopes = OffMainScopes(
            pbar_factory=self._progress.make_factory(token, owner_id=req.name)
        )
        name = req.name
        info = req.info

        def setup_work() -> object:
            driver.setup(info, stop_event=stop_event)
            return driver.get_info()

        try:
            self._bg.submit(
                setup_work,
                scopes,
                run_in_pool=False,
                on_done=lambda result: self._on_setup_done(name, stop_event, result),
                on_error=lambda exc: self._on_setup_failed(name, str(exc)),
            )
            self._bus.emit(
                DeviceSetupStartedPayload(name=req.name),
            )
        except Exception:
            self._abort_unstarted_operation(req.name)
            raise
        return token

    def cancel_device_operation(self, name: str) -> None:
        if (
            self._active_name != name
            or self._active_kind is not OperationKind.DEVICE_SETUP
            or self._active_token is None
        ):
            raise RuntimeError(f"No cancellable device setup is active for {name!r}")
        # Async notification via the handle: set the operation's stop_event and
        # return. The worker self-judges 'cancelled' and emits its cancelled
        # signal — no direct worker.cancel() coupling.
        self._handles.cancel(self._active_token)

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
        """Assemble the read-time projection of a device-state entry from State.

        Live setup progress is no longer spliced here — it is polled separately
        via operation.progress (by operation_id, ADR-0013 device↔run alignment).
        """
        return DeviceSnapshot(
            name=dev.name,
            type_name=dev.type_name,
            address=dev.address,
            status=dev.status,
            info=dev.info,
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

    def get_active_setup(self) -> DeviceSetupSnapshot | None:
        snapshot = self.get_active_device_operation()
        if snapshot is None or snapshot.status is not DeviceStatus.SETTING_UP:
            return None
        return DeviceSetupSnapshot(device_name=snapshot.name)

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

    def get_memory_device_address(self, name: str) -> str | None:
        dev = self._state.get_device(name)
        if dev is None or not dev.is_memory_only():
            return None
        return dev.address

    def get_device_unit(self, name: str) -> str:
        dev = self._state.get_device(name)
        if dev is None:
            return "none"
        mode_unit = _mode_dependent_unit(dev)
        if mode_unit is not None:
            return mode_unit
        return _DEVICE_DEFAULT_UNITS.get(dev.type_name, "none")

    def get_device_unit_strict(self, name: str) -> str:
        """Resolve the flux unit for binding a context to ``name``, Fast-Fail.

        Unlike the lenient :meth:`get_device_unit` (used for UI labels, which
        tolerates unknown devices by returning "none"), this is the binding
        path: the device must exist and its type must be on the unit whitelist
        (``_DEVICE_DEFAULT_UNITS``). Anything else raises — a context's flux
        value/unit must come from a *known* flux device, not an arbitrary one.
        """
        dev = self._state.get_device(name)
        if dev is None:
            raise DeviceRegistrationError(f"No such device: {name!r}")
        if dev.type_name not in _DEVICE_DEFAULT_UNITS:
            raise DeviceRegistrationError(
                f"Device {name!r} of type {dev.type_name!r} cannot bind a "
                f"flux context (supported: {sorted(_DEVICE_DEFAULT_UNITS)})"
            )
        mode_unit = _mode_dependent_unit(dev)
        if mode_unit is not None:
            return mode_unit
        return _DEVICE_DEFAULT_UNITS[dev.type_name]

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

    def poll_device_info(self, name: str) -> None:
        """Best-effort off-main live-read of a device's real driver values.

        The dialog-scoped poller calls this once per tick for the selected
        device. It is the off-main twin of :meth:`get_device_info`: the worker
        only *reads* the driver (the blocking SCPI I/O), and the main-thread
        ``on_done`` does the cache compare + bump + DEVICE_CHANGED emit, so the
        State main-thread invariant holds (the worker never touches State).

        Skip without raising when the device is not a live read target — not
        connected / memory-only, or being mutated (connect/disconnect/setup,
        i.e. SETTING_UP ramp): a poll must never compete with a mutation. A
        single read that fails (timeout / no response / driver gone) is logged
        and swallowed so the next tick still runs — this is the best-effort
        core, not a Fast-Fail path.
        """
        from zcu_tools.device.manager import GlobalDeviceManager

        dev = self._state.get_device(name)
        if dev is None or dev.is_memory_only():
            return
        if self._gate.is_device_mutating(name):
            return

        def read_work() -> BaseDeviceInfo:
            # Worker thread: pure driver read, no State touch, no emit.
            return GlobalDeviceManager.get_info(name)

        def on_read(info: object) -> None:
            # Main thread: the device may have gone away or started mutating
            # between submit and delivery; re-check before the cache refresh so
            # a late poll result never bumps a now-mutating / removed device.
            current = self._state.get_device(name)
            if current is None or current.is_memory_only():
                return
            if self._gate.is_device_mutating(name):
                return
            if self._state.refresh_device_info_cache(name, cast(BaseDeviceInfo, info)):
                self._emit_device_changed(name)

        def on_read_failed(exc: Exception) -> None:
            # Best-effort: a failed single read is logged and dropped so the
            # poller keeps ticking.
            logger.debug("device poll read failed: name=%r error=%s", name, exc)

        self._bg.submit(
            read_work,
            run_in_pool=True,
            on_done=on_read,
            on_error=on_read_failed,
        )

    def get_device_value_for_new_context(self, name: str) -> float | None:
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

    def _begin_operation(
        self,
        kind: OperationKind,
        name: str,
        pending: DeviceState,
        stop_event: threading.Event | None = None,
    ) -> None:
        # Symmetric release: this lease + _active_name/_active_prior are cleared
        # exactly-once on the terminal path via _finish_operation, called from
        # every _on_*_finished/_failed/_cancelled and _on_operation_failed.
        # stop_event is set only for cancellable operations (setup); connect /
        # disconnect have no cancellation point, so cancel is a no-op for them.
        # Compose both leaves (ADR-0019): fail-fast on conflict before the handle,
        # then mint the handle (holding stop_event) and register the exclusion
        # under the same token. _active_kind is tracked because outcome handling
        # (_on_operation_failed) branches on the operation kind.
        self._gate.ensure_can_start(kind)
        token = self._handles.create(stop_event=stop_event)
        self._gate.register(token, kind, owner_id=name, resource_id=name)
        prior = self._state.get_device(name)
        self._active_token = token
        self._active_kind = kind
        self._active_name = name
        self._active_prior = prior
        self._state.put_device(pending)
        # Announce the optimistic state. A DEVICE_CHANGED subscriber (e.g. a View
        # redraw) raising here is swallowed + logged by the EventBus and does NOT
        # abort the operation or roll back the optimistic write — the lease is
        # released at the real terminal (_finish_operation), never on a
        # subscriber's failure (one bad subscriber must not break the publisher).
        self._emit_device_changed(name)

    def _finish_operation(self, name: str, outcome: OperationOutcome) -> None:
        if self._active_name != name or self._active_token is None:
            raise RuntimeError(f"Device operation for {name!r} has no active operation")
        token = self._active_token
        self._active_name = None
        self._active_token = None
        self._active_kind = None
        self._active_prior = None
        # Destroy this operation's progress container (a no-op for connect/
        # disconnect, which never minted one; setup's leave=True bars never emit
        # CLOSE, so the terminal path must clear them), settle the handle, free
        # the exclusion.
        self._progress.discard_operation(token)
        self._handles.settle(token, outcome)
        self._gate.release(token)

    def _submit_command(
        self,
        name: str,
        work: Callable[[], object],
        on_done: Callable[[object], None],
    ) -> None:
        """Submit a connect/disconnect command off-main (OffMain-thread strategy,
        no scopes — no progress, no cancellation point). Aborts the in-flight
        transition if the submit fails to start."""
        try:
            self._bg.submit(
                work,
                run_in_pool=False,
                on_done=on_done,
                on_error=lambda exc: self._on_operation_failed(name, exc),
            )
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
        logger.info(
            "device connect succeeded: name=%r type=%r", req.name, req.type_name
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
        logger.info("device disconnect succeeded: name=%r", req.name)
        self._finish_operation(req.name, OperationOutcome("finished"))
        self._emit_device_changed(req.name)
        self.device_disconnected.emit(req)

    def _on_setup_done(
        self, name: str, stop_event: threading.Event, result: object
    ) -> None:
        # bg reports a normal return; we own the stop_event, so we interpret
        # cancellation here (ADR-0019): the driver's setup() returns normally even
        # when cancelled, so a set stop_event on a normal return is 'cancelled',
        # otherwise 'finished'. A raise goes straight to _on_setup_failed.
        if stop_event.is_set():
            self._on_setup_cancelled(name)
        else:
            self._on_setup_finished(name, result)

    def _on_setup_finished(self, name: str, info: object) -> None:
        logger.info("device setup finished: name=%r", name)
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
        self._emit_device_changed(name)
        self._bus.emit(
            DeviceSetupFinishedPayload(name=name, outcome="finished"),
        )
        self.setup_finished.emit(name)

    def _on_setup_failed(self, name: str, error: str) -> None:
        # ``error`` is a pre-formatted message (the exception was already consumed
        # into a string by the worker), so no exc_info is available here.
        logger.warning("device setup failed: name=%r error=%s", name, error)
        self._restore_prior_device(name, error)
        self._finish_operation(name, OperationOutcome("failed", error))
        self._emit_device_changed(name)
        self._bus.emit(
            DeviceSetupFinishedPayload(
                name=name, outcome="failed", error_message=error
            ),
        )
        self.setup_failed.emit(name, error)

    def _on_setup_cancelled(self, name: str) -> None:
        self._restore_prior_device(name, None)
        self._finish_operation(
            name, OperationOutcome("cancelled", f"device {name!r} setup was cancelled")
        )
        self._emit_device_changed(name)
        self._bus.emit(
            DeviceSetupFinishedPayload(name=name, outcome="cancelled"),
        )
        self.setup_cancelled.emit(name)

    def _on_operation_failed(self, name: str, error: object) -> None:
        message = str(error)
        # The real traceback is logged at the worker (background.py); ``error`` is
        # the marshalled value, so log the message at WARNING here.
        logger.warning("device operation failed: name=%r error=%s", name, message)
        if (
            self._active_kind is OperationKind.DEVICE_CONNECT
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
        self._bus.emit(DeviceChangedPayload(name=name))

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
