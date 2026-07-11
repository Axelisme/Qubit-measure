from __future__ import annotations

import importlib
import logging
import threading
from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    cast,
    runtime_checkable,
)

from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from zcu_tools.device.base import BaseDevice, BaseDeviceInfo
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.events import (
    DeviceChangedPayload,
    DeviceSetupFinishedPayload,
    DeviceSetupStartedPayload,
)
from zcu_tools.gui.session.operation_handles import OperationHandles, OperationOutcome
from zcu_tools.gui.session.operation_runner import (
    BgResult,
    ExclusionRequest,
    OperationRunner,
    OperationSpec,
    SettleFn,
)
from zcu_tools.gui.session.ports import (
    BackgroundExecutor,
    DeviceMemoryInfo,
    DeviceRegistryPort,
    DriverFactoryPort,
    ExclusionGate,
    OperationConflictError,
    OperationKind,
)
from zcu_tools.gui.session.scopes import progress_ambient
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

    - ``setup`` runs **off the main thread** (via ``BackgroundRunner``), never
      the Qt main thread. It MUST poll ``stop_event`` during any long operation:
      cancellation works by ``stop_event.set()``, and the worker reports
      "cancelled" only if it returns with the event set. A ``setup`` that ignores
      ``stop_event`` makes cancel a no-op (the worker blocks until natural
      completion). ``progress`` may drive a pbar via the ambient pbar factory.
    - ``get_info`` returns a fresh value snapshot. It is called on the worker
      right after ``setup``, by the synchronous idle read path, and by the
      off-main poll path that may refresh current values during setup/ramp. It
      must not mutate device state.
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
    # Fine-grained lifecycle status (DeviceStatus.value) rather than a coarse
    # bool, so the list view matches the snapshot/active-operations projections
    # (single-status SSOT; FC7). One of: memory_only | connecting | connected |
    # disconnecting | setting_up.
    status: str


@dataclass(frozen=True)
class DeviceSetupSnapshot:
    # Live progress is polled via operation.progress (by operation_id, ADR-0013
    # device↔run alignment), not carried here — this names *which* device is
    # setting up.
    device_name: str


@dataclass(frozen=True)
class ActiveDeviceOperation:
    """One in-flight device operation in the concurrent-enumeration view.

    Phase C runs operations for different devices in parallel, so the read-only
    "what is in flight" surface is a *set*, not a single op. Each entry pairs the
    device-state projection with the operation ``kind`` (connect / disconnect /
    setup) so an agent knows both which device and which kind of operation is
    live without re-deriving it from ``status``. ``device_name`` mirrors
    ``snapshot.name`` for callers that only need the key.
    """

    device_name: str
    kind: OperationKind
    snapshot: DeviceSnapshot
    # The operation handle (runner token), so a concurrent-enumeration reader can
    # drive gui_op_poll / gui_op_wait per in-flight op without re-resolving it by
    # device name.
    token: int


@dataclass(frozen=True)
class _InflightOp:
    """One in-flight device operation, keyed by device name in DeviceService.

    Phase C makes the device state machine per-device concurrent: connect /
    disconnect / setup of *different* devices run in parallel, so the formerly
    scalar ``_active_*`` fields become a ``dict[str, _InflightOp]`` keyed by the
    operation's device name (its resource_id). Each entry carries everything the
    terminal path needs for that device's operation, independent of any other:

    - ``token``: the operation token (handle + gate lease key), unique per op.
    - ``kind``: the operation kind — outcome handling branches on it
      (``_on_operation_failed`` distinguishes a failed first-connect from a
      failed mutation of an existing device).
    - ``prior``: the device-state rollback buffer captured at begin (the State
      entry before the optimistic write); ``None`` when there was no prior entry
      (a brand-new connect), which the rollback paths interpret as "remove".
    """

    token: int
    kind: OperationKind
    prior: DeviceState | None


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


def _extract_numeric_device_value(info: BaseDeviceInfo | None) -> float | None:
    """Return a strict numeric ``value`` field from cached/live device info."""
    if info is None:
        return None
    raw = getattr(info, "value", None)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw)


def list_supported_device_types() -> list[str]:
    return list(_DEVICE_TYPE_REGISTRY.keys())


class GlobalDeviceRegistryAdapter:
    """Thin instance adapter over ``GlobalDeviceManager``'s classmethods.

    ``GlobalDeviceManager`` is a classmethod-only singleton; this adapter wraps
    its five registry methods as instance methods so ``DeviceService`` can satisfy
    the instance-method ``DeviceRegistryPort`` Protocol without touching the
    singleton directly (ADR-0026 §D.2).  The wrapper adds no logic of its own.
    """

    def register_device(self, name: str, device: object) -> None:
        from zcu_tools.device import GlobalDeviceManager

        GlobalDeviceManager.register_device(name, cast(BaseDevice[Any], device))

    def drop_device(self, name: str, ignore_error: bool = False) -> None:
        from zcu_tools.device import GlobalDeviceManager

        GlobalDeviceManager.drop_device(name, ignore_error=ignore_error)

    def get_device(self, name: str) -> object:
        from zcu_tools.device import GlobalDeviceManager

        return GlobalDeviceManager.get_device(name)

    def get_all_devices(self) -> dict[str, object]:
        from zcu_tools.device import GlobalDeviceManager

        # Cast: GlobalDeviceManager returns dict[str, BaseDevice[Unknown]]; the
        # port contract uses dict[str, object] (covariance not expressible with
        # dict directly).  The values are never mutated through this view.
        return cast(dict[str, object], GlobalDeviceManager.get_all_devices())

    def get_info(self, name: str) -> object:
        from zcu_tools.device import GlobalDeviceManager

        return GlobalDeviceManager.get_info(name)


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
        runner: OperationRunner,
        handles: OperationHandles,
        *,
        driver_factory: DriverFactoryPort | None = None,
        device_registry: DeviceRegistryPort | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._bus = bus
        self._state = state
        # Device composes both leaves (ADR-0019): Exclusion (device mutation vs
        # run / another mutation of the same device) + a Handle (operation_id +
        # await + cancel for setup). OperationRunner owns the mechanism (ADR-0026 §1);
        # gate is kept directly for is_device_mutating / _reject_mutating_read;
        # bg is kept directly for poll_device_info (best-effort read, not a runner op).
        self._gate = gate
        self._bg = bg
        self._runner = runner
        # handles is kept for cancel_device_operation (handles.cancel) — the same
        # instance that runner holds internally.
        self._handles = handles
        self._driver_factory = driver_factory or _default_driver_factory
        # Registry port: hides the GlobalDeviceManager singleton behind an
        # instance-method interface so tests can inject an in-memory fake without
        # touching the real singleton (ADR-0026 §D).
        self._registry: DeviceRegistryPort = (
            device_registry
            if device_registry is not None
            else GlobalDeviceRegistryAdapter()
        )
        # Device state lives in State (the SSOT). This service holds only the
        # live driver (in the registry), the worker threads, and the in-flight
        # operation transients below. Setup progress lives in the shared
        # ProgressService, keyed by each operation's token (owner = name).
        #
        # Phase C: per-device concurrency. Connect / disconnect / setup of
        # different devices run in parallel, so the in-flight bookkeeping is a
        # dict keyed by device name (each op's resource_id) rather than a single
        # scalar set. The gate enforces that the SAME device has at most one
        # in-flight mutation, so each name maps to exactly one _InflightOp.
        self._inflight: dict[str, _InflightOp] = {}

    def start_connect_device(self, req: ConnectDeviceRequest) -> int:
        current = self._state.get_device(req.name)
        if current is not None and current.is_live():
            raise FailedPreconditionError(
                f"Device {req.name!r} is already connected or busy"
            )
        initial = current or DeviceState(
            name=req.name,
            type_name=req.type_name,
            address=req.address,
            status=DeviceStatus.MEMORY_ONLY,
            remember=req.remember,
        )
        prior = current  # capture before any State write (pre-open)
        name = req.name

        def work(factory: Any) -> Any:  # factory is None (wants_progress=False)
            return self._connect(req)

        def on_terminal(bg: BgResult, settle: SettleFn) -> None:
            if bg.ok:
                _on_connect_succeeded(cast(BaseDeviceInfo, bg.result), settle)
            else:
                assert bg.error is not None
                _on_operation_failed(str(bg.error), settle)

        def _on_connect_succeeded(info: BaseDeviceInfo, settle: SettleFn) -> None:
            self._state.put_device(
                DeviceState(
                    name=name,
                    type_name=req.type_name,
                    address=req.address,
                    status=DeviceStatus.CONNECTED,
                    remember=req.remember,
                    info=info,
                )
            )
            logger.info(
                "device connect succeeded: name=%r type=%r", name, req.type_name
            )
            self._inflight.pop(name, None)
            # State is observable before settle (invariant 1).
            settle(OperationOutcome("finished"))
            self._emit_device_changed(name)
            self.device_connected.emit(req)

        def _on_operation_failed(message: str, settle: SettleFn) -> None:
            logger.warning("device operation failed: name=%r error=%s", name, message)
            op = self._inflight.pop(name, None)
            # first-connect fail (no prior) → remove; mutation fail → restore prior
            if (
                op is not None
                and op.kind is OperationKind.DEVICE_CONNECT
                and prior is None
            ):
                self._state.remove_device(name)
            else:
                self._restore_prior_device_from(name, prior, message)
            # State is observable before settle.
            settle(OperationOutcome("failed", message))
            self._emit_device_changed(name)
            self.operation_failed.emit(name, message)

        spec = OperationSpec(
            exclusion=ExclusionRequest(
                kind=OperationKind.DEVICE_CONNECT,
                owner_id=name,
                note=f"connect device: {name}",
                resource_id=name,
            ),
            owner_id=name,
            wants_progress=False,
            cancel_hook=None,
            work=work,
            run_in_pool=False,
            on_terminal=on_terminal,
        )

        token = self._runner.begin(spec)

        # POST-BEGIN: register _inflight, write optimistic pending state, announce.
        self._inflight[name] = _InflightOp(
            token=token, kind=OperationKind.DEVICE_CONNECT, prior=prior
        )
        self._state.put_device(
            replace(initial, status=DeviceStatus.CONNECTING, error=None)
        )
        with self._bus.origin(self._handles.event_origin(token)):
            self._emit_device_changed(name)
        return token

    def start_reconnect_device(self, name: str) -> int:
        # Reconnect is a thin alias for connect of a remembered (memory-only)
        # device: it reuses the stored type/address. Return the connect
        # operation's token so the wire/MCP layers can expose it as an async
        # handle (FC1 — reconnect must produce an operation_id like the other
        # device starts, otherwise gui_op_wait cannot track a name-only reconnect).
        dev = self._require_device(name)
        if not dev.is_memory_only():
            raise FailedPreconditionError(
                f"Device {name!r} is not a memory-only device"
            )
        return self.start_connect_device(
            ConnectDeviceRequest(
                type_name=dev.type_name,
                name=dev.name,
                address=dev.address,
                remember=True,
            )
        )

    def start_disconnect_device(self, req: DisconnectDeviceRequest) -> int:
        current = self._require_connected_device(req.name)
        prior = current  # capture before any State write (pre-open)
        name = req.name

        def work(factory: Any) -> Any:  # factory is None (wants_progress=False)
            return self._disconnect(name)

        def on_terminal(bg: BgResult, settle: SettleFn) -> None:
            if bg.ok:
                _on_disconnect_succeeded(settle)
            else:
                assert bg.error is not None
                _on_operation_failed(str(bg.error), settle)

        def _on_disconnect_succeeded(settle: SettleFn) -> None:
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
                self._state.remove_device(name)
            logger.info("device disconnect succeeded: name=%r", name)
            self._inflight.pop(name, None)
            # State is observable before settle.
            settle(OperationOutcome("finished"))
            self._emit_device_changed(name)
            self.device_disconnected.emit(req)

        def _on_operation_failed(message: str, settle: SettleFn) -> None:
            logger.warning("device operation failed: name=%r error=%s", name, message)
            self._inflight.pop(name, None)
            self._restore_prior_device_from(name, prior, message)
            # State is observable before settle.
            settle(OperationOutcome("failed", message))
            self._emit_device_changed(name)
            self.operation_failed.emit(name, message)

        spec = OperationSpec(
            exclusion=ExclusionRequest(
                kind=OperationKind.DEVICE_DISCONNECT,
                owner_id=name,
                note=f"disconnect device: {name}",
                resource_id=name,
            ),
            owner_id=name,
            wants_progress=False,
            cancel_hook=None,
            work=work,
            run_in_pool=False,
            on_terminal=on_terminal,
        )

        token = self._runner.begin(spec)

        # POST-BEGIN: register _inflight, write optimistic pending state, announce.
        self._inflight[name] = _InflightOp(
            token=token, kind=OperationKind.DEVICE_DISCONNECT, prior=prior
        )
        self._state.put_device(
            replace(current, status=DeviceStatus.DISCONNECTING, error=None)
        )
        with self._bus.origin(self._handles.event_origin(token)):
            self._emit_device_changed(name)
        return token

    def start_setup_device(self, req: SetupDeviceRequest) -> int:
        current = self._require_connected_device(req.name)
        driver = cast(DeviceProtocol, self._registry.get_device(req.name))
        # Single owner of the cancellation flag: passed to both the gate (set on
        # cancel) and the worker (polls it / self-judges 'cancelled').
        stop_event = threading.Event()
        prior = current  # capture before any State write (pre-open)
        name = req.name
        info = req.info

        def work(factory: Any) -> Any:
            # Setup is the OffMain-thread strategy with the progress scope only (no
            # figure routing; the driver's setup() polls stop_event
            # directly). progress_ambient is session-layer (no Qt) so device.py
            # can import it without crossing the session→app boundary (ADR-0026 §2).
            with progress_ambient(factory):
                driver.setup(info, stop_event=stop_event)
                return driver.get_info()

        def on_terminal(bg: BgResult, settle: SettleFn) -> None:
            # bg reports outcome; we own stop_event, so we interpret cancellation
            # (ADR-0019): normal return + stop_event set = cancelled.
            if bg.ok:
                if stop_event.is_set():
                    _on_setup_cancelled(settle)
                else:
                    _on_setup_finished(cast(BaseDeviceInfo, bg.result), settle)
            else:
                assert bg.error is not None
                _on_setup_failed(str(bg.error), settle)

        def _on_setup_finished(info: BaseDeviceInfo, settle: SettleFn) -> None:
            logger.info("device setup finished: name=%r", name)
            current_dev = self._require_device(name)
            self._state.put_device(
                replace(
                    current_dev,
                    status=DeviceStatus.CONNECTED,
                    info=info,
                    error=None,
                )
            )
            self._inflight.pop(name, None)
            # State is observable before settle (invariant 1).
            settle(OperationOutcome("finished"))
            self._emit_device_changed(name)
            self._bus.emit(DeviceSetupFinishedPayload(name=name, outcome="finished"))
            self.setup_finished.emit(name)

        def _on_setup_failed(error: str, settle: SettleFn) -> None:
            logger.warning("device setup failed: name=%r error=%s", name, error)
            self._inflight.pop(name, None)
            self._restore_prior_device_from(name, prior, error)
            # State is observable before settle.
            settle(OperationOutcome("failed", error))
            self._emit_device_changed(name)
            self._bus.emit(
                DeviceSetupFinishedPayload(
                    name=name, outcome="failed", error_message=error
                )
            )
            self.setup_failed.emit(name, error)

        def _on_setup_cancelled(settle: SettleFn) -> None:
            self._inflight.pop(name, None)
            self._restore_prior_device_from(name, prior, None)
            # State is observable before settle.
            settle(
                OperationOutcome("cancelled", f"device {name!r} setup was cancelled")
            )
            self._emit_device_changed(name)
            self._bus.emit(DeviceSetupFinishedPayload(name=name, outcome="cancelled"))
            self.setup_cancelled.emit(name)

        spec = OperationSpec(
            exclusion=ExclusionRequest(
                kind=OperationKind.DEVICE_SETUP,
                owner_id=name,
                note=f"setup device: {name}",
                resource_id=name,
            ),
            owner_id=name,
            wants_progress=True,
            cancel_hook=stop_event.set,
            work=work,
            run_in_pool=False,
            on_terminal=on_terminal,
        )

        token = self._runner.begin(spec)

        # POST-BEGIN: register _inflight, write optimistic pending state, announce.
        self._inflight[name] = _InflightOp(
            token=token, kind=OperationKind.DEVICE_SETUP, prior=prior
        )
        self._state.put_device(
            replace(current, status=DeviceStatus.SETTING_UP, error=None)
        )
        with self._bus.origin(self._handles.event_origin(token)):
            self._emit_device_changed(name)
            self._bus.emit(DeviceSetupStartedPayload(name=name))
        return token

    def cancel_device_operation(self, name: str) -> None:
        op = self._inflight.get(name)
        if op is None:
            # No operation at all for this device — not busy.
            raise FailedPreconditionError(
                f"No operation in flight for device {name!r}."
            )
        if op.kind is not OperationKind.DEVICE_SETUP:
            # An op is in flight, but connect/disconnect have no cancellation point
            # (they run to natural completion; only apply/setup polls stop_event).
            raise FailedPreconditionError(
                f"Device {name!r} has a {op.kind.value} operation in flight, "
                f"which has no cancellation point (only apply/setup is cancellable)."
            )
        # Async notification via the handle: set the operation's stop_event and
        # return. The worker self-judges 'cancelled' and emits its cancelled
        # signal — no direct worker.cancel() coupling.
        self._handles.cancel(op.token)

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
            raise FailedPreconditionError(
                f"Device {name!r} is not a memory-only device"
            )
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

    def active_operation_token(self, device_name: str) -> int | None:
        """Return the handle token for the first in-flight op on ``device_name``,
        or None if no op is in progress for that device."""
        op = self._inflight.get(device_name)
        return op.token if op is not None else None

    def get_active_device_operations(self) -> tuple[ActiveDeviceOperation, ...]:
        """Enumerate *all* in-flight device operations (Phase C concurrency).

        Returns one entry per ``_inflight`` op — each carrying the device-state
        projection plus the operation ``kind`` — sorted by device name so the
        order is stable for agents (dict insertion order is not a contract). A
        device whose State entry has vanished is skipped (it has no projection)."""
        out: list[ActiveDeviceOperation] = []
        for name in sorted(self._inflight):
            dev = self._state.get_device(name)
            if dev is None:
                continue
            out.append(
                ActiveDeviceOperation(
                    device_name=name,
                    kind=self._inflight[name].kind,
                    snapshot=self._project(dev),
                    token=self._inflight[name].token,
                )
            )
        return tuple(out)

    def get_active_device_setups(self) -> tuple[DeviceSetupSnapshot, ...]:
        """Name *every* device currently setting up (Phase C concurrency).

        Filters the in-flight set to ``DEVICE_SETUP`` ops whose State status is
        ``SETTING_UP``, sorted by device name for a stable agent-facing order.
        The device dialog derives the same set from per-device snapshot status."""
        return tuple(
            DeviceSetupSnapshot(device_name=name)
            for name in sorted(self._inflight)
            if self._inflight[name].kind is OperationKind.DEVICE_SETUP
            and (dev := self._state.get_device(name)) is not None
            and dev.status is DeviceStatus.SETTING_UP
        )

    def list_devices(self) -> list[DeviceEntry]:
        return [
            DeviceEntry(
                name=snapshot.name,
                type_name=snapshot.type_name,
                status=snapshot.status.value,
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

    def get_cached_device_value(self, name: str) -> float | None:
        """Return the cached numeric predictor/flux value for a live device.

        This is a cheap State-only query: it does not touch hardware, refresh the
        cache, bump versions, or emit events. The type whitelist intentionally
        matches the existing flux-device unit whitelist so RF sources are not
        treated as predictor device-value providers by accident.
        """
        dev = self._state.get_device(name)
        if dev is None or dev.status is not DeviceStatus.CONNECTED:
            return None
        if dev.type_name not in _DEVICE_DEFAULT_UNITS:
            return None
        return _extract_numeric_device_value(dev.info)

    def get_device_info(self, name: str) -> BaseDeviceInfo | None:
        self._reject_mutating_read(name)
        dev = self._state.get_device(name)
        if dev is None or dev.is_memory_only():
            return None
        try:
            info = cast(BaseDeviceInfo, self._registry.get_info(name))
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

        Skip without raising when the device is not a live read target —
        missing / memory-only, connecting, disconnecting, or under an unknown
        mutation. ``DEVICE_SETUP`` is the one mutation whose driver read is
        allowed: setup/ramp code owns the write lock, while ``get_info`` is a
        best-effort current-value read. A single read that fails (timeout / no
        response / driver gone) is logged and swallowed so the next tick still
        runs — this is the best-effort core, not a Fast-Fail path.
        """
        if not self._can_poll_device_info(name):
            return

        def read_work() -> BaseDeviceInfo:
            # Worker thread: pure driver read, no State touch, no emit.
            return cast(BaseDeviceInfo, self._registry.get_info(name))

        def on_read(info: object) -> None:
            # Main thread: the device may have gone away or started a non-setup
            # mutation between submit and delivery; re-check before the cache
            # refresh so a late poll result never bumps that State.
            if not self._can_poll_device_info(name):
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

    def _can_poll_device_info(self, name: str) -> bool:
        """Return whether ``poll_device_info`` may live-read this device now.

        The concrete gate only exposes the coarse "device is mutating" question.
        DeviceService owns the finer in-flight metadata, so setup can be the
        only admitted mutation without expanding the gate contract.
        """
        dev = self._state.get_device(name)
        if dev is None or dev.is_memory_only():
            return False
        op = self._inflight.get(name)
        if op is not None:
            return (
                op.kind is OperationKind.DEVICE_SETUP
                and dev.status is DeviceStatus.SETTING_UP
            )
        if dev.status is not DeviceStatus.CONNECTED:
            return False
        return not self._gate.is_device_mutating(name)

    def get_device_value_for_new_context(self, name: str) -> float | None:
        info = self.get_device_info(name)
        return _extract_numeric_device_value(info)

    def _connect(self, req: ConnectDeviceRequest) -> BaseDeviceInfo:
        if req.name in self._registry.get_all_devices():
            raise DeviceRegistrationError(f"Device {req.name!r} is already registered")
        device: DeviceProtocol | None = None
        registered = False
        try:
            device = self._driver_factory(req.type_name, req.address)
            self._registry.register_device(req.name, device)
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

    def _cleanup_failed_connection(
        self, name: str, device: DeviceProtocol | None, registered: bool
    ) -> None:
        # Was @staticmethod; now instance method so it can use self._registry.
        if registered:
            self._registry.drop_device(name, ignore_error=True)
        if device is not None:
            try:
                device.close()
            except Exception:
                logger.exception(
                    "Failed to close device %r after connect failure", name
                )

    def _disconnect(self, name: str) -> object:
        # Was @staticmethod; now instance method so it can use self._registry.
        device = cast(DeviceProtocol, self._registry.get_device(name))
        device.close()
        self._registry.drop_device(name)
        return None

    def _restore_prior_device_from(
        self, name: str, prior: DeviceState | None, error: str | None
    ) -> None:
        """Rollback device State to ``prior`` (or a memory-only stub if None).

        Replaces the old ``_restore_prior_device`` which read prior from _inflight;
        this version receives prior directly so it works whether or not the inflight
        entry is still present (it may already be popped by the caller).
        """
        if prior is None:
            pending = self._state.get_device(name)
            if pending is None:
                return
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
            raise FailedPreconditionError(f"Device {name!r} is not known")
        return dev

    def _require_connected_device(self, name: str) -> DeviceState:
        dev = self._require_device(name)
        if not dev.is_connected():
            raise FailedPreconditionError(f"Device {name!r} is not connected")
        return dev
