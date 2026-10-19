from __future__ import annotations

import importlib
import logging
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    cast,
    runtime_checkable,
)

from zcu_tools.device import DeviceActivity, DeviceInfo
from zcu_tools.device.base import BaseDevice, BaseDeviceInfo
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.device.yoko import YOKOGS200Info
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.owner import OwnerThreadGuard
from zcu_tools.gui.session.device_errors import DeviceRegistrationError
from zcu_tools.gui.session.events import (
    DeviceChangedPayload,
    DeviceOperationFinishedPayload,
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
    BackgroundAdmissionClosedError,
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
    # Live progress is polled via operation.progress (by operation_id, ADR-0068
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
    # await each in-flight GUI operation by token, or expose it through MCP
    # status() as an agent handle, without re-resolving by device name.
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
    - ``kind``: the operation kind, used to distinguish setup from other mutations.
    - ``prior``: the device-state rollback buffer captured at begin (the State
      entry before the optimistic write); ``None`` for a brand-new connect.
      Failed first connections retain a memory-only row with the error.
    """

    token: int
    kind: OperationKind
    prior: DeviceState | None


_DEVICE_TYPE_REGISTRY: dict[str, tuple[str, bool]] = {
    "YOKOGS200": ("zcu_tools.device.yoko.YOKOGS200", True),
    "RohdeSchwarzSGS100A": ("zcu_tools.device.sgs100a.RohdeSchwarzSGS100A", True),
}

_FLUX_DEVICE_TYPES = {"FakeDevice", "YOKOGS200"}


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


class DeviceService:
    """Own device mutation workers and cached render snapshots."""

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
    ) -> None:
        self._bus = bus
        self._state = state
        self._owner_guard = OwnerThreadGuard()
        # Device composes both leaves (ADR-0066): Exclusion (device mutation vs
        # run / another mutation of the same device) + a Handle (operation_id +
        # await + cancel for setup). OperationRunner owns the mechanism (ADR-0066);
        # gate is kept directly for is_device_mutating / _reject_mutating_read;
        # bg is kept directly for poll_device_info (best-effort read, not a runner op).
        self._gate = gate
        self._bg = bg
        self._runner = runner
        # handles is kept for cancel_device_operation (handles.cancel) — the same
        # instance that runner holds internally.
        self._handles = handles
        self._driver_factory = driver_factory or _default_driver_factory
        from zcu_tools.device import DeviceManager

        self._registry: DeviceRegistryPort = (
            device_registry if device_registry is not None else DeviceManager()
        )
        # State owns serializable snapshots; this service owns live drivers and
        # per-name operation transients. The gate prevents concurrent mutations
        # of the same device. ProgressService owns per-operation setup progress.
        self._inflight: dict[str, _InflightOp] = {}

    def register_device(self, name: str, driver: BaseDevice[Any]) -> None:
        """Adopt an already built driver and publish its current info on owner thread.

        ``name`` is a nonblank session name, preserved verbatim. ``driver`` must
        be a BaseDevice; its info supplies type and address. Replace only an idle
        MEMORY_ONLY row, preserving its remember flag, or create a nonremembered
        row. Reject live names with FailedPreconditionError, invalid inputs with
        InvalidInputError, and Run or same-device mutation with
        OperationConflictError. Info/registry failures propagate without changing
        the cache. Ownership transfers to this service only on success; rejected
        candidates remain caller-owned and are not closed. No driver is rebuilt.
        """
        self._owner_guard.assert_owner()
        if not name.strip():
            raise InvalidInputError("Device name must not be empty or all whitespace")
        with self._device_lease(name, OperationKind.DEVICE_CONNECT):
            prior = self._state.get_device(name)
            if (
                prior is not None and prior.is_live()
            ) or name in self._registry.get_all_devices():
                raise FailedPreconditionError(f"Device {name!r} is already registered")
            info = driver.get_info()
            self._registry.register_device(name, driver)
            self._state.put_device(
                DeviceState(
                    name=name,
                    type_name=info.type,
                    address=info.address,
                    status=DeviceStatus.CONNECTED,
                    remember=prior.remember if prior is not None else False,
                    info=info,
                )
            )
            self._emit_device_changed(name)

    def close_all_devices(self) -> None:
        """Try every registered driver once, publishing each result on owner thread.

        Call after app work drains. Run or a device mutation rejects the whole
        call before any close. Successful closes remove registry entries and
        publish MEMORY_ONLY rows; failures retain CONNECTED rows with error.
        Raise ExceptionGroup containing all close failures after every attempt.
        No RF/output changes or retries are implied. Remaining drivers stay
        service-owned; another explicit call may retry them. GUI teardown must
        isolate/log this aggregate rather than prevent window closure.
        """
        self._owner_guard.assert_owner()
        names = tuple(self._registry.get_all_devices())
        for name in names:
            self._gate.ensure_can_start(
                OperationKind.DEVICE_DISCONNECT, resource_id=name
            )
        errors: list[Exception] = []
        for name in names:
            prior = self._state.get_device(name)
            with self._device_lease(name, OperationKind.DEVICE_DISCONNECT):
                try:
                    self._registry.close_device(name)
                except Exception as exc:
                    errors.append(exc)
                    logger.exception("Failed to close registered device %r", name)
                    if prior is not None:
                        self._state.put_device(
                            replace(
                                prior, status=DeviceStatus.CONNECTED, error=str(exc)
                            )
                        )
                else:
                    if prior is not None:
                        self._state.put_device(
                            replace(prior, status=DeviceStatus.MEMORY_ONLY, error=None)
                        )
                if prior is not None:
                    self._emit_device_changed(name)
        if errors:
            raise ExceptionGroup("Failed to close registered devices", errors)

    @contextmanager
    def _device_lease(self, name: str, kind: OperationKind) -> Generator[None]:
        self._gate.ensure_can_start(kind, resource_id=name)
        origin = self._bus.current_origin
        token = self._handles.create(cancel_hook=None, origin=origin)
        self._gate.register(
            token,
            kind,
            owner_id=name,
            origin_kind=origin.kind,
            note=f"{kind.value}: {name}",
            resource_id=name,
        )
        try:
            yield
        except Exception as exc:
            self._handles.settle(token, OperationOutcome("failed", str(exc)))
            raise
        else:
            self._handles.settle(token, OperationOutcome("finished"))
        finally:
            self._gate.release(token)

    def publish_initial_devices(self) -> None:
        """Publish injected live drivers before preference restore, on owner loop.

        No drivers are reconstructed or connected. Read failure propagates;
        the caller must drain work and close the service's registered drivers.
        Existing published rows are rejected rather than overwritten.
        """
        entries = self._registry.get_all_devices()
        infos = {name: driver.get_info() for name, driver in entries.items()}
        for name, info in infos.items():
            if self._state.get_device(name) is not None:
                raise ValueError(f"device already published: {name}")
            self._state.put_device(
                DeviceState(
                    name=name,
                    type_name=info.type,
                    address=info.address,
                    status=DeviceStatus.CONNECTED,
                    remember=False,
                    info=info,
                )
            )
            self._emit_device_changed(name)

    def start_connect_device(self, req: ConnectDeviceRequest) -> int:
        """Start an off-main connection and return its operation token.

        Live or busy names raise FailedPreconditionError. Driver failures settle
        the operation as failed and retain an offline snapshot with the reason,
        allowing callers to submit the same name with a corrected address.
        """
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

        def work(_factory: Any) -> Any:  # factory is None (wants_progress=False)
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
            self._bus.emit(
                DeviceOperationFinishedPayload(
                    name=name,
                    action="connect",
                    success=True,
                )
            )

        def _on_operation_failed(message: str, settle: SettleFn) -> None:
            logger.warning("device operation failed: name=%r error=%s", name, message)
            self._inflight.pop(name, None)
            self._restore_prior_device_from(name, prior, message)
            # State is observable before settle.
            settle(OperationOutcome("failed", message))
            self._emit_device_changed(name)
            self._bus.emit(
                DeviceOperationFinishedPayload(
                    name=name,
                    action="connect",
                    success=False,
                    error_message=message,
                )
            )

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
        # device starts, otherwise MCP wait(op) cannot track a name-only reconnect).
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

        def work(_factory: Any) -> Any:  # factory is None (wants_progress=False)
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
            self._bus.emit(
                DeviceOperationFinishedPayload(
                    name=name,
                    action="disconnect",
                    success=True,
                )
            )

        def _on_operation_failed(message: str, settle: SettleFn) -> None:
            logger.warning("device operation failed: name=%r error=%s", name, message)
            self._inflight.pop(name, None)
            self._restore_prior_device_from(name, prior, message)
            # State is observable before settle.
            settle(OperationOutcome("failed", message))
            self._emit_device_changed(name)
            self._bus.emit(
                DeviceOperationFinishedPayload(
                    name=name,
                    action="disconnect",
                    success=False,
                    error_message=message,
                )
            )

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
            # can import it without crossing the session→app boundary (ADR-0066).
            with progress_ambient(factory):
                driver.setup(info, stop_event=stop_event)
                return driver.get_info()

        def on_terminal(bg: BgResult, settle: SettleFn) -> None:
            # bg reports outcome; we own stop_event, so we interpret cancellation
            # (ADR-0066): normal return + stop_event set = cancelled.
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

        def _on_setup_cancelled(settle: SettleFn) -> None:
            self._inflight.pop(name, None)
            self._restore_prior_device_from(name, prior, None)
            # State is observable before settle.
            settle(
                OperationOutcome("cancelled", f"device {name!r} setup was cancelled")
            )
            self._emit_device_changed(name)
            self._bus.emit(DeviceSetupFinishedPayload(name=name, outcome="cancelled"))

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

    def rename_device(self, old_name: str, new_name: str) -> None:
        """Rename a MEMORY_ONLY device on the session owner thread, without I/O.

        Names are preserved verbatim. Empty or all-whitespace names and names
        already used by another device raise InvalidInputError. Unknown, live,
        or in-flight devices raise FailedPreconditionError, even for a same-name
        request. A same-name request on an idle memory row is a silent no-op.
        Rejection leaves all state unchanged. Success preserves the row's other
        fields and publishes one whole-device-set change after both keys have
        been updated, refreshing value sources and future settings captures.
        Aliases, open tabs and historical captures are not rewritten.
        """
        dev = self._require_device(old_name)
        if old_name in self._inflight or self._gate.is_device_mutating(old_name):
            raise FailedPreconditionError(
                f"Device {old_name!r} has an in-flight operation; wait before renaming"
            )
        if not dev.is_memory_only():
            raise FailedPreconditionError(
                f"Disconnect device {old_name!r} before renaming"
            )
        if not new_name or not new_name.strip():
            raise InvalidInputError("Device name must not be empty or all whitespace")
        if new_name == old_name:
            return
        if (
            self._state.has_device(new_name)
            or new_name in self._registry.get_all_devices()
        ):
            raise InvalidInputError(f"Device name {new_name!r} is already in use")
        # No callbacks or I/O between the two writes: observers see only the
        # committed set, and MEMORY_ONLY has no driver or lease key to migrate.
        self._state.remove_device(old_name)
        self._state.put_device(replace(dev, name=new_name))
        self._bus.emit(DeviceChangedPayload(name=None))

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
        via operation.progress (by operation_id, ADR-0068 device↔run alignment).
        """
        return DeviceSnapshot(
            name=dev.name,
            type_name=dev.type_name,
            address=dev.address,
            status=dev.status,
            info=dev.info,
            error=dev.error,
        )

    def activity_snapshot(self) -> dict[str, DeviceActivity]:
        """Observe named drivers on the session owner without hardware I/O.

        Return a caller-owned mapping of detached immutable activity values.
        No operation/I/O locks are awaited; fields and drivers are not atomic.
        End timestamps are process-local monotonic seconds, not wall time.
        This query does not grant admission or settle operations. Registry
        query failures propagate.
        """
        return self._registry.activity_snapshot()

    def refresh_connected_device_info(self) -> None:
        """Schedule one best-effort background observation per connected driver.

        Call on the session owner after a run's terminal delivery. Each device
        uses poll_device_info: successful changed reads commit cache and emit
        on the owner; failed reads log and retain the previous cache. Missing,
        disconnecting or otherwise ineligible devices and closed background
        admission are skipped. No retry, handle, lease or run evidence changes.
        Hardware I/O runs off-owner and can block. Unexpected submit/registry
        errors propagate, just as for the individual poll command.
        """
        for name in self.get_connected_devices():
            self.poll_device_info(name)

    def get_connected_devices(self) -> dict[str, BaseDevice[Any]]:
        """Borrow the registered drivers without transferring lifecycle ownership."""
        return dict(self._registry.get_all_devices())

    def capture_live_native_device_info(self) -> dict[str, DeviceInfo]:
        """Synchronously read every registered connected driver for provenance.

        Call on the session owner thread. Read each device once through
        get_device_info, which owns mutation checks and State cache refresh.
        Return detached native DeviceInfo models in registry order; no connected
        devices returns an empty mapping. Missing reads, unsupported native info
        and read exceptions raise RuntimeError naming the failing device. Never
        substitute cached info or retry. Earlier successful reads may already
        have refreshed State when a later read fails; there is no rollback.
        Driver I/O can block, and a ramp can expose an intermediate value.
        """
        captured: dict[str, DeviceInfo] = {}
        for name in self.get_connected_devices():
            try:
                info = self.get_device_info(name)
            except Exception as error:
                raise RuntimeError(f"Device {name!r} live read failed") from error
            if not isinstance(info, DeviceInfo):
                raise RuntimeError(f"Device {name!r} has no native live info")
            captured[name] = info.model_copy(deep=True)
        return captured

    def capture_native_device_info(self) -> dict[str, DeviceInfo]:
        """Capture ordered native info for live registered devices without I/O.

        Call on the session owner thread. Values come only from State's cached
        info, not driver get_info or a poll. Remembered-only devices are absent.
        Missing info or a cached model outside the native DeviceInfo union raises
        ValueError instead of silently omitting a live device. Return detached
        native models for FormEnv and startup DeviceSnapshot; drivers and their
        lifecycle remain session-owned. No setup or hardware read occurs.
        """
        captured: dict[str, DeviceInfo] = {}
        for device in self._state.list_devices():
            if not device.is_live():
                continue
            info = device.info
            if not isinstance(info, DeviceInfo):
                raise ValueError(f"Device {device.name!r} has no native cached info")
            captured[device.name] = info.model_copy(deep=True)
        return captured

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
        """Read cached flux Info.value_unit for a UI label, without hardware I/O.

        Missing device/Info and non-flux Info return none. This read does not
        infer a unit from device type or change connection/cache state.
        """
        dev = self._state.get_device(name)
        if dev is not None and isinstance(dev.info, (YOKOGS200Info, FakeDeviceInfo)):
            return dev.info.value_unit
        return "none"

    def get_device_unit_strict(self, name: str) -> str:
        """Read a cached flux unit for context binding, without hardware I/O.

        The device must be registered as YOKOGS200 or FakeDevice and have valid
        flux Info. Otherwise DeviceRegistrationError names the device. Missing
        Info never selects a unit from type; connection/cache state is unchanged.
        """
        dev = self._state.get_device(name)
        if dev is None:
            raise DeviceRegistrationError(f"No such device: {name!r}")
        if dev.type_name not in _FLUX_DEVICE_TYPES:
            raise DeviceRegistrationError(
                f"Device {name!r} of type {dev.type_name!r} cannot bind a "
                f"flux context (supported: {sorted(_FLUX_DEVICE_TYPES)})"
            )
        if not isinstance(dev.info, (YOKOGS200Info, FakeDeviceInfo)):
            raise DeviceRegistrationError(f"Device {name!r} has no cached flux Info")
        return dev.info.value_unit

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
        if dev.type_name not in _FLUX_DEVICE_TYPES:
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
        runs — this is the best-effort core, not a Fast-Fail path. Terminal
        executor admission rejection also skips the tick without scheduling I/O.
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

        try:
            self._bg.submit(
                read_work,
                run_in_pool=True,
                on_done=on_read,
                on_error=on_read_failed,
            )
        except BackgroundAdmissionClosedError:
            # Terminal shutdown makes further polling unnecessary.
            return

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
            self._cleanup_failed_connection(req.name, device, registered=registered)
            raise
        except Exception as exc:
            self._cleanup_failed_connection(req.name, device, registered=registered)
            raise DeviceRegistrationError(
                f"Failed to connect {req.type_name} {req.name!r}: {exc}"
            ) from exc

    def _cleanup_failed_connection(
        self, name: str, device: DeviceProtocol | None, *, registered: bool
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
        self._registry.close_device(name)
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
