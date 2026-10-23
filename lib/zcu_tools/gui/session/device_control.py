"""Narrow device-control facet for shared UI and remote driving adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.events import (
    DeviceChangedPayload,
    DeviceSetupFinishedPayload,
    DeviceSetupStartedPayload,
)

if TYPE_CHECKING:
    from zcu_tools.device import DeviceActivity
    from zcu_tools.device.base import BaseDeviceInfo
    from zcu_tools.gui.session.device_models import (
        ActiveDeviceOperation,
        ConnectDeviceRequest,
        DeviceEntry,
        DeviceSnapshot,
        DisconnectDeviceRequest,
        SetupDeviceRequest,
    )
    from zcu_tools.gui.session.pbar_host import ProgressBarModel
    from zcu_tools.gui.session.progress_control import OwnerProgressPort
    from zcu_tools.gui.session.services.device import DeviceService


class DeviceControlPort(Protocol):
    """Device lifecycle/query/progress surface for shared consumers."""

    def on_device_changed(
        self, handler: Callable[[DeviceChangedPayload], None]
    ) -> Callable[[], None]: ...
    def on_device_setup_started(
        self, handler: Callable[[DeviceSetupStartedPayload], None]
    ) -> Callable[[], None]: ...
    def on_device_setup_finished(
        self, handler: Callable[[DeviceSetupFinishedPayload], None]
    ) -> Callable[[], None]: ...

    def start_connect_device(self, req: ConnectDeviceRequest) -> int: ...
    def start_disconnect_device(self, req: DisconnectDeviceRequest) -> int: ...
    def start_reconnect_device(self, name: str) -> int: ...
    def start_setup_device(self, req: SetupDeviceRequest) -> int: ...
    def rename_device(self, old_name: str, new_name: str) -> None:
        """Rename an idle MEMORY_ONLY row verbatim on the owner thread.

        Empty/whitespace or conflicting names raise InvalidInputError; unknown,
        live or in-flight rows raise FailedPreconditionError, including same-name
        requests. Otherwise same-name is a no-op. Success refreshes live values
        and remembered settings, not aliases, tabs or historical captures.
        """
        ...

    def forget_device(self, name: str) -> None: ...
    def cancel_device_operation(self, name: str) -> None: ...

    def activity_snapshot(self) -> dict[str, DeviceActivity]:
        """Read detached named driver activity on the session owner thread.

        No hardware I/O or operation-lock wait occurs. The caller owns the
        mapping; immutable values are non-atomic observations across fields
        and drivers. End times use process-local monotonic seconds. Never use
        this query for admission or settlement. Registry query errors propagate.
        """
        ...

    def list_devices(self) -> list[DeviceEntry]: ...
    def get_device_snapshot(self, name: str) -> DeviceSnapshot | None: ...
    def get_device_info(self, name: str) -> BaseDeviceInfo | None: ...
    def get_cached_device_value(self, name: str) -> float | None: ...
    def poll_device_info(self, name: str) -> None: ...
    def try_poll_device_info(self, name: str) -> bool:
        """Return whether a live read was submitted, not whether it completed.

        Missing, memory-only, unsafe mutation and closed-executor targets skip
        without raising and return False. Later read/cache delivery may fail
        even after True. This call must run on the session owner thread.
        """
        ...

    def is_memory_device(self, name: str) -> bool: ...
    def get_device_unit(self, name: str) -> str: ...
    def get_active_device_operations(self) -> tuple[ActiveDeviceOperation, ...]: ...

    def attach_progress(
        self, owner_id: str, listener: Callable[[], None]
    ) -> Callable[[], None]: ...
    def progress_bars(
        self, owner_id: str
    ) -> tuple[tuple[int, ProgressBarModel], ...]: ...


class DeviceControlFacet:
    """Composite adapter over device service, progress service, and event bus."""

    def __init__(
        self,
        *,
        bus: BaseEventBus,
        device: DeviceService,
        progress: OwnerProgressPort,
    ) -> None:
        self._bus = bus
        self._device = device
        self._progress = progress

    def on_device_changed(
        self, handler: Callable[[DeviceChangedPayload], None]
    ) -> Callable[[], None]:
        return self._bus.subscribe(DeviceChangedPayload, handler).unsubscribe

    def on_device_setup_started(
        self, handler: Callable[[DeviceSetupStartedPayload], None]
    ) -> Callable[[], None]:
        return self._bus.subscribe(DeviceSetupStartedPayload, handler).unsubscribe

    def on_device_setup_finished(
        self, handler: Callable[[DeviceSetupFinishedPayload], None]
    ) -> Callable[[], None]:
        return self._bus.subscribe(DeviceSetupFinishedPayload, handler).unsubscribe

    def start_connect_device(self, req: ConnectDeviceRequest) -> int:
        return self._device.start_connect_device(req)

    def start_disconnect_device(self, req: DisconnectDeviceRequest) -> int:
        return self._device.start_disconnect_device(req)

    def start_reconnect_device(self, name: str) -> int:
        return self._device.start_reconnect_device(name)

    def start_setup_device(self, req: SetupDeviceRequest) -> int:
        return self._device.start_setup_device(req)

    def rename_device(self, old_name: str, new_name: str) -> None:
        """Apply DeviceControlPort.rename_device's owner and failure contract."""
        self._device.rename_device(old_name, new_name)

    def forget_device(self, name: str) -> None:
        self._device.forget_device(name)

    def cancel_device_operation(self, name: str) -> None:
        self._device.cancel_device_operation(name)

    def activity_snapshot(self) -> dict[str, DeviceActivity]:
        """Read detached named driver activity on the session owner thread.

        The caller owns the mapping; values are immutable. No hardware I/O or
        operation-lock wait occurs. Fields/drivers are not observed atomically.
        End times are process-local monotonic seconds. This query grants no
        admission and does not settle operations. Registry errors propagate.
        """
        return self._device.activity_snapshot()

    def list_devices(self) -> list[DeviceEntry]:
        return self._device.list_devices()

    def get_device_snapshot(self, name: str) -> DeviceSnapshot | None:
        return self._device.get_device_snapshot(name)

    def get_device_info(self, name: str) -> BaseDeviceInfo | None:
        return self._device.get_device_info(name)

    def get_cached_device_value(self, name: str) -> float | None:
        return self._device.get_cached_device_value(name)

    def poll_device_info(self, name: str) -> None:
        self._device.poll_device_info(name)

    def try_poll_device_info(self, name: str) -> bool:
        """Return the submission receipt described by DeviceControlPort."""
        return self._device.poll_device_info(name)

    def is_memory_device(self, name: str) -> bool:
        return self._device.is_memory_device(name)

    def get_device_unit(self, name: str) -> str:
        return self._device.get_device_unit(name)

    def get_active_device_operations(self) -> tuple[ActiveDeviceOperation, ...]:
        return self._device.get_active_device_operations()

    def attach_progress(
        self, owner_id: str, listener: Callable[[], None]
    ) -> Callable[[], None]:
        return self._progress.attach_by_owner(owner_id, listener)

    def progress_bars(self, owner_id: str) -> tuple[tuple[int, ProgressBarModel], ...]:
        return self._progress.bars_for_owner(owner_id)
