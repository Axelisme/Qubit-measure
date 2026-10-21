"""Device requests and read projections shared by session services and frontends."""

from __future__ import annotations

from dataclasses import dataclass

from zcu_tools.device.base import BaseDeviceInfo
from zcu_tools.gui.session.ports import OperationKind
from zcu_tools.gui.session.state import DeviceStatus


@dataclass(frozen=True)
class ConnectDeviceRequest:
    """Inputs for connecting a new or memory-only device.

    Attributes:
        type_name: Driver type name passed to the driver factory.
        name: Session device key, either new or currently memory-only.
        address: Connection address passed to the driver factory.
        remember: Remember flag committed to State on successful connection.
            Defaults to True; this does not itself write preferences to disk.
    """

    type_name: str
    name: str
    address: str
    remember: bool = True


@dataclass(frozen=True)
class DisconnectDeviceRequest:
    """Inputs for disconnecting a connected device.

    Attributes:
        name: Session key of the connected device to disconnect.
        remember: On success, True retains a memory-only row with its type and
            address; False removes the row. Defaults to True.
    """

    name: str
    remember: bool = True


@dataclass(frozen=True)
class SetupDeviceRequest:
    """Configuration to apply to a connected device through its driver.

    Attributes:
        name: Session key of the connected device to configure.
        info: Device-specific configuration passed to the driver's setup method.
    """

    name: str
    info: BaseDeviceInfo


@dataclass(frozen=True)
class DeviceSnapshot:
    """Read-time projection of State-owned device data for UI and remote readers.

    State remains the device-state authority; this projection is not stored.
    Frozen fields do not make the referenced info deeply immutable. Live setup
    progress is queried separately through operation.progress by operation ID.

    Attributes:
        name: Session device key.
        type_name: Driver type name recorded in State.
        address: Connection address recorded in State.
        status: Lifecycle state: MEMORY_ONLY, CONNECTING, CONNECTED,
            DISCONNECTING, or SETTING_UP.
        info: Cached device info reference, or None when no info is available.
            It is not a fresh hardware read or a deep copy.
        error: Recorded operation error message, or None when no error is set.
    """

    name: str
    type_name: str
    address: str
    status: DeviceStatus
    info: BaseDeviceInfo | None = None
    error: str | None = None


@dataclass(frozen=True)
class DeviceEntry:
    """Device identity and lifecycle status for list views and selectors.

    Attributes:
        name: Session device key.
        type_name: Driver type name recorded in State.
        status: DeviceStatus.value: memory_only, connecting, connected,
            disconnecting, or setting_up.
    """

    name: str
    type_name: str
    status: str


@dataclass(frozen=True)
class DeviceSetupSnapshot:
    """Identity of a device with an in-flight setup and SETTING_UP status.

    Live progress is queried separately through operation.progress.

    Attributes:
        device_name: Session key of the device currently setting up.
    """

    device_name: str


@dataclass(frozen=True)
class ActiveDeviceOperation:
    """One in-flight device operation in the concurrent-enumeration view.

    Different devices can have operations in flight concurrently. The kind
    identifies the command without deriving it from the projected status.

    Attributes:
        device_name: Session device key, matching snapshot.name.
        kind: DEVICE_CONNECT, DEVICE_DISCONNECT, or DEVICE_SETUP from
            the session OperationKind vocabulary.
        snapshot: Read-time projection of the device's current State entry.
        token: Runner operation handle, used to await the operation or expose
            its status through remote operation APIs.
    """

    device_name: str
    kind: OperationKind
    snapshot: DeviceSnapshot
    token: int
