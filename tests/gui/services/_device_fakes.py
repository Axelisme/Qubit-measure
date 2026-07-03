from __future__ import annotations

from typing import Any


class FakeDeviceRegistry:
    """In-memory DeviceRegistryPort for unit tests.

    MagicMock drivers are useful for asserting service interactions, but the
    production GlobalDeviceManager intentionally accepts only BaseDevice
    instances. This registry keeps those concerns separate.
    """

    def __init__(self) -> None:
        self._devices: dict[str, Any] = {}

    def register_device(self, name: str, device: Any) -> None:
        self._devices[name] = device

    def drop_device(self, name: str, ignore_error: bool = False) -> None:
        if name not in self._devices:
            if ignore_error:
                return
            raise ValueError(f"Device {name} not found")
        del self._devices[name]

    def get_device(self, name: str) -> Any:
        if name not in self._devices:
            raise ValueError(f"Device {name} not found")
        return self._devices[name]

    def get_all_devices(self) -> dict[str, Any]:
        return dict(self._devices)

    def get_info(self, name: str) -> Any:
        return self.get_device(name).get_info()
