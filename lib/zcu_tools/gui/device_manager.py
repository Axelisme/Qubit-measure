from __future__ import annotations

from typing import Any

from zcu_tools.device import GlobalDeviceManager


class DeviceManager:
    """Thin wrapper around GlobalDeviceManager for GUI use."""

    def register_device(self, name: str, device: Any) -> None:
        """Register an already-constructed device instance."""
        GlobalDeviceManager.register_device(name, device)

    def drop_device(self, name: str) -> None:
        GlobalDeviceManager.drop_device(name)

    def list_devices(self) -> dict[str, str]:
        """Return {name: device_type_string} for display."""
        devices = GlobalDeviceManager.get_all_devices()
        return {name: type(dev).__name__ for name, dev in devices.items()}

    def get_device_value(self, name: str) -> Any:
        """Read the current value from a device via get_value()."""
        dev = GlobalDeviceManager.get_device(name)
        return dev.get_value()  # type: ignore[attr-defined]

    def set_device_value(self, name: str, value: Any) -> Any:
        """Set a device value via set_value(); return the actual value applied."""
        dev = GlobalDeviceManager.get_device(name)
        return dev.set_value(value)  # type: ignore[attr-defined]

    def get_device_info(self, name: str) -> Any:
        """Return the DeviceInfo object for a registered device."""
        dev = GlobalDeviceManager.get_device(name)
        return dev.get_info()  # type: ignore[attr-defined]

    def setup_device(self, name: str, info: Any) -> None:
        """Apply a DeviceInfo to the device via dev.setup(info)."""
        dev = GlobalDeviceManager.get_device(name)
        dev.setup(info)  # type: ignore[attr-defined]

    def get_all_info(self) -> dict[str, Any]:
        """Delegate to GlobalDeviceManager.get_all_info()."""
        return GlobalDeviceManager.get_all_info()
