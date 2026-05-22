from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from zcu_tools.gui.device_manager import DeviceManager
    from zcu_tools.gui.state import State


class DeviceService:
    """Encapsulates device registration and manipulation."""

    def __init__(self, state: "State", device_manager: "DeviceManager") -> None:
        self._state = state
        self._dm = device_manager

    def register_device(self, name: str, device: Any) -> None:
        if self._state.is_running:
            raise RuntimeError("Cannot register device while a run is active")
        self._dm.register_device(name, device)

    def drop_device(self, name: str) -> None:
        if self._state.is_running:
            raise RuntimeError("Cannot drop device while a run is active")
        self._dm.drop_device(name)

    def list_devices(self) -> dict[str, str]:
        return self._dm.list_devices()

    def set_device_value(self, name: str, value: Any) -> Any:
        if self._state.is_running:
            raise RuntimeError("Cannot set device value while a run is active")
        return self._dm.set_device_value(name, value)

    def get_device_value(self, name: str) -> Any:
        return self._dm.get_device_value(name)

    def get_device_info(self, name: str) -> Any:
        return self._dm.get_device_info(name)

    def setup_device(
        self, name: str, info: Any, pbar_factory: Optional[Any] = None
    ) -> Any:
        if self._state.is_running:
            raise RuntimeError("Cannot setup device while a run is active")
        return self._dm.setup_device(name, info, pbar_factory)
