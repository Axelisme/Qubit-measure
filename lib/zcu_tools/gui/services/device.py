from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.device_manager import DeviceManager
    from zcu_tools.gui.state import State


class DeviceService:
    """Encapsulates device registration and manipulation."""

    def __init__(self, state: "State", device_manager: "DeviceManager") -> None:
        self._state = state
        self._dm = device_manager

    def register_device(self, name: str, device: Any) -> None:
        logger.info("register_device: name=%r type=%s", name, type(device).__name__)
        if self._state.is_run_active():
            raise RuntimeError("Cannot register device while a run is active")
        self._dm.register_device(name, device)

    def drop_device(self, name: str) -> None:
        logger.info("drop_device: name=%r", name)
        if self._state.is_run_active():
            raise RuntimeError("Cannot drop device while a run is active")
        self._dm.drop_device(name)

    def list_devices(self) -> dict[str, str]:
        return self._dm.list_devices()

    def set_device_value(self, name: str, value: Any) -> Any:
        if self._state.is_run_active():
            raise RuntimeError("Cannot set device value while a run is active")
        return self._dm.set_device_value(name, value)

    def get_device_value(self, name: str) -> Any:
        return self._dm.get_device_value(name)

    def get_device_info(self, name: str) -> Any:
        return self._dm.get_device_info(name)

    def setup_device(
        self, name: str, info: Any, pbar_factory: Optional[Any] = None
    ) -> Any:
        if self._state.is_run_active():
            raise RuntimeError("Cannot setup device while a run is active")
        return self._dm.setup_device(name, info, pbar_factory)
