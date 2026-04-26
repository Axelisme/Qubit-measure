from __future__ import annotations

import threading
import warnings

from typing_extensions import Any, ClassVar, Mapping, Union, TypeAlias, cast

from .base import BaseDevice, BaseDeviceInfo
from .yoko import YOKOGS200, YOKOGS200Info
from .sgs100a import RohdeSchwarzSGS100A, RohdeSchwarzSGS100AInfo
from .fake import FakeDevice, FakeDeviceInfo

DeviceInfo: TypeAlias = Union[YOKOGS200Info, RohdeSchwarzSGS100AInfo, FakeDeviceInfo]


class GlobalDeviceManager:
    _devices: ClassVar[dict[str, BaseDevice]] = {}
    _lock: ClassVar[threading.RLock] = threading.RLock()

    @classmethod
    def register_device(cls, name: str, device: Any) -> None:
        with cls._lock:
            if name in cls._devices:
                warnings.warn(f"Device {name} already registered, overwriting")
            cls._devices[name] = device

    @classmethod
    def drop_device(cls, name: str, ignore_error: bool = False) -> None:
        with cls._lock:
            if name not in cls._devices:
                if ignore_error:
                    return
                raise ValueError(f"Device {name} not found")
            del cls._devices[name]

    @classmethod
    def get_device(cls, name: str) -> BaseDevice:
        with cls._lock:
            if name not in cls._devices:
                raise ValueError(f"Device {name} not found")
            return cls._devices[name]

    @classmethod
    def get_all_devices(cls) -> dict[str, BaseDevice]:
        with cls._lock:
            return dict(cls._devices)

    @classmethod
    def setup_devices(
        cls, dev_cfg: Mapping[str, DeviceInfo], *, progress: bool = True
    ) -> None:
        with cls._lock:
            for name, cfg in dev_cfg.items():
                cls._devices[name].setup(cfg, progress=progress)

    @classmethod
    def get_all_info(cls) -> dict[str, DeviceInfo]:
        with cls._lock:
            return {
                name: cast(DeviceInfo, device.get_info())
                for name, device in cls._devices.items()
            }


__all__ = [
    # base
    "BaseDevice",
    "BaseDeviceInfo",
    # devices
    "YOKOGS200",
    "YOKOGS200Info",
    "RohdeSchwarzSGS100A",
    "RohdeSchwarzSGS100AInfo",
    "FakeDevice",
    "FakeDeviceInfo",
    # manager
    "GlobalDeviceManager",
    "DeviceInfo",
]
