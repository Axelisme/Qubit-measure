from __future__ import annotations

import warnings

from typing_extensions import Any, Mapping

from .base import BaseDevice, DeviceInfo


class GlobalDeviceManager:
    _devices: dict[str, BaseDevice] = {}

    @classmethod
    def register_device(cls, name: str, device: Any) -> None:
        if name in cls._devices:
            warnings.warn(f"Device {name} already registered, overwriting")
        cls._devices[name] = device

    @classmethod
    def drop_device(cls, name: str, ignore_error: bool = False) -> None:
        if name not in cls._devices and not ignore_error:
            raise ValueError(f"Device {name} not found")
        del cls._devices[name]

    @classmethod
    def get_device(cls, name: str) -> BaseDevice:
        if name not in cls._devices:
            raise ValueError(f"Device {name} not found")
        return cls._devices[name]

    @classmethod
    def get_all_devices(cls) -> dict[str, BaseDevice]:
        return cls._devices

    @classmethod
    def setup_devices(
        cls, dev_cfg: Mapping[str, DeviceInfo], *, progress: bool = True
    ) -> None:
        for name, cfg in dev_cfg.items():
            cls._devices[name].setup(cfg, progress=progress)

    @classmethod
    def get_all_info(cls) -> dict[str, DeviceInfo]:
        return {name: device.get_info() for name, device in cls._devices.items()}
