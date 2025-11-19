from typing import Any, Dict, Mapping

from .base import BaseDevice, DeviceInfo


class GlobalDeviceManager:
    devices: Dict[str, BaseDevice] = {}

    @classmethod
    def register_device(cls, name: str, device: Any) -> None:
        cls.devices[name] = device

    @classmethod
    def drop_device(cls, name: str, ignore_error: bool = False) -> None:
        if name not in cls.get_all_devices() and not ignore_error:
            raise ValueError(f"Device {name} not found")
        del cls.devices[name]

    @classmethod
    def get_device(cls, name: str) -> BaseDevice:
        if name not in cls.devices:
            raise ValueError(f"Device {name} not found")
        return cls.devices[name]

    @classmethod
    def get_all_devices(cls) -> Dict[str, BaseDevice]:
        return cls.devices

    @classmethod
    def setup_devices(
        cls, dev_cfg: Mapping[str, DeviceInfo], *, progress: bool = True
    ) -> None:
        for name, cfg in dev_cfg.items():
            cls.get_device(name).setup(cfg, progress=progress)

    @classmethod
    def get_all_info(cls) -> Dict[str, DeviceInfo]:
        return {
            name: device.get_info() for name, device in cls.get_all_devices().items()
        }
