from typing import Any, Dict


class GlobalDeviceManager:
    devices: Dict[str, Any] = {}

    @classmethod
    def register_device(cls, name: str, device: Any) -> None:
        cls.devices[name] = device

    @classmethod
    def get_device(cls, name: str) -> Any:
        if name not in cls.devices:
            raise ValueError(f"Device {name} not found")
        return cls.devices[name]

    @classmethod
    def get_all_devices(cls) -> Dict[str, Any]:
        return cls.devices
