from typing import Any


class GlobalDeviceManager:
    devices = {}

    def register_device(self, name: str, device: Any) -> None:
        self.devices[name] = device

    def get_device(self, name: str) -> Any:
        if name not in self.devices:
            raise ValueError(f"Device {name} not found")
        return self.devices[name]

    def get_all_devices(self) -> dict[str, Any]:
        return self.devices
