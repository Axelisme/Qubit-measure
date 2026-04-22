from __future__ import annotations

from typeguard import check_type
from typing_extensions import Any, Literal, Optional

from .base import BaseDevice, DeviceInfo


class FakeDeviceInfo(DeviceInfo, closed=True):
    type: Literal["FakeDevice"]
    output: Literal["on", "off"]
    value: float


class FakeDevice(BaseDevice):
    def __init__(self, address: str = "FAKE::INSTR", rm: Optional[Any] = None) -> None:
        # Fake device can run without a real VISA session.
        if rm is not None:
            super().__init__(address, rm)
        else:
            self.address = address
            self.session = None
        self.output: Literal["on", "off"] = "off"
        self.value: float = 0.0

    def _setup(self, cfg: FakeDeviceInfo, /, progress: bool = True) -> None:
        cfg = check_type(cfg, FakeDeviceInfo)  # runtime check

        self.output = cfg["output"]
        self.value = cfg["value"]

    def set_field(self, field: str, value: Any) -> None:
        if field == "output":
            if value not in {"on", "off"}:
                raise ValueError("output must be 'on' or 'off'")
            self.output = value
            return
        if field == "value":
            self.value = float(value)
            return
        raise KeyError(f"Unknown field: {field}")

    def get_info(self) -> FakeDeviceInfo:
        return FakeDeviceInfo(
            {
                "type": "FakeDevice",
                "address": self.address,
                "output": self.output,
                "value": self.value,
            }
        )
