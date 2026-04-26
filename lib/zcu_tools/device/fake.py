from __future__ import annotations

from typing_extensions import Literal

from .base import BaseDevice, BaseDeviceInfo


class FakeDeviceInfo(BaseDeviceInfo):
    type: Literal["fake"] = "fake"
    output: Literal["on", "off"] = "off"
    value: float


class FakeDevice(BaseDevice):
    info_model = FakeDeviceInfo

    def __init__(self) -> None:
        self.address = "none"
        self.output: Literal["on", "off"] = "off"
        self.value = 0.0

    def get_output(self) -> Literal["on", "off"]:
        return self.output

    def set_output(self, status: Literal["on", "off"]) -> None:
        self.output = status

    # Turn on output
    def output_on(self) -> None:
        self.set_output("on")

    # Turn off output
    def output_off(self) -> None:
        self.set_output("off")

    # ==========================================================================#

    def get_value(self) -> float:
        return self.value

    def set_value(self, value: float) -> float:
        self.value = value
        return self.value

    # ==========================================================================#

    def _setup(self, cfg: FakeDeviceInfo, /, progress: bool = True) -> None:
        self.set_output(cfg.output)
        self.set_value(cfg.value)

    def get_info(self) -> FakeDeviceInfo:
        return FakeDeviceInfo(
            address=self.address,
            output=self.output,
            value=self.value,
        )
