from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Literal

from .base import BaseDevice, BaseDeviceInfo, device_operation

if TYPE_CHECKING:
    from pyvisa import ResourceManager


STATUS_MAP = {"on": "1", "off": "0"}


STATUS_MAP_INV = {v: k for k, v in STATUS_MAP.items()}


class AnritsuMG3692Info(BaseDeviceInfo):
    type: Literal["AnritsuMG3692"] = "AnritsuMG3692"
    output: Literal["on", "off"] = "off"
    freq_Hz: float = 1e9
    power_dBm: float = -120.0

    def set_frequency(self, freq_Hz: float) -> None:
        self.freq_Hz = freq_Hz

    def set_power(self, power_dBm: float) -> None:
        self.power_dBm = power_dBm

    def set_output(self, output: Literal["on", "off"]) -> None:
        self.output = output


class AnritsuMG3692(BaseDevice[AnritsuMG3692Info]):
    info_model = AnritsuMG3692Info

    def __init__(self, address: str, rm: ResourceManager) -> None:
        super().__init__(address, rm)

    def get_output(self) -> Literal["on", "off"]:
        return STATUS_MAP_INV[self.query(":OUTP?")]  # type: ignore

    @device_operation
    def set_output(self, status: Literal["on", "off"]) -> None:
        self.write(f":OUTP {STATUS_MAP[status]}")

    # Turn on output
    @device_operation
    def output_on(self) -> None:
        self.set_output("on")

    # Turn off output
    @device_operation
    def output_off(self) -> None:
        self.set_output("off")

    # ==========================================================================#

    def get_frequency(self) -> float:
        return float(self.query("FREQ?"))

    @device_operation
    def set_frequency(self, freq_Hz: float) -> float:
        if not (2e9 <= freq_Hz <= 20e9):
            raise ValueError(
                f"Frequency {freq_Hz} Hz is outside expected range (2e9, 20e9)"
            )
        self.write(f"FREQ {freq_Hz:.2f} Hz")
        return self.get_frequency()

    # ==========================================================================#

    def get_power(self) -> float:
        return float(self.query("POW?"))

    @device_operation
    def set_power(self, power_dBm: float) -> float:
        if not (-120 <= power_dBm <= 27):
            raise ValueError(
                f"Power {power_dBm} dBm is outside expected range (-120, 27)"
            )
        self.write(f"POW {power_dBm:.2f} DBM")
        return self.get_power()

    # ==========================================================================#

    def _setup(
        self,
        cfg: AnritsuMG3692Info,
        *,
        progress: bool = True,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.set_output(cfg.output)
        self.set_frequency(cfg.freq_Hz)
        self.set_power(cfg.power_dBm)

    def get_info(self) -> AnritsuMG3692Info:
        return AnritsuMG3692Info(
            address=self.address,
            output=self.get_output(),
            freq_Hz=self.get_frequency(),
            power_dBm=self.get_power(),
        )
