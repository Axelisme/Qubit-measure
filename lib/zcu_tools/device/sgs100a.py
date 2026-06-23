from __future__ import annotations

import threading
from typing import Literal

from .base import BaseDevice, BaseDeviceInfo

STATUS_MAP = {"on": "1", "off": "0"}


STATUS_MAP_INV = {v: k for k, v in STATUS_MAP.items()}


class RohdeSchwarzSGS100AInfo(BaseDeviceInfo):
    type: Literal["RohdeSchwarzSGS100A"] = "RohdeSchwarzSGS100A"
    output: Literal["on", "off"] = "off"
    IQ: Literal["on", "off"] = "off"
    freq_Hz: float = 1e9
    power_dBm: float = -120.0

    def set_freq(self, freq_Hz: float) -> None:
        self.freq_Hz = freq_Hz

    def set_power(self, power_dBm: float) -> None:
        self.power_dBm = power_dBm

    def set_output(self, output: Literal["on", "off"]) -> None:
        self.output = output


class RohdeSchwarzSGS100A(BaseDevice[RohdeSchwarzSGS100AInfo]):
    info_model = RohdeSchwarzSGS100AInfo

    def get_output(self) -> Literal["on", "off"]:
        with self._lock:
            return STATUS_MAP_INV[self.query(":OUTPut?")]  # type: ignore

    def set_output(self, status: Literal["on", "off"]) -> None:
        with self._lock:
            self.write(f":OUTPut {STATUS_MAP[status]}")

    # Turn on output
    def output_on(self) -> None:
        with self._lock:
            self.set_output("on")

    # Turn off output
    def output_off(self) -> None:
        with self._lock:
            self.set_output("off")

    # ==========================================================================#

    def get_IQ_state(self) -> Literal["on", "off"]:
        with self._lock:
            return STATUS_MAP_INV[self.query(":IQ:STAT?")]  # type: ignore

    def set_IQ_state(self, state: Literal["on", "off"]) -> None:
        with self._lock:
            self.write(f":IQ:STAT {STATUS_MAP[state]}")

    def IQ_on(self) -> None:
        with self._lock:
            self.set_IQ_state("on")

    def IQ_off(self) -> None:
        with self._lock:
            self.set_IQ_state("off")

    # ==========================================================================#

    def get_frequency(self) -> float:
        with self._lock:
            return float(self.query("SOUR:FREQ?"))

    def set_frequency(self, freq_Hz: float) -> float:
        if not (1e6 <= freq_Hz <= 20e9):
            raise ValueError(
                f"Frequency {freq_Hz} Hz is outside expected range (1e6, 20e9)"
            )
        # write+read must stay in one critical section so a concurrent setter can
        # not change the level between the write and the confirming read-back.
        with self._lock:
            self.write(f":SOUR:FREQ {freq_Hz:.2f}")
            return self.get_frequency()

    # ==========================================================================#

    def get_power(self) -> float:
        with self._lock:
            return float(self.query("SOUR:POW:POW?"))

    def set_power(self, power_dBm: float) -> float:
        if not (-120 <= power_dBm <= 25):
            raise ValueError(
                f"Power {power_dBm} dBm is outside expected range (-120, 25)"
            )
        # write+read must stay in one critical section (see set_frequency).
        with self._lock:
            self.write(f":SOUR:POW:POW {power_dBm:.2f}")
            return self.get_power()

    # ==========================================================================#

    def _setup(
        self,
        cfg,
        *,
        progress: bool = True,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.set_output(cfg.output)
        self.set_IQ_state(cfg.IQ)
        self.set_frequency(cfg.freq_Hz)
        self.set_power(cfg.power_dBm)

    def get_info(self) -> RohdeSchwarzSGS100AInfo:
        with self._lock:
            return RohdeSchwarzSGS100AInfo(
                address=self.address,
                output=self.get_output(),
                IQ=self.get_IQ_state(),
                freq_Hz=self.get_frequency(),
                power_dBm=self.get_power(),
            )
