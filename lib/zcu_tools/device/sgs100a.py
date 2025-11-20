from typing import Literal

from .base import BaseDevice, DeviceInfo

STATUS_MAP = {"on": "1", "off": "0"}
STATUS_MAP_INV = {"1": "on", "0": "off"}


class RohdeSchwarzSGS100AInfo(DeviceInfo):
    output: Literal["on", "off"]
    IQ: Literal["on", "off"]
    freq_Hz: float
    power_dBm: float


class RohdeSchwarzSGS100A(BaseDevice[RohdeSchwarzSGS100AInfo]):
    def get_output(self) -> Literal["on", "off"]:
        return STATUS_MAP_INV[self.query(":OUTPut?")]  # type: ignore

    def set_output(self, status: Literal["on", "off"]) -> None:
        self.write(f":OUTPut {STATUS_MAP[status]}")

    # Turn on output
    def output_on(self) -> None:
        self.set_output("on")

    # Turn off output
    def output_off(self) -> None:
        self.set_output("off")

    # ==========================================================================#

    def get_IQ_state(self) -> Literal["on", "off"]:
        return STATUS_MAP_INV[self.query(":IQ:STAT?")]  # type: ignore

    def set_IQ_state(self, state: Literal["on", "off"]) -> None:
        self.write(f":IQ:STAT {STATUS_MAP[state]}")

    def IQ_on(self) -> None:
        self.set_IQ_state("on")

    def IQ_off(self) -> None:
        self.set_IQ_state("off")

    # ==========================================================================#

    def get_frequency(self) -> float:
        return float(self.query("SOUR:FREQ?"))

    def set_frequency(self, freq_Hz: float) -> float:
        if not (1e6 <= freq_Hz <= 20e9):
            raise ValueError(
                f"Frequency {freq_Hz} Hz is outside expected range (1e6, 20e9)"
            )
        self.write(f":SOUR:FREQ {freq_Hz:.2f}")
        return self.get_frequency()

    # ==========================================================================#

    def get_power(self) -> float:
        return float(self.query("SOUR:POW:POW?"))

    def set_power(self, power_dBm: float) -> float:
        if not (-120 <= power_dBm <= 25):
            raise ValueError(
                f"Power {power_dBm} dBm is outside expected range (-120, 25)"
            )
        self.write(f":SOUR:POW:POW {power_dBm:.2f}")
        return self.get_power()

    # ==========================================================================#

    def _setup(self, cfg: RohdeSchwarzSGS100AInfo, *, progress: bool = True) -> None:
        self.set_output(cfg["output"])
        self.set_IQ_state(cfg["IQ"])
        self.set_frequency(cfg["freq_Hz"])
        self.set_power(cfg["power_dBm"])

    def get_info(self) -> RohdeSchwarzSGS100AInfo:
        return RohdeSchwarzSGS100AInfo(
            {
                "type": self.__class__.__name__,
                "address": self.address,
                "output": self.get_output(),
                "IQ": self.get_IQ_state(),
                "freq_Hz": self.get_frequency(),
                "power_dBm": self.get_power(),
            }
        )
