from typing import Any, Dict, Literal

from .base import BaseDevice, DeviceInfo


# Rohde&Schwarz RF Source
class RFSource(BaseDevice):
    # ==========================================================================#

    # Turn on output
    def output_on(self) -> None:
        self.session.write("OUTP 1")

    # Turn off output
    def output_off(self) -> None:
        self.session.write("OUTP 0")

    # ==========================================================================#

    # set frequency
    def set_frequency(self, freq_Hz: float) -> None:
        self.session.write(f":SOUR:FREQ {freq_Hz:.8f}Hz")

    # get frequency
    def get_frequency(self) -> float:
        self.session.write(":SOUR:FREQ?")
        result = self.session.read()
        return float(result.rstrip("\n"))

    # ==========================================================================#

    # set power
    def set_power(self, power_dBm: float) -> None:
        self.session.write(f":SOUR:POW:POW {power_dBm:.8f}")

    # get power
    def get_power(self) -> float:
        self.session.write("SOUR:POW:POW?")
        result = self.session.read()
        return float(result.rstrip("\n"))

    # ==========================================================================#

    # set ALC
    def set_alc(self, alc: Literal["ON", "OFF", "AUTO"]) -> None:
        self.session.write(f":SOUR:POW:ALC {alc}")

    # get ALC
    def get_alc(self) -> Literal["ON", "OFF", "AUTO"]:
        self.session.write("SOUR:POW:ALC?")
        result = self.session.read()
        return result.rstrip("\n")

    def _setup(self, cfg: Dict[str, Any], *, progress: bool = True) -> None:
        self.output_on()

        self.set_frequency(cfg["freq"])
        self.set_power(cfg["power"])
        self.set_alc(cfg["alc"])

    def get_info(self) -> DeviceInfo:
        return {
            "type": self.__class__.__name__,
            "address": self.VISAaddress,
            "freq": self.get_frequency(),
            "power": self.get_power(),
            "alc": self.get_alc(),
        }
