import time
from typing import Any, Dict, Literal, Optional

import numpy as np
from tqdm.auto import tqdm

from .base import BaseDevice, ResourceManager, DeviceInfo

DEFAULT_RAMPSTEP = 1e-6  # increment step when setting voltage/current
DEFAULT_RAMPINTERVAL = 0.01  # dwell time for each voltage step # Default MATLAB is 0.01, CANNOT be lower than 0.001 otherwise fridge heats up


class YOKOGS200(BaseDevice):
    # Initializes session for device.
    # VISAaddress: address of device, rm: VISA resource manager
    def __init__(self, VISAaddress: str, rm: ResourceManager) -> None:
        super().__init__(VISAaddress, rm)

        self._rampstep = DEFAULT_RAMPSTEP
        self._rampinterval = DEFAULT_RAMPINTERVAL

    # ==========================================================================#

    # Turn on output
    def output_on(self) -> None:
        self.session.write(":OUTPut 1")

    # Turn off output
    def output_off(self) -> None:
        self.session.write(":OUTPut 0")

    # ==========================================================================#

    def _set_voltage_direct(self, voltage: float) -> None:
        self.session.write(":SOURce:LEVel:AUTO %.8f" % voltage)
        time.sleep(self._rampinterval)

    def _set_voltage_smart(self, voltage: float, progress: bool = False) -> None:
        # sweep to the target value step by step
        current_voltage = self.get_voltage()
        if current_voltage == voltage:
            return

        if progress:
            dist = abs(current_voltage - voltage)
            pbar = tqdm(total=round(dist, 2), unit="V", leave=False)

        step = 10 * self._rampstep
        steps = max(1, round(abs(voltage - current_voltage) / step))
        voltages = np.linspace(current_voltage, voltage, num=steps + 1, endpoint=True)
        for tempvolt in voltages:
            self._set_voltage_direct(tempvolt)

            if progress:
                cur_dist = abs(tempvolt - voltage)
                pbar.update(round(dist - cur_dist, 2) - pbar.n)

        if progress:
            pbar.close()

    # Ramp up the voltage (volts) in increments of _rampstep, waiting _rampinterval
    # between each increment.
    def set_voltage(self, voltage: float, progress: bool = True) -> float:
        mode = self.get_mode()
        if mode != "voltage":
            raise RuntimeError(
                f"One can only set voltage when the device is in voltage mode. but it is in {mode} mode."
            )

        self.output_on()
        self._set_voltage_smart(voltage, progress=progress)

        return self.get_voltage()

    def _set_current_direct(self, current: float) -> None:
        self.session.write(":SOURce:LEVel:AUTO %.8f" % current)
        time.sleep(self._rampinterval)

    def _set_current_smart(self, current: float, progress: bool = False) -> None:
        # sweep to the target value step by step
        current_current = self.get_current()
        if current_current == current:
            return

        if progress:
            dist = 1e3 * abs(current_current - current)
            pbar = tqdm(total=round(dist, 2), unit="mA", leave=False)

        step = 10 * self._rampstep
        steps = max(1, round(abs(current - current_current) / step))
        currents = np.linspace(current_current, current, num=steps + 1, endpoint=True)
        for tempcurrent in currents:
            self._set_current_direct(tempcurrent)

            if progress:
                cur_dist = 1e3 * abs(tempcurrent - current)
                pbar.update(round(dist - cur_dist, 2) - pbar.n)

        if progress:
            pbar.close()

    # Ramp up the current (amps) in increments of _rampstep, waiting _rampinterval
    # between each increment.
    def set_current(self, current: float, progress: bool = True) -> float:
        mode = self.get_mode()
        if mode != "current":
            raise RuntimeError(
                f"One can only set current when the device is in current mode. but it is in {mode} mode."
            )

        self.output_on()
        self._set_current_smart(current, progress=progress)

        return self.get_current()

    # Set to either current or voltage mode.
    def set_mode(
        self,
        mode: Literal["voltage", "current"],
        force: bool = False,
        rampstep: Optional[float] = None,
    ) -> None:
        cur_mode = self.get_mode()

        if cur_mode != mode:
            if cur_mode == "voltage":
                value = self.get_voltage()
            else:
                value = self.get_current()
            if value != 0.0 and not force:
                raise RuntimeError(
                    "Try to change mode while value is not zero. Please set value to zero before changing mode"
                    "Or set force=True to override, make sure you know what you are doing"
                )

        self.session.write(":SOURce:FUNCtion %s" % mode)

        # update rampstep
        if rampstep is not None:
            self._rampstep = rampstep

    # ==========================================================================#

    # Returns the voltage in volts as a float
    def get_voltage(self) -> float:
        mode = self.get_mode()
        if mode != "voltage":
            raise RuntimeError(
                f"One can only get voltage when the device is in voltage mode. but it is in {mode} mode."
            )

        self.session.write(":SOURce:FUNCtion VOLTage")
        self.session.write(":SOURce:LEVel?")
        result = self.session.read()
        return float(result.rstrip("\n"))

    # Returns the current in amps as a float
    def get_current(self) -> float:
        mode = self.get_mode()
        if mode != "current":
            raise RuntimeError(
                f"One can only get current when the device is in current mode. but it is in {mode} mode."
            )

        self.session.write(":SOURce:FUNCtion CURRent")
        self.session.write(":SOURce:LEVel?")
        result = self.session.read()
        return float(result.rstrip("\n"))

    # Returns the mode (voltage or current)
    def get_mode(self) -> Literal["voltage", "current"]:
        self.session.write(":SOURce:FUNCtion?")
        result = self.session.read()
        result = result.rstrip("\n")
        if result == "VOLT":
            return "voltage"
        else:
            return "current"

    # ==========================================================================#

    def _setup(self, cfg: Dict[str, Any], *, progress: bool = True) -> None:
        cur_mode = self.get_mode()

        if cfg["mode"] != cur_mode:
            raise RuntimeError(
                f"Current mode {cur_mode} in device {self.VISAaddress}, but cfg requires {cfg['mode']}"
                "YOKOGS200 does not support implicit setup mode to prevent sudden current/voltage change"
                "Please change the device mode manually before calling setup"
                "Remember to turn value to zero before changing mode"
            )

        CHECK_CURRENT_LIMIT = 20e-3
        CHECK_VOLTAGE_LIMIT = 20

        value = cfg["value"]
        if cur_mode == "current":
            # protect large current
            if abs(value) > CHECK_CURRENT_LIMIT:
                if cfg.get("enable_large_value", False):
                    raise RuntimeError(
                        f"Try to set current to over {CHECK_CURRENT_LIMIT}A, are you sure you want to do this?"
                        "If you are sure, set enable_large_value=True to override"
                    )

            self.set_current(value, progress=progress)
        elif cur_mode == "voltage":
            # protect large voltage
            if abs(value) > CHECK_VOLTAGE_LIMIT:
                if cfg.get("enable_large_value", False):
                    raise RuntimeError(
                        f"Try to set voltage to over {CHECK_VOLTAGE_LIMIT}V, are you sure you want to do this?"
                        "If you are sure, set enable_large_value=True to override"
                    )

            self.set_voltage(value, progress=progress)
        else:
            raise RuntimeError(f"Unknown mode {cur_mode} in device {self.VISAaddress}")

    def get_info(self) -> DeviceInfo:
        mode = self.get_mode()
        if mode == "voltage":
            value = self.get_voltage()
        else:
            value = self.get_current()

        return {
            "type": self.__class__.__name__,
            "VISAaddress": self.VISAaddress,
            "mode": mode,
            "value": value,
        }
