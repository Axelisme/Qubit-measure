import time
import warnings
from typing import Literal, Optional

import numpy as np
from tqdm.auto import tqdm

from .base import BaseDevice, DeviceInfo, ResourceManager

STATUS_MAP = {"on": "1", "off": "0"}
STATUS_MAP_INV = {"1": "on", "0": "off"}
MODE_MAPS = {"voltage": "VOLT", "current": "CURR"}
MODE_MAPS_INV = {"VOLT": "voltage", "CURR": "current"}


class YOKOGS200Info(DeviceInfo):
    output: Literal["on", "off"]
    mode: Literal["voltage", "current"]
    value: float


class YOKOGS200(BaseDevice[YOKOGS200Info]):
    # Initializes session for device.
    # address: address of device, rm: VISA resource manager
    def __init__(self, address: str, rm: ResourceManager) -> None:
        super().__init__(address, rm)

        mode = self.get_mode()

        if mode == "voltage":
            self._rampstep = 1e-4
        elif mode == "current":
            self._rampstep = 1e-8
        else:
            raise ValueError(f"Unknown mode {mode}")

        self._rampinterval = 0.01

    # ==========================================================================#

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

    def _check_voltage(self, voltage: float) -> None:
        CHECK_VOLTAGE_LIMIT = 7
        if abs(voltage) > CHECK_VOLTAGE_LIMIT:
            raise RuntimeError(
                f"Try to set voltage to over {CHECK_VOLTAGE_LIMIT}V, are you sure you want to do this?"
            )

    def _set_voltage_direct(self, voltage: float) -> None:
        self._check_voltage(voltage)
        self.write(f":SOURce:LEVel:AUTO {voltage:.8f}")
        time.sleep(self._rampinterval)

    def _set_voltage_smart(self, voltage: float, progress: bool = False) -> None:
        # sweep to the target value step by step
        current_voltage = self.get_voltage()
        if current_voltage == voltage:
            return

        self._check_voltage(voltage)

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

    def set_voltage(self, voltage: float, progress: bool = True) -> float:
        mode = self.get_mode()
        if mode != "voltage":
            raise RuntimeError(
                f"One can only set voltage when the device is in voltage mode. but it is in {mode} mode."
            )

        self.output_on()
        self._set_voltage_smart(voltage, progress=progress)

        return self.get_voltage()

    def _check_current(self, current: float) -> None:
        CHECK_CURRENT_LIMIT = 7e-3
        if abs(current) > CHECK_CURRENT_LIMIT:
            raise RuntimeError(
                f"Try to set current to over {CHECK_CURRENT_LIMIT}A, are you sure you want to do this?"
            )

    def _set_current_direct(self, current: float) -> None:
        self._check_current(current)
        self.write(f":SOURce:LEVel:AUTO {current:.8f}")
        time.sleep(self._rampinterval)

    def _set_current_smart(self, current: float, progress: bool = False) -> None:
        # sweep to the target value step by step
        current_current = self.get_current()
        if current_current == current:
            return

        self._check_current(current)

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
                    "Try to change mode while value is not zero. Please set value to zero before changing mode, "
                    "Or set force=True to override, make sure you know what you are doing"
                )

        self.write(f":SOURce:FUNCtion {MODE_MAPS[mode]}")

        # update rampstep
        if rampstep is not None:
            self._rampstep = rampstep

    # ==========================================================================#

    def _get_level(self) -> float:
        return float(self.query(":SOURce:LEVel?"))

    # Returns the voltage in volts as a float
    def get_voltage(self) -> float:
        mode = self.get_mode()
        if mode != "voltage":
            raise RuntimeError(
                f"One can only get voltage when the device is in voltage mode. but it is in {mode} mode."
            )

        self.write(f"SOURce:FUNCtion {MODE_MAPS['voltage']}")
        return self._get_level()

    # Returns the current in amps as a float
    def get_current(self) -> float:
        mode = self.get_mode()
        if mode != "current":
            raise RuntimeError(
                f"One can only get current when the device is in current mode. but it is in {mode} mode."
            )

        self.write(f"SOURce:FUNCtion {MODE_MAPS['current']}")
        return self._get_level()

    # Returns the mode (voltage or current)
    def get_mode(self) -> Literal["voltage", "current"]:
        return MODE_MAPS_INV[self.query(":SOURce:FUNCtion?")]  # type: ignore

    # ==========================================================================#

    def _setup(self, cfg: YOKOGS200Info, *, progress: bool = True) -> None:
        if self.get_output() != "on" and cfg["output"] == "on":
            warnings.warn("YOKOGS200 output is off, did you forget to turn it on?")

        cur_mode = self.get_mode()

        if cfg["mode"] != cur_mode:
            raise RuntimeError(
                f"Current mode: {cur_mode} in device {self.address}, but cfg requires: {cfg['mode']} mode, "
                "YOKOGS200 does not support implicit setup mode to prevent sudden current/voltage change, "
                "Please change the device mode manually before calling setup, "
                "Remember to turn value to zero before changing mode"
            )

        value = cfg["value"]
        if cur_mode == "current":
            self.set_current(value, progress=progress)
        elif cur_mode == "voltage":
            self.set_voltage(value, progress=progress)
        else:
            raise RuntimeError(f"Unknown mode {cur_mode} in device {self.address}")

    def get_info(self) -> YOKOGS200Info:
        return YOKOGS200Info(
            {
                "type": self.__class__.__name__,
                "address": self.address,
                "output": self.get_output(),
                "mode": self.get_mode(),
                "value": self._get_level(),
            }
        )
