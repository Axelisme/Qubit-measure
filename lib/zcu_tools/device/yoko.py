import sys
import time
from typing import Literal, Optional

import numpy as np
import pyvisa as visa
from tqdm.auto import tqdm

DEFAULT_RAMPSTEP = 1e-6  # increment step when setting voltage/current
DEFAULT_RAMPINTERVAL = 0.01  # dwell time for each voltage step # Default MATLAB is 0.01, CANNOT be lower than 0.001 otherwise fridge heats up


class YOKOGS200:
    # Initializes session for device.
    # VISAaddress: address of device, rm: VISA resource manager
    def __init__(self, VISAaddress: str, rm: visa.ResourceManager) -> None:
        self.VISAaddress = VISAaddress
        self._rampstep = DEFAULT_RAMPSTEP
        self._rampinterval = DEFAULT_RAMPINTERVAL

        try:
            self.session = rm.open_resource(VISAaddress)
        except visa.Error:
            sys.stderr.write("Couldn't connect to '%s', exiting now..." % VISAaddress)
            sys.exit()

        self.mode = self.get_mode()

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
        self, mode: Literal["voltage", "current"], rampstep: Optional[float] = None
    ) -> None:
        if not (mode == "voltage" or mode == "current"):
            sys.stderr.write("Unknown output mode %s." % mode)
            return
        self.session.write(":SOURce:FUNCtion %s" % mode)

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
