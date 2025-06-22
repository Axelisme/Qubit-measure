import sys
import time
from typing import Literal

import numpy as np
import pyvisa as visa
from tqdm import tqdm

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

    # ==========================================================================#

    # Turn on output
    def OutputOn(self) -> None:
        self.session.write("OUTPut 1")

    # Turn off output
    def OutputOff(self) -> None:
        self.session.write("OUTPut 0")

    # ==========================================================================#

    def _set_voltage_direct(self, voltage: float) -> None:
        self.session.write(":SOURce:LEVel:AUTO %.8f" % voltage)
        time.sleep(self._rampinterval)

    def _set_voltage_smart(self, voltage: float, progress: bool = False) -> None:
        # sweep to the target value step by step
        current_voltage = self.GetVoltage()
        if current_voltage == voltage:
            return

        if progress:
            dist = 1e3 * abs(current_voltage - voltage)
            pbar = tqdm(total=round(dist, 2), unit="V", leave=False)

        step = 10 * self._rampstep
        steps = max(1, round(abs(voltage - current_voltage) / step))
        voltages = np.linspace(current_voltage, voltage, num=steps + 1, endpoint=True)
        for tempvolt in voltages:
            self._set_voltage_direct(tempvolt)

            if progress:
                cur_dist = 1e3 * abs(tempvolt - voltage)
                pbar.update(round(dist - cur_dist, 2) - pbar.n)

        if progress:
            pbar.close()

    # Ramp up the voltage (volts) in increments of _rampstep, waiting _rampinterval
    # between each increment.
    def SetVoltage(self, voltage: float, progress: bool = True) -> None:
        self.OutputOn()
        self._set_voltage_smart(voltage, progress=progress)

    def _set_current_direct(self, current: float) -> None:
        self.session.write(":SOURce:LEVel:AUTO %.8f" % current)
        time.sleep(self._rampinterval)

    def _set_current_smart(self, current: float, progress: bool = False) -> None:
        # sweep to the target value step by step
        current_current = self.GetCurrent()
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
    def SetCurrent(self, current: float, progress: bool = True) -> None:
        self.OutputOn()
        self._set_current_smart(current, progress=progress)

    # Set to either current or voltage mode.
    def SetMode(self, mode: Literal["voltage", "current"]) -> None:
        if not (mode == "voltage" or mode == "current"):
            sys.stderr.write("Unknown output mode %s." % mode)
            return
        self.session.write("SOURce:FUNCtion %s" % mode)

    def SetRate(self, rate: float) -> None:
        self._rampstep = rate

    # ==========================================================================#

    # Returns the voltage in volts as a float
    def GetVoltage(self) -> float:
        self.session.write("SOURce:FUNCtion VOLTage")
        self.session.write("SOURce:LEVel?")
        result = self.session.read()
        return float(result.rstrip("\n"))

    # Returns the current in amps as a float
    def GetCurrent(self) -> float:
        self.session.write("SOURce:FUNCtion CURRent")
        self.session.write("SOURce:LEVel?")
        result = self.session.read()
        return float(result.rstrip("\n"))

    # Returns the mode (voltage or current)
    def GetMode(self) -> Literal["voltage", "current"]:
        self.session.write("SOURce:FUNCtion?")
        result = self.session.read()
        result = result.rstrip("\n")
        if result == "VOLT":
            return "voltage"
        else:
            return "current"

    # ==========================================================================#
