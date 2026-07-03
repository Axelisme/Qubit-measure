from __future__ import annotations

import threading
import time
import warnings
from typing import TYPE_CHECKING, Literal

from ._ramp import ramp_linear
from .base import BaseDevice, BaseDeviceInfo, device_operation

if TYPE_CHECKING:
    from pyvisa import ResourceManager

STATUS_MAP = {"on": "1", "off": "0"}
MODE_MAPS = {"voltage": "VOLT", "current": "CURR"}
DEFAULT_RAMPSTEP = {
    "voltage": 1e-3,
    "current": 1e-6,
}


STATUS_MAP_INV = {v: k for k, v in STATUS_MAP.items()}
MODE_MAPS_INV = {v: k for k, v in MODE_MAPS.items()}


class YOKOGS200Info(BaseDeviceInfo):
    type: Literal["YOKOGS200"] = "YOKOGS200"
    output: Literal["on", "off"] = "off"
    mode: Literal["voltage", "current"] = "voltage"
    value: float = 0.0
    rampstep: float = DEFAULT_RAMPSTEP["voltage"]

    def set_flux(self, value: float) -> None:
        self.value = value


class YOKOGS200(BaseDevice[YOKOGS200Info]):
    info_model = YOKOGS200Info

    # Initializes session for device.
    # address: address of device, rm: VISA resource manager
    def __init__(self, address: str, rm: ResourceManager) -> None:
        super().__init__(address, rm)

        mode = self.get_mode()

        self._rampstep = DEFAULT_RAMPSTEP[mode]
        self._rampinterval = 0.01

    # ==========================================================================#

    def get_output(self) -> Literal["on", "off"]:
        return STATUS_MAP_INV[self.query(":OUTPut?")]  # type: ignore

    @device_operation
    def set_output(self, status: Literal["on", "off"]) -> None:
        self.write(f":OUTPut {STATUS_MAP[status]}")

    # Turn on output
    @device_operation
    def output_on(self) -> None:
        self.set_output("on")

    # Turn off output
    @device_operation
    def output_off(self) -> None:
        self.set_output("off")

    # ==========================================================================#

    def _check_voltage(self, voltage: float) -> None:
        CHECK_VOLTAGE_LIMIT = 20
        if abs(voltage) > CHECK_VOLTAGE_LIMIT:
            raise RuntimeError(
                f"Try to set voltage to over {CHECK_VOLTAGE_LIMIT}V, are you sure you want to do this?"
            )

    def _set_voltage_direct(self, voltage: float) -> None:
        self._check_voltage(voltage)
        self.write(f":SOURce:LEVel:AUTO {voltage:.8f}")
        time.sleep(self._rampinterval)

    def _set_voltage_smart(
        self,
        voltage: float,
        progress: bool = False,
        stop_event: threading.Event | None = None,
    ) -> None:
        self._check_voltage(voltage)
        current_voltage = self.get_voltage()

        ramp_linear(
            start=current_voltage,
            target=voltage,
            step=self._rampstep,
            apply_value=self._set_voltage_direct,
            progress=progress,
            desc="Ramp voltage",
            unit="V",
            progress_decimals=2,
            stop_event=stop_event,
            include_start=True,
        )

    @device_operation
    def set_voltage(
        self,
        voltage: float,
        progress: bool = True,
        stop_event: threading.Event | None = None,
    ) -> float:
        mode = self.get_mode()
        if mode != "voltage":
            raise RuntimeError(
                f"One can only set voltage when the device is in voltage mode. but it is in {mode} mode."
            )

        if self.get_output() != "on" and voltage != 0.0:
            raise RuntimeError(
                "Output is off, please turn on the output before setting voltage"
            )
        self._set_voltage_smart(voltage, progress=progress, stop_event=stop_event)

        return self.get_voltage()

    def _check_current(self, current: float) -> None:
        CHECK_CURRENT_LIMIT = 20e-3
        if abs(current) > CHECK_CURRENT_LIMIT:
            raise RuntimeError(
                f"Try to set current to over {CHECK_CURRENT_LIMIT}A, are you sure you want to do this?"
            )

    def _set_current_direct(self, current: float) -> None:
        self._check_current(current)
        self.write(f":SOURce:LEVel:AUTO {current:.8f}")
        time.sleep(self._rampinterval)

    def _set_current_smart(
        self,
        current: float,
        progress: bool = False,
        stop_event: threading.Event | None = None,
    ) -> None:
        self._check_current(current)
        current_current = self.get_current()

        ramp_linear(
            start=current_current,
            target=current,
            step=self._rampstep,
            apply_value=self._set_current_direct,
            progress=progress,
            desc="Ramp current",
            unit="mA",
            progress_scale=1e3,
            progress_decimals=2,
            stop_event=stop_event,
            include_start=True,
        )

    # Ramp up the current (amps) in increments of _rampstep, waiting _rampinterval
    # between each increment.
    @device_operation
    def set_current(
        self,
        current: float,
        progress: bool = True,
        stop_event: threading.Event | None = None,
    ) -> float:
        mode = self.get_mode()
        if mode != "current":
            raise RuntimeError(
                f"One can only set current when the device is in current mode. but it is in {mode} mode."
            )

        if self.get_output() != "on" and current != 0.0:
            raise RuntimeError(
                "Output is off, please turn on the output before setting current"
            )
        self._set_current_smart(current, progress=progress, stop_event=stop_event)

        return self.get_current()

    # Set to either current or voltage mode.
    @device_operation
    def set_mode(
        self,
        mode: Literal["voltage", "current"],
        force: bool = False,
        rampstep: float | None = None,
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

        if rampstep is None:
            rampstep = DEFAULT_RAMPSTEP[mode]

        self.write(f":SOURce:FUNCtion {MODE_MAPS[mode]}")
        self._rampstep = rampstep

    # Returns the mode (voltage or current)
    def get_mode(self) -> Literal["voltage", "current"]:
        return MODE_MAPS_INV[self.query(":SOURce:FUNCtion?")]  # type: ignore

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

        return self._get_level()

    # Returns the current in amps as a float
    def get_current(self) -> float:
        mode = self.get_mode()
        if mode != "current":
            raise RuntimeError(
                f"One can only get current when the device is in current mode. but it is in {mode} mode."
            )

        return self._get_level()

    # ==========================================================================#

    def _setup(
        self,
        cfg: YOKOGS200Info,
        *,
        progress: bool = True,
        stop_event: threading.Event | None = None,
    ) -> None:
        if self.get_output() != "on" and cfg.output == "on":
            warnings.warn("YOKOGS200 output is off, did you forget to turn it on?")

        cur_mode = self.get_mode()

        if cfg.mode != cur_mode:
            raise RuntimeError(
                f"Current mode: {cur_mode} in device {self.address}, but cfg requires: {cfg.mode} mode, "
                "YOKOGS200 does not support implicit setup mode to prevent sudden current/voltage change, "
                "Please change the device mode manually before calling setup, "
                "Remember to turn value to zero before changing mode"
            )

        self._rampstep = cfg.rampstep

        value = cfg.value
        if cur_mode == "current":
            self.set_current(value, progress=progress, stop_event=stop_event)
        elif cur_mode == "voltage":
            self.set_voltage(value, progress=progress, stop_event=stop_event)
        else:
            raise ValueError(f"Unknown mode {cur_mode} in device {self.address}")

    def get_info(self) -> YOKOGS200Info:
        return YOKOGS200Info(
            address=self.address,
            output=self.get_output(),
            mode=self.get_mode(),
            value=self._get_level(),
            rampstep=self._rampstep,
        )
