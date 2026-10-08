from __future__ import annotations

import math
import threading
import time
import warnings
from typing import TYPE_CHECKING, Literal, Self

from pydantic import Field, field_validator, model_validator

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
    """Validated GS200 setup and readback, with sample-specific output limits.

    Value and rampstep use V in voltage mode and A in current mode. Invalid
    numeric values, non-positive steps/limits, and values or steps outside the
    active mode's output/rampstep limits raise Pydantic ValidationError.
    """

    type: Literal["YOKOGS200"] = "YOKOGS200"
    output: Literal["on", "off"] = Field(
        default="off", description="Output enable state."
    )
    mode: Literal["voltage", "current"] = Field(
        default="voltage",
        description="Source mode; determines value and rampstep units.",
    )
    value: float = Field(
        default=0.0, allow_inf_nan=False, description="Output level in V or A."
    )
    rampstep: float = Field(
        default=DEFAULT_RAMPSTEP["voltage"],
        gt=0,
        allow_inf_nan=False,
        description="Positive ramp increment in V or A; defaults to 1e-3 V or 1e-6 A.",
    )
    max_voltage: float = Field(
        default=20.0,
        gt=0,
        allow_inf_nan=False,
        description="Maximum absolute output voltage in V.",
    )
    max_current: float = Field(
        default=20e-3,
        gt=0,
        allow_inf_nan=False,
        description="Maximum absolute output current in A.",
    )

    max_voltage_rampstep: float = Field(
        default=1e-2,
        gt=0,
        allow_inf_nan=False,
        description="Maximum ramp increment in voltage mode, in V.",
    )
    max_current_rampstep: float = Field(
        default=1e-5,
        gt=0,
        allow_inf_nan=False,
        description="Maximum ramp increment in current mode, in A.",
    )

    @model_validator(mode="before")
    @classmethod
    def _default_rampstep(cls, data: object) -> object:
        if (
            isinstance(data, dict)
            and "rampstep" not in data
            and data.get("mode") == "current"
        ):
            return {**data, "rampstep": DEFAULT_RAMPSTEP["current"]}
        return data

    @model_validator(mode="after")
    def _validate_output_limit(self) -> Self:
        limit = self.max_voltage if self.mode == "voltage" else self.max_current
        if abs(self.value) > limit:
            raise ValueError(f"value exceeds {self.mode} output limit {limit}")
        step_limit = (
            self.max_voltage_rampstep
            if self.mode == "voltage"
            else self.max_current_rampstep
        )
        if self.rampstep > step_limit:
            raise ValueError(
                f"rampstep exceeds {self.mode} rampstep limit {step_limit}"
            )
        return self

    @field_validator("value", mode="before")
    @classmethod
    def _validate_value(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "value must be a real numeric scalar (int or float); "
                "bool and strings are not accepted"
            )
        return value

    def set_flux(self, value: float) -> None:
        """Set the output level in the active mode's units, enforcing its limit.

        ValidationError leaves the previous level unchanged.
        """
        updated = self.with_updates(value=value)
        self.value = updated.value


class YOKOGS200(BaseDevice[YOKOGS200Info]):
    info_model = YOKOGS200Info

    # Initializes session for device.
    # address: address of device, rm: VISA resource manager
    def __init__(self, address: str, rm: ResourceManager) -> None:
        super().__init__(address, rm)

        mode = self.get_mode()

        defaults = YOKOGS200Info(address=address, mode=mode)
        self._rampstep = defaults.rampstep
        self._max_voltage = defaults.max_voltage
        self._max_current = defaults.max_current
        self._max_voltage_rampstep = defaults.max_voltage_rampstep
        self._max_current_rampstep = defaults.max_current_rampstep
        self._rampinterval = 0.01

    # ==========================================================================#

    def get_output(self) -> Literal["on", "off"]:
        return STATUS_MAP_INV[self.query(":OUTPut?")]  # type: ignore

    @device_operation
    def set_output(self, status: Literal["on", "off"]) -> None:
        """Set output on/off only when the source level is zero.

        An unchanged status returns without checking the level or writing.
        RuntimeError rejects a transition at a nonzero level without writing;
        the caller must first ramp to zero. This method never ramps automatically.
        """
        if self.get_output() == status:
            return
        if self._get_level() != 0.0:
            raise RuntimeError(
                "Cannot switch output while level is nonzero. Please ramp to zero first."
            )
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
        if not math.isfinite(voltage) or abs(voltage) > self._max_voltage:
            raise RuntimeError(
                f"Voltage must be finite and not over {self._max_voltage:g}V in magnitude"
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
        if not math.isfinite(current) or abs(current) > self._max_current:
            raise RuntimeError(
                f"Current must be finite and not over {self._max_current:g}A in magnitude"
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
        """Select source mode and a positive finite ramp step (V or A).

        Omitted rampstep uses the mode's default. ValidationError rejects invalid
        or over-limit steps before I/O. RuntimeError rejects a nonzero mode switch unless force
        is true; force does not change the configured output limits.
        """
        cfg = YOKOGS200Info(
            address=self.address,
            mode=mode,
            rampstep=DEFAULT_RAMPSTEP[mode] if rampstep is None else rampstep,
            max_voltage_rampstep=self._max_voltage_rampstep,
            max_current_rampstep=self._max_current_rampstep,
        )
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
        self._rampstep = cfg.rampstep

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
        self.set_output(cfg.output)

        cur_mode = self.get_mode()

        if cfg.mode != cur_mode:
            raise RuntimeError(
                f"Current mode: {cur_mode} in device {self.address}, but cfg requires: {cfg.mode} mode, "
                "YOKOGS200 does not support implicit setup mode to prevent sudden current/voltage change, "
                "Please change the device mode manually before calling setup, "
                "Remember to turn value to zero before changing mode"
            )

        self._rampstep = cfg.rampstep
        self._max_voltage = cfg.max_voltage
        self._max_current = cfg.max_current
        self._max_voltage_rampstep = cfg.max_voltage_rampstep
        self._max_current_rampstep = cfg.max_current_rampstep

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
            max_voltage=self._max_voltage,
            max_current=self._max_current,
            max_voltage_rampstep=self._max_voltage_rampstep,
            max_current_rampstep=self._max_current_rampstep,
        )
