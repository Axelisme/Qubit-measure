from __future__ import annotations

import threading
import time
from typing import Literal

from pydantic import Field, field_validator

from ._ramp import ramp_linear
from .base import BaseDevice, BaseDeviceInfo, device_operation

DEFAULT_RAMPSTEP = 0.01
RAMP_INTERVAL = 0.01  # seconds between steps (skipped in fast_mode)


class FakeDeviceInfo(BaseDeviceInfo):
    """Memory-device settings and readback in fixed, unconverted coordinates.

    Construction requires address. value_unit defaults to native, is stored in
    dumps; V and A are also supported without conversion. value is a
    strict real scalar; output controls enable state and rampstep is the native
    ramp increment.
    """

    type: Literal["FakeDevice"] = "FakeDevice"
    value_unit: Literal["native", "V", "A"] = Field(
        default="native", description="Fixed value/rampstep coordinate; no conversion."
    )
    output: Literal["on", "off"] = "off"
    value: float = 0.0
    rampstep: float = DEFAULT_RAMPSTEP

    @field_validator("value", mode="before")
    @classmethod
    def _validate_value(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "value must be a real numeric scalar (int or float); "
                "bool and strings are not accepted"
            )
        return value


class FakeDevice(BaseDevice[FakeDeviceInfo]):
    info_model = FakeDeviceInfo

    def __init__(
        self,
        *,
        fast_mode: bool = False,
        value_unit: Literal["native", "V", "A"] = "native",
    ) -> None:
        """Create a memory source with a lifetime-fixed value/rampstep unit.

        fast_mode disables ramp sleeps. value_unit is native, V or A; no numeric
        conversion is performed. Invalid units raise ValueError before creation.
        """
        if value_unit not in ("native", "V", "A"):
            raise ValueError("value_unit must be native, V or A")
        super().__init__("none", rm=None)
        self._value_unit: Literal["native", "V", "A"] = value_unit
        self.output: Literal["on", "off"] = "off"
        self.value = 0.0
        self._rampstep = DEFAULT_RAMPSTEP
        self._fast_mode = fast_mode

    def _open_session(self, rm: object | None) -> None:
        return None

    def get_output(self) -> Literal["on", "off"]:
        return self.output

    @device_operation
    def set_output(self, status: Literal["on", "off"]) -> None:
        self.output = status

    @device_operation
    def output_on(self) -> None:
        self.set_output("on")

    @device_operation
    def output_off(self) -> None:
        self.set_output("off")

    # ==========================================================================#

    def get_value(self) -> float:
        return self.value

    @device_operation
    def set_value(self, value: float) -> float:
        self.value = value
        return self.value

    @device_operation
    def close(self) -> None:
        """Fake devices own no external session."""

    def _set_value_smart(
        self,
        value: float,
        stop_event: threading.Event | None = None,
        *,
        progress: bool = False,
    ) -> None:
        current_value = self.get_value()

        def apply_value(target: float) -> None:
            self.value = float(target)
            if not self._fast_mode:
                time.sleep(RAMP_INTERVAL)

        ramp_linear(
            start=current_value,
            target=value,
            step=self._rampstep,
            apply_value=apply_value,
            progress=progress,
            desc="Ramp value",
            progress_decimals=6,
            stop_event=stop_event,
            include_start=False,
        )

    # ==========================================================================#

    def _setup(
        self,
        cfg: FakeDeviceInfo,
        *,
        progress: bool = True,
        stop_event: threading.Event | None = None,
    ) -> None:
        if cfg.value_unit != self._value_unit:
            raise ValueError("setup value_unit differs from fixed device unit")
        self.set_output(cfg.output)
        self._rampstep = cfg.rampstep
        self._set_value_smart(cfg.value, progress=progress, stop_event=stop_event)

    def get_info(self) -> FakeDeviceInfo:
        """Return fresh memory settings with the fixed unit; no I/O or mutation."""
        return FakeDeviceInfo(
            address=self.address,
            output=self.output,
            value=self.value,
            rampstep=self._rampstep,
            value_unit=self._value_unit,
        )
