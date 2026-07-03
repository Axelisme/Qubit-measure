from __future__ import annotations

import threading
import time
from typing import Literal

from ._ramp import ramp_linear
from .base import BaseDevice, BaseDeviceInfo, device_operation

DEFAULT_RAMPSTEP = 0.01
RAMP_INTERVAL = 0.01  # seconds between steps (skipped in fast_mode)


class FakeDeviceInfo(BaseDeviceInfo):
    type: Literal["FakeDevice"] = "FakeDevice"
    output: Literal["on", "off"] = "off"
    value: float = 0.0
    rampstep: float = DEFAULT_RAMPSTEP

    def set_flux(self, value: float) -> None:
        self.value = value

    def set_freq(self, freq_Hz: float) -> None:
        self.value = freq_Hz

    def set_power(self, power_dBm: float) -> None:
        self.value = power_dBm

    def set_output(self, output: Literal["on", "off"]) -> None:
        self.output = output


class FakeDevice(BaseDevice[FakeDeviceInfo]):
    info_model = FakeDeviceInfo

    def __init__(self, fast_mode: bool = False) -> None:
        super().__init__("none", rm=None)
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
        progress: bool = False,
        stop_event: threading.Event | None = None,
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
        self.set_output(cfg.output)
        self._rampstep = cfg.rampstep
        self._set_value_smart(cfg.value, progress=progress, stop_event=stop_event)

    def get_info(self) -> FakeDeviceInfo:
        return FakeDeviceInfo(
            address=self.address,
            output=self.output,
            value=self.value,
            rampstep=self._rampstep,
        )
