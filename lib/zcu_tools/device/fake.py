from __future__ import annotations

import threading
import time
from typing import Literal

import numpy as np

from zcu_tools.progress_bar import make_pbar

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
        self.address = "none"
        self.output: Literal["on", "off"] = "off"
        self.value = 0.0
        self._rampstep = DEFAULT_RAMPSTEP
        self._fast_mode = fast_mode

        # FakeDevice bypasses BaseDevice.__init__ (no pyvisa session), so it must
        # create the same per-instance operation lock as real devices.
        self._init_locks()

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
        if current_value == value:
            return

        dist = abs(current_value - value)
        step = self._rampstep
        steps = max(1, round(dist / step))
        targets = np.linspace(current_value, value, num=steps + 1, endpoint=True)

        total = round(dist, 6)
        pbar = make_pbar(
            total=total,
            desc="Ramp value",
            leave=False,
            disable=not progress,
        )
        start = current_value
        for target in targets[1:]:  # skip first (current value)
            if stop_event is not None and stop_event.is_set():
                break
            self.value = float(target)
            current_value = self.value
            if not self._fast_mode:
                time.sleep(RAMP_INTERVAL)
            # Anchor the bar to absolute distance covered: independent rounding
            # of each step's |Δ| does not telescope back to `total`, so summing
            # could push the cumulative past `total` and trip tqdm's "clamping
            # frac" warning. min() guards the final fp residue.
            covered = min(round(abs(current_value - start), 6), total)
            pbar.set_progress(covered)
        pbar.close()

    # ==========================================================================#

    def _setup(
        self,
        cfg,
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
