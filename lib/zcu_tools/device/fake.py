from __future__ import annotations

import threading
import time
from typing import Literal, Optional

import numpy as np

from zcu_tools.progress_bar import make_pbar

from .base import BaseDevice, BaseDeviceInfo

DEFAULT_RAMPSTEP = 0.01
RAMP_INTERVAL = 0.01  # seconds between steps (skipped in fast_mode)


class FakeDeviceInfo(BaseDeviceInfo):
    type: Literal["FakeDevice"] = "FakeDevice"
    output: Literal["on", "off"] = "off"
    value: float = 0.0
    rampstep: float = DEFAULT_RAMPSTEP


class FakeDevice(BaseDevice[FakeDeviceInfo]):
    info_model = FakeDeviceInfo

    def __init__(self, fast_mode: bool = False) -> None:
        self.address = "none"
        self.output: Literal["on", "off"] = "off"
        self.value = 0.0
        self._rampstep = DEFAULT_RAMPSTEP
        self._fast_mode = fast_mode

    def get_output(self) -> Literal["on", "off"]:
        return self.output

    def set_output(self, status: Literal["on", "off"]) -> None:
        self.output = status

    def output_on(self) -> None:
        self.set_output("on")

    def output_off(self) -> None:
        self.set_output("off")

    # ==========================================================================#

    def get_value(self) -> float:
        return self.value

    def set_value(self, value: float) -> float:
        self.value = value
        return self.value

    def close(self) -> None:
        """Fake devices own no external session."""

    def _set_value_smart(
        self,
        value: float,
        progress: bool = False,
        stop_event: threading.Event | None = None,
    ) -> None:
        if self.value == value:
            return

        dist = abs(self.value - value)
        step = 10 * self._rampstep
        steps = max(1, round(dist / step))
        targets = np.linspace(self.value, value, num=steps + 1, endpoint=True)

        pbar = make_pbar(
            total=round(dist, 6),
            desc="Ramp value",
            leave=False,
            disable=not progress,
        )
        for target in targets[1:]:  # skip first (current value)
            if stop_event is not None and stop_event.is_set():
                break
            prev = self.value
            self.value = float(target)
            if not self._fast_mode:
                time.sleep(RAMP_INTERVAL)
            pbar.update(round(abs(self.value - prev), 6))
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
