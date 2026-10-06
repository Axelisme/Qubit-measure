"""Experiment-facing integration with the existing Schedule runtime."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import is_dataclass

import numpy as np
from numpy.typing import NDArray
from qick import QickConfig

from ..stop_signal import StopSignal
from ..v2.runtime import Schedule, SignalBuffer
from ..v2.runtime.schedule import ScheduleOutcome
from .models import MissingCapability
from .ports import DevicePort


class Run[Cfg]:
    """One experiment's cfg copy, native capabilities, and cancellation signal.

    cfg is a deepcopy of the supplied dataclass and may be modified by the
    experiment. cancel_signal is the engine-owned StopSignal; do not clear it.
    soc/soccfg/devices raise MissingCapability when their ports are absent.
    outcome retains the first non-completed Schedule result. Constructor
    soc/soccfg/devices arguments are native handles or None, never copied or
    connected here. Construction does not set up devices, acquire, or save.
    """

    def __init__(
        self,
        cfg: Cfg,
        cancel_signal: StopSignal,
        *,
        soc: object | None = None,
        soccfg: QickConfig | None = None,
        devices: DevicePort | None = None,
    ) -> None:
        if isinstance(cfg, type) or not is_dataclass(cfg):
            raise TypeError("Experiment cfg must be a dataclass instance")
        self.cfg: Cfg = deepcopy(cfg)
        self.cancel_signal = cancel_signal
        self._soc = soc
        self._soccfg = soccfg
        self._devices = devices
        self._outcome = ScheduleOutcome()
        self._buffers: list[tuple[SignalBuffer, tuple[NDArray[np.float64], ...]]] = []

    @property
    def soc(self) -> object:
        """Return the native SoC handle, or raise MissingCapability('soc')."""
        if self._soc is None:
            raise MissingCapability("soc")
        return self._soc

    @property
    def soccfg(self) -> QickConfig:
        """Return the injected QICK config, or raise MissingCapability('soc')."""
        if self._soccfg is None:
            raise MissingCapability("soc")
        return self._soccfg

    @property
    def devices(self) -> DevicePort:
        """Return the connected-device adapter, or raise MissingCapability."""
        if self._devices is None:
            raise MissingCapability("devices")
        return self._devices

    @property
    def outcome(self) -> ScheduleOutcome:
        """Return a detached first non-completed Schedule outcome.

        exception is the original cause, not a deepcopy. Later completed
        schedules never erase failed, interrupted, or stopped outcomes.
        """
        return ScheduleOutcome(
            self._outcome.status, self._outcome.reason, self._outcome.exception
        )

    def buffer(
        self,
        shape: tuple[int, ...],
        *,
        axes: tuple[NDArray[np.float64], ...],
    ) -> SignalBuffer:
        """Create a complex128 SignalBuffer and remember detached float64 axes.

        Each dimension must be a positive integer. There must be one finite
        one-dimensional float64 axis of matching length per dimension.
        Invalid dimensions, dtype, rank, length, or values raise ValueError.
        Buffers start with NaNs; partial data remains valid runtime output.
        """
        if not shape or any(type(size) is not int or size <= 0 for size in shape):
            raise ValueError("Buffer dimensions must be positive integers")
        if len(axes) != len(shape):
            raise ValueError("Buffer axes must match shape rank")
        for size, axis in zip(shape, axes, strict=True):
            if (
                axis.dtype != np.float64
                or axis.ndim != 1
                or axis.size != size
                or not np.isfinite(axis).all()
            ):
                raise ValueError("Buffer axes must be finite matching float64 vectors")
        buffer = SignalBuffer(shape)
        self._buffers.append((buffer, tuple(axis.copy() for axis in axes)))
        return buffer

    @contextmanager
    def schedule(
        self, buffer: SignalBuffer
    ) -> Generator[Schedule[Cfg, Run[Cfg]], None, None]:
        """Open a Schedule with a local cfg copy, env=self, and the same signal.

        buffer must have been created by this Run, otherwise ValueError.
        The outcome is collected even if the with body raises. Arbitrary body
        exceptions propagate; only failures captured by Schedule are recoverable.
        """
        if not any(existing is buffer for existing, _ in self._buffers):
            raise ValueError("Schedule buffer must belong to this Run")
        schedule = Schedule[Cfg, Run[Cfg]](
            self.cfg, buffer, stop=self.cancel_signal, env=self
        )
        try:
            with schedule:
                yield schedule
        finally:
            if self._outcome.status == "completed":
                self._outcome = schedule.outcome
