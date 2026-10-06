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
from .display import Live1D, Live2D, Live2DRow
from .models import MissingCapability
from .ports import DevicePort, DeviceSetup, PlotPort


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
        self._live: Live1D | Live2D | Live2DRow | None = None
        self._plots: PlotPort | None = None
        self._live_error: BaseException | None = None

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
    def device_setup(self) -> tuple[DeviceSetup, ...] | None:
        """Return cfg.dev settings, or None when absent/disabled.

        A present dev must be None or a tuple of DeviceSetup. Invalid cfg
        adaptation raises TypeError before the device adapter is invoked.
        """
        settings = getattr(self.cfg, "dev", None)
        if settings is None:
            return None
        if not isinstance(settings, tuple) or not all(
            isinstance(setting, DeviceSetup) for setting in settings
        ):
            raise TypeError("cfg.dev must be tuple[DeviceSetup, ...] or None")
        return settings

    @property
    def live_error(self) -> BaseException | None:
        """Return the first live failure cause, preserving object identity.

        Engine checks this before interpreting a Schedule outcome, because
        Schedule may have caught a projection or snapshot callback failure.
        """
        return self._live_error

    def bind_live(self, live: Live1D | Live2D | Live2DRow, plots: PlotPort) -> None:
        """Bind one live declaration before experiment code creates buffers.

        This package-internal Engine seam is not an additional host API.
        More than one buffer with live enabled is ambiguous and fails the run.
        Projection/refresh errors propagate and remain available in live_error.
        Binding after buffer creation or binding twice raises ValueError.
        """
        if self._buffers or self._live is not None:
            raise ValueError("Live must be bound once before buffer creation")
        self._live = live
        self._plots = plots

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
        captured_axes = tuple(axis.copy() for axis in axes)
        if self._live is not None and self._buffers:
            error = ValueError("Live requires exactly one unambiguous buffer")
            self._live_error = error
            raise error

        def update(data: NDArray[np.complex128]) -> None:
            self._update_live(data, captured_axes)

        buffer = SignalBuffer(
            shape, on_update=update if self._live is not None else None
        )
        self._buffers.append((buffer, captured_axes))
        return buffer

    def _update_live(
        self, data: NDArray[np.complex128], axes: tuple[NDArray[np.float64], ...]
    ) -> None:
        live, plots = self._live, self._plots
        if live is None or plots is None:
            raise RuntimeError("Live callback has no binding")
        try:
            projected = live.y(data)
            if projected.shape != data.shape or not np.isrealobj(projected):
                raise ValueError(
                    "Live projection must preserve shape and produce real values"
                )
            if isinstance(live, Live1D):
                if data.ndim != 1:
                    raise ValueError("Live1D requires a one-dimensional buffer")
                live.line.set_data(axes[0], projected)
                figure = live.line.get_figure(root=True)
            else:
                if isinstance(live, Live2D):
                    if data.ndim != 2:
                        raise ValueError("Live2D requires a two-dimensional buffer")
                    live.image.set_data(projected)
                else:
                    image = np.array(
                        live.image.get_array(), dtype=np.float64, copy=True
                    )
                    if (
                        data.ndim != 1
                        or image.ndim != 2
                        or type(live.row) is not int
                        or not 0 <= live.row < image.shape[0]
                        or data.size != image.shape[1]
                    ):
                        raise ValueError(
                            "Live2DRow requires a valid row and matching width"
                        )
                    image[live.row] = projected
                    live.image.set_data(image)
                figure = live.image.get_figure(root=True)
            if figure is None:
                raise ValueError("Live artist must belong to a Figure")
            plots.refresh(figure)
        except (Exception, KeyboardInterrupt) as error:
            # Keep the display source even when Schedule catches this callback.
            if self._live_error is None:
                self._live_error = error
            raise

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
