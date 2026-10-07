"""Host capabilities used by workflow execution, without frontend imports."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qick import QickConfig

from ...device import DeviceInfo
from ...progress_bar import BaseProgressBar
from ..stop_signal import StopSignal


@dataclass(frozen=True)
class DeviceSnapshot:
    """Provenance at start/resume, captured by the engine as detached data.

    ``items`` maps connected-device names to native typed DeviceInfo models in
    setup order; an empty mapping means no devices. Models contain the driver
    identity, settings, and native units defined by their device owner.
    This snapshot does not assert that hardware stays unchanged afterwards.
    """

    items: Mapping[str, DeviceInfo]


class DevicePort(Protocol):
    """Host-owned connected devices; adapters enforce device limits and cancel."""

    def set_value(self, name: str, value: float, cancel_signal: StopSignal) -> None:
        """Set an absolute value in native device units, or raise on failure.

        Observe ``cancel_signal`` between cooperative work segments. Unknown or
        disconnected names, invalid limits, and I/O failures must raise.
        """
        ...

    def setup(
        self, settings: Mapping[str, DeviceInfo], cancel_signal: StopSignal
    ) -> None:
        """Apply native DeviceInfo models in mapping order, or raise on failure.

        Keys name connected devices. Reuse the native device owner for name,
        address, model, and setting validation; retain its original failures.
        Pass cancel_signal.event to cooperative native setup and return when
        cancelled. The engine decides whether to discard the step. It does not
        connect devices, repair settings, or retry this operation.
        """
        ...


class PlotPort(Protocol):
    """Figure capability called only on the engine execution thread."""

    def axes(self, name: str) -> Axes:
        """Return stable axes for a nonempty name; the host chooses the layout."""
        ...

    def refresh(self, figure: Figure) -> None:
        """Capture a detached snapshot before returning, or raise on failure.

        Do not let frontend code mutate engine-owned artists. GUI scheduling and
        snapshot throttling belong to the host, not to the engine state machine.
        """
        ...


class Clock(Protocol):
    """Wall-clock timing with cooperative cancellation, shared by host adapters."""

    def now(self) -> datetime:
        """Return a timezone-aware UTC time."""
        ...

    def wait_until(self, target: datetime, cancel_signal: StopSignal) -> None:
        """Return when the aware target is reached or cancellation is requested.

        A past target returns immediately. Do not raise a control exception for
        cancellation; the engine decides whether to discard the step.
        """
        ...


# Existing runtime bars use the same variadic factory through use_pbar_factory.
type ProgressFactory = Callable[..., BaseProgressBar]


@dataclass(frozen=True)
class EnginePorts[C]:
    """Capabilities owned and injected by the host.

    ``plots`` snapshots engine figures. ``progress`` accepts the existing
    make_pbar arguments and returns BaseProgressBar, including runtime bars.
    Do not use make_pbar itself as this factory; use a concrete backend factory.
    ``clock`` implements aware UTC time and interruptible waits.
    ``context`` is detached read-only workflow data, or None when unavailable.
    ``soc`` is an opaque native SoC handle. ``soccfg`` is its QICK configuration;
    both must be present or both absent. ``devices`` is a connected-device
    adapter, or None. The engine never connects or disconnects these handles.
    """

    plots: PlotPort
    progress: ProgressFactory
    clock: Clock
    context: C | None = None
    soc: object | None = None
    soccfg: QickConfig | None = None
    devices: DevicePort | None = None

    def __post_init__(self) -> None:
        if (self.soc is None) != (self.soccfg is None):
            raise ValueError("soc and soccfg must both be present or both be absent")
