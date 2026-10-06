"""Host capabilities used by workflow execution, without frontend imports."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import JsonValue
from qick import QickConfig

from ...progress_bar import BaseProgressBar
from ..stop_signal import StopSignal


@dataclass(frozen=True)
class JsonParameters:
    """Driver-owned JSON settings; ``values`` maps parameter names to JSON values.

    The driver adapter defines valid names, units, ranges, and combinations.
    Workflow execution copies these settings before handing them to the adapter.
    """

    values: dict[str, JsonValue]


@dataclass(frozen=True)
class DeviceSetup:
    """One connected device's complete setup.

    ``name`` is the host's connected-device name. ``parameters`` contains opaque
    driver settings, not workflow tunables or expressions.
    """

    name: str
    parameters: JsonParameters

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("DeviceSetup.name must not be empty")


@dataclass(frozen=True)
class DeviceSnapshot:
    """Provenance at start/resume; ``items`` is the ordered device setup tuple.

    This snapshot does not assert that hardware stays unchanged afterwards.
    """

    items: tuple[DeviceSetup, ...]


class DevicePort(Protocol):
    """Host-owned connected devices; adapters enforce device limits and cancel."""

    def set_value(self, name: str, value: float, cancel_signal: StopSignal) -> None:
        """Set an absolute value in native device units, or raise on failure.

        Observe ``cancel_signal`` between cooperative work segments. Unknown or
        disconnected names, invalid limits, and I/O failures must raise.
        """
        ...

    def setup(
        self, settings: tuple[DeviceSetup, ...], cancel_signal: StopSignal
    ) -> None:
        """Apply explicit settings in order; reject unknown settings or I/O errors.

        Observe cancellation while operating hardware. The engine does not
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
