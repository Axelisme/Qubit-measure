"""Host capabilities used by workflow execution, without frontend imports."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from qick import QickConfig

from ...datafile import JsonObject
from ...device import DeviceInfo
from ...plotting.plots import Plots
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
    """Host-owned connected devices; drivers enforce their hardcoded limits.

    Adapters forward cooperative cancellation and propagate driver failures.
    Neither the adapter nor the engine declares or checks device limits.
    """

    def capture_snapshot(self) -> Mapping[str, JsonObject]:
        """Read all connected drivers once into detached JSON, or raise.

        Called in the workflow worker before each experiment callback, under
        the host's RUN lease. Reuse capture_device_snapshot and the public
        driver mapping; do not read GUI cache or update GUI state. An explicitly
        empty mapping means no connected devices. Read/serialization errors
        propagate, without retries or cached fallbacks.
        """
        ...

    def set_value(
        self,
        name: str,
        value: float,
        cancel_signal: StopSignal,
        *,
        unit: Literal["A", "V", "native"],
    ) -> None:
        """Set an absolute value in explicit A, V or native units, or raise.

        The host checks unit against the driver without converting values or
        changing mode, rampstep or output. Fake devices require native.
        Observe ``cancel_signal`` between cooperative work segments. Unknown or
        disconnected names, invalid limits, and I/O failures must raise.
        Unknown/disconnected-device errors must identify the requested name.
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

    ``plots`` is the host-owned Plots whose commands synchronously dispatch
    artists to its PlotHost owner. The host stops execution before finish/release.
    ``progress`` accepts the existing
    make_pbar arguments and returns BaseProgressBar, including runtime bars.
    Do not use make_pbar itself as this factory; use a concrete backend factory.
    ``clock`` implements aware UTC time and interruptible waits.
    ``context`` is detached read-only workflow data, or None when unavailable.
    ``soc`` is an opaque native SoC handle. ``soccfg`` is its QICK configuration;
    both must be present or both absent. ``devices`` is a connected-device
    adapter, or None. The engine never connects or disconnects these handles.
    """

    plots: Plots
    progress: ProgressFactory
    clock: Clock
    context: C | None = None
    soc: object | None = None
    soccfg: QickConfig | None = None
    devices: DevicePort | None = None

    def __post_init__(self) -> None:
        if (self.soc is None) != (self.soccfg is None):
            raise ValueError("soc and soccfg must both be present or both be absent")
