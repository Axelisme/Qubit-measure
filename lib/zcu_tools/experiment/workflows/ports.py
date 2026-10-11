"""Host capabilities used by workflow execution, without frontend imports."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from qick import QickConfig

from ...datafile import JsonObject
from ...device import BaseDeviceInfo
from ...plotting.plots import Plots
from ...progress_bar import BaseProgressBar
from ..stop_signal import StopSignal


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

    def get_device_cfg(self, name: str) -> BaseDeviceInfo:
        """Live-read detached settings retaining the actual driver Info subclass.

        Synchronous worker read, not an effect or cached GUI mapping. Unknown or
        disconnected errors identify name; native I/O failures propagate.
        """
        ...

    def setup_device(
        self, name: str, cfg: BaseDeviceInfo, cancel_signal: StopSignal
    ) -> None:
        """Revalidate complete cfg through the selected driver's Info model.

        Do not rely on model_validate(instance) to revalidate unchecked copies.
        Pass the same StopSignal through this seam and its original Event to
        native setup. Unknown/disconnected names identify name; model validation,
        driver limits and I/O errors propagate without retries or clamping.
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
