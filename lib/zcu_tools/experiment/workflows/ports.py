"""Host capabilities used by workflow execution, without frontend imports."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from qick import QickConfig

from ...device import BaseDeviceInfo, DeviceMapping
from ...progress_bar import BaseProgressBar
from ..stop_signal import StopSignal
from .plotting import WorkflowPlots


class DevicePort(Protocol):
    """Host-owned connected devices; drivers enforce their hardcoded limits.

    Adapters forward cooperative cancellation and propagate driver failures.
    Neither the adapter nor the engine declares or checks device limits.
    """

    def get_connected_devices(self) -> DeviceMapping:
        """Return a detached name-to-driver mapping, empty when none are connected.

        Call on the workflow worker under the RUN lease. Drivers remain borrowed;
        their lifecycle belongs to the session. Do not connect, read hardware or
        use GUI caches here. env.run owns the pre-call device snapshot.
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

    ``plots`` owns workflow trends and the latest independent operation group.
    Its commands synchronously dispatch artists to the same PlotHost owner.
    env.run finishes each operation; the host stops execution before finishing
    workflow plots and releasing both presentations.
    ``progress`` accepts the existing
    make_pbar arguments and returns BaseProgressBar, including runtime bars.
    Do not use make_pbar itself as this factory; use a concrete backend factory.
    ``clock`` implements aware UTC time and interruptible waits.
    ``context`` is detached read-only workflow data, or None when unavailable.
    ``soc`` is an opaque native SoC handle. ``soccfg`` is its QICK configuration;
    both must be present or both absent. ``devices`` is a connected-device
    adapter, or None. The engine never connects or disconnects these handles.
    """

    plots: WorkflowPlots
    progress: ProgressFactory
    clock: Clock
    context: C | None = None
    soc: object | None = None
    soccfg: QickConfig | None = None
    devices: DevicePort | None = None

    def __post_init__(self) -> None:
        if (self.soc is None) != (self.soccfg is None):
            raise ValueError("soc and soccfg must both be present or both be absent")
