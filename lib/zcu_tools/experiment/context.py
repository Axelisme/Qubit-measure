"""Explicit capabilities for one QICK experiment run."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from zcu_tools.device.base import BaseDevice
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.plotting.plots import Plots


@dataclass(frozen=True)
class RunContext:
    """Hardware handles and caller-owned plots for a single run.

    QICK handles may be local objects or transport proxies. Their dynamic API
    stays at this hardware boundary. The caller finishes plots after the producer
    has stopped and releases presentation separately.
    """

    soc: Any
    soccfg: Any
    plots: Plots
    devices: Mapping[str, BaseDevice[Any]]
    cancel_signal: StopSignal

    def __post_init__(self) -> None:
        # Freeze names, not drivers: their lifecycle remains with the caller.
        object.__setattr__(self, "devices", MappingProxyType(dict(self.devices)))
