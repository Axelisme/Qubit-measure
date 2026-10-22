"""Explicit capabilities for one QICK experiment run."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from zcu_tools.device.base import DeviceMapping
from zcu_tools.experiment.scope import Scope
from zcu_tools.plotting.plots import Plots


@dataclass(frozen=True)
class RunContext:
    """Hardware handles and caller-owned plots for a single run.

    QICK handles may be local objects or transport proxies. Their dynamic API
    stays at this hardware boundary. The caller finishes plots after the producer
    has stopped and releases presentation separately. scope is the explicit
    run/operation boundary supplied by that owner; core passes it to Schedule
    and uses scope.stop only for low-level cancellation APIs.
    """

    soc: Any
    soccfg: Any
    plots: Plots
    devices: DeviceMapping
    scope: Scope

    def __post_init__(self) -> None:
        # Freeze names, not drivers: their lifecycle remains with the caller.
        object.__setattr__(self, "devices", MappingProxyType(dict(self.devices)))
