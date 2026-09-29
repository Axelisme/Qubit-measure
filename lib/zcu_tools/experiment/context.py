"""Explicit capabilities for one QICK experiment run."""

from dataclasses import dataclass
from typing import Any

from zcu_tools.plotting.plots import Plots


@dataclass(frozen=True)
class QickContext:
    """Hardware handles and caller-owned plots for a single run.

    QICK handles may be local objects or transport proxies. Their dynamic API
    stays at this hardware boundary. The caller finishes plots after the producer
    has stopped and releases presentation separately.
    """

    soc: Any
    soccfg: Any
    plots: Plots
