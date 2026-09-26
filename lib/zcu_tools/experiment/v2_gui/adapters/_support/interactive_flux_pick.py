"""Result and analyze-params types shared by the measure flux-dependence adapters."""

from __future__ import annotations

from dataclasses import dataclass

from matplotlib.figure import Figure

from zcu_tools.gui.app.main.adapter import AnalyzeResultBase


@dataclass
class FluxPickParams:
    """The projection is fixed by the adapter, not a user analyze parameter."""


@dataclass
class FluxPickResult(AnalyzeResultBase):
    flx_half: float
    flx_int: float
    flx_period: float
    figure: Figure | None = None
