"""GE post analysis of an explicitly adopted primary record."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TypeAlias

from zcu_tools.experiment.records import AnalysisRecord, RunRecord
from zcu_tools.experiment.v2.singleshot.ge import (
    GE_Cfg,
    GE_Exp,
    GE_Result,
    GEAnalysis,
    GEAnalyzeOptions,
    GEPostAnalysis,
    GEPostAnalyzeOptions,
)
from zcu_tools.notebook.plotting import NotebookPlotHost
from zcu_tools.plotting.plots import PlotHost, Plots

GEPrimaryRecord: TypeAlias = AnalysisRecord[
    GE_Cfg, GE_Result, GEAnalyzeOptions, GEAnalysis
]


@dataclass(frozen=True)
class GEPostAnalysisRecord(
    AnalysisRecord[GE_Cfg, GE_Result, GEPostAnalyzeOptions, GEPostAnalysis]
):
    """Retain the adopted primary; derive the source instead of accepting another."""

    primary: GEPrimaryRecord
    source: RunRecord[GE_Cfg, GE_Result] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", self.primary.source)
        super().__post_init__()


class GEPostAnalyzer:
    """Publish post results only after finishing this operation's presentation.

    The caller selects a primary record explicitly. This tool never reads another
    adapter's current run or FIT. Retained figures stay usable after later calls
    and after releasing their separate presentation handles.
    """

    def __init__(self, core: GE_Exp, *, host: PlotHost | None = None) -> None:
        self._core = core
        self._host = NotebookPlotHost() if host is None else host
        self.analysis: GEPostAnalysisRecord | None = None
        self.analysis_plots: Plots | None = None

    def analyze(
        self, primary: GEPrimaryRecord, options: GEPostAnalyzeOptions
    ) -> GEPostAnalysisRecord:
        raise NotImplementedError("Explicit adopted-primary post analysis")
