"""GE post analysis of an explicitly adopted primary record."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TypeAlias

from zcu_tools.experiment.records import AnalysisRecord, RunRecord
from zcu_lab.v2.singleshot.ge.core import GE_Cfg
from zcu_lab.v2.singleshot.ge.core import GE_Exp
from zcu_lab.v2.singleshot.ge.core import GE_Result
from zcu_lab.v2.singleshot.ge.core import GEAnalysis
from zcu_lab.v2.singleshot.ge.core import GEAnalyzeOptions
from zcu_lab.v2.singleshot.ge.core import GEPostAnalysis
from zcu_lab.v2.singleshot.ge.core import GEPostAnalyzeOptions
from zcu_tools.notebook.plotting import NotebookPlotHost, finish_failed_plots
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
        retained_options = deepcopy(options)
        working_source = RunRecord(cfg=primary.source.cfg, result=primary.source.result)
        plots = Plots(self._host)
        try:
            result = self._core.post_analyze(
                working_source,
                primary.result,
                deepcopy(retained_options),
                plots=plots,
            )
            figures = plots.finish()
            record = GEPostAnalysisRecord(
                primary=primary,
                options=retained_options,
                result=result,
                figures=figures,
            )
        except BaseException as error:
            finish_failed_plots(plots, error, operation="Post analysis")
            raise
        self.analysis = record
        self.analysis_plots = plots
        return record
