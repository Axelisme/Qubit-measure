"""Notebook convenience entry for singleshot GE FIT and post analysis."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from zcu_tools.experiment.context import QickContext
from zcu_tools.experiment.v2.singleshot.ge import (
    GE_ANALYZE_DEFAULTS,
    GE_POST_ANALYZE_DEFAULTS,
    GE_Cfg,
    GE_Result,
    GEAnalysis,
    GEAnalyzeOptions,
    GEPostAnalysis,
    GEPostAnalyzeOptions,
)
from zcu_tools.experiment.v2.singleshot.ge import GE_Exp as GECore
from zcu_tools.notebook.plotting import NotebookPlotHost
from zcu_tools.plotting.plots import NonPresentingHost, Plots


@dataclass(frozen=True)
class GEAnalysisRecord:
    source: GE_Result
    options: GEAnalyzeOptions
    result: GEAnalysis
    plots: Plots


@dataclass(frozen=True)
class GEPostAnalysisRecord:
    source: GE_Result
    primary: GEAnalysis
    primary_options: GEAnalyzeOptions
    options: GEPostAnalyzeOptions
    result: GEPostAnalysis
    plots: Plots


class GEExp:
    """Publish successful GE operations as retained, separately named plots.

    Analysis without an explicit result uses the latest successful run or load.
    Post analysis always uses the adopted FIT record's source and calibration.
    A failed operation preserves both published records and releases only its
    own presentation; users may retain older Plots and save their figures.
    """

    def __init__(self, *, present: bool = True) -> None:
        self._core = GECore()
        self._host = NotebookPlotHost() if present else NonPresentingHost()
        self.last_result: GE_Result | None = None
        self.run_plots: Plots | None = None
        self._analysis: GEAnalysisRecord | None = None
        self._post_analysis: GEPostAnalysisRecord | None = None

    @property
    def analysis(self) -> GEAnalysisRecord | None:
        return self._analysis

    @property
    def post_analysis(self) -> GEPostAnalysisRecord | None:
        return self._post_analysis

    @property
    def analysis_plots(self) -> Plots:
        if self._analysis is None:
            raise RuntimeError("No successful GE analysis")
        return self._analysis.plots

    @property
    def post_analysis_plots(self) -> Plots:
        if self._post_analysis is None:
            raise RuntimeError("No successful GE post analysis")
        return self._post_analysis.plots

    def run(self, soc: Any, soccfg: Any, cfg: GE_Cfg) -> GE_Result:
        plots = Plots(self._host)
        try:
            result = self._core.run(cfg, context=QickContext(soc, soccfg, plots))
            plots.finish()
        except BaseException:
            try:
                plots.finish(present=False)
            finally:
                plots.release()
            raise
        self.last_result = result
        self.run_plots = plots
        self._analysis = None
        self._post_analysis = None
        return result

    def analyze(  # noqa: PLR0913 - Notebook flattens confirmed GE FIT options
        self,
        result: GE_Result | None = None,
        *,
        initial_state: Literal["ground", "excited"] = GE_ANALYZE_DEFAULTS.initial_state,
        backend: Literal["pca", "center"] = GE_ANALYZE_DEFAULTS.backend,
        logscale: bool = GE_ANALYZE_DEFAULTS.logscale,
        align_t1: bool = GE_ANALYZE_DEFAULTS.align_t1,
        length_ratio: float | None = GE_ANALYZE_DEFAULTS.length_ratio,
        angle: float | None = GE_ANALYZE_DEFAULTS.angle,
    ) -> GEAnalysis:
        source = self.last_result if result is None else result
        if source is None:
            raise ValueError("No GE result to analyze")
        options = GEAnalyzeOptions(
            initial_state=initial_state,
            backend=backend,
            logscale=logscale,
            align_t1=align_t1,
            length_ratio=length_ratio,
            angle=angle,
        )
        plots = Plots(self._host)
        try:
            analysis = self._core.analyze(source, options, plots=plots)
            plots.finish()
        except BaseException:
            try:
                plots.finish(present=False)
            finally:
                plots.release()
            raise
        self._analysis = GEAnalysisRecord(source, options, analysis, plots)
        self._post_analysis = None
        return analysis

    def post_analyze(
        self,
        *,
        radius: float | None = GE_POST_ANALYZE_DEFAULTS.radius,
        consider_other: bool = GE_POST_ANALYZE_DEFAULTS.consider_other,
    ) -> GEPostAnalysis:
        primary_record = self._analysis
        if primary_record is None:
            raise ValueError("No GE analysis to post analyze")
        options = GEPostAnalyzeOptions(radius=radius, consider_other=consider_other)
        plots = Plots(self._host)
        try:
            analysis = self._core.post_analyze(
                primary_record.source, primary_record.result, options, plots=plots
            )
            plots.finish()
        except BaseException:
            try:
                plots.finish(present=False)
            finally:
                plots.release()
            raise
        self._post_analysis = GEPostAnalysisRecord(
            primary_record.source,
            primary_record.result,
            primary_record.options,
            options,
            analysis,
            plots,
        )
        return analysis

    def save(
        self,
        filepath: str | Path,
        result: GE_Result | None = None,
        comment: str | None = None,
        tag: str | None = None,
        *,
        server_ip: str | None = None,
        port: int = 4999,
    ) -> None:
        selected = self.last_result if result is None else result
        if selected is None:
            raise ValueError("No GE result to save")
        self._core.save(
            selected,
            Path(filepath),
            comment=comment,
            tag=tag,
            server_ip=server_ip,
            port=port,
        )

    def load(
        self,
        filepath: str | Path,
        *,
        server_ip: str | None = None,
        port: int = 4999,
    ) -> GE_Result:
        result = self._core.load(Path(filepath), server_ip=server_ip, port=port)
        self.last_result = result
        self.run_plots = None
        self._analysis = None
        self._post_analysis = None
        return result
