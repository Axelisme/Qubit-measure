"""Notebook convenience entry for the ordinary T1 experiment."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zcu_tools.experiment.context import QickContext
from zcu_tools.experiment.v2.twotone.time_domain.t1 import (
    T1_ANALYZE_DEFAULTS,
    T1Analysis,
    T1AnalyzeOptions,
    T1Cfg,
    T1Result,
)
from zcu_tools.experiment.v2.twotone.time_domain.t1 import T1Exp as T1Core
from zcu_tools.notebook.plotting import NotebookPlotHost
from zcu_tools.plotting.plots import NonPresentingHost, Plots


@dataclass(frozen=True)
class T1AnalysisRecord:
    source: T1Result
    options: T1AnalyzeOptions
    result: T1Analysis
    plots: Plots


class T1Exp:
    """Retain the latest successful Result and analysis for Notebook use.

    Each operation owns fresh plots. Replacing references does not close older
    figures: retain their Plots and call release() explicitly to close widgets
    while keeping the Figure objects saveable. Failed operations release only
    their own presentation and leave the previous successful record unchanged.
    """

    def __init__(self, *, present: bool = True) -> None:
        self._core = T1Core()
        self._host = NotebookPlotHost() if present else NonPresentingHost()
        self.last_result: T1Result | None = None
        self.run_plots: Plots | None = None
        self._analysis: T1AnalysisRecord | None = None

    @property
    def analysis(self) -> T1AnalysisRecord | None:
        return self._analysis

    @property
    def analysis_plots(self) -> Plots:
        if self._analysis is None:
            raise RuntimeError("No successful T1 analysis")
        return self._analysis.plots

    def run(self, soc: Any, soccfg: Any, cfg: T1Cfg) -> T1Result:
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
        return result

    def analyze(
        self,
        result: T1Result | None = None,
        *,
        dual_exp: bool = T1_ANALYZE_DEFAULTS.dual_exp,
        skip: int = T1_ANALYZE_DEFAULTS.skip,
    ) -> T1Analysis:
        source = self.last_result if result is None else result
        if source is None:
            raise ValueError("No T1 result to analyze")
        options = T1AnalyzeOptions(dual_exp=dual_exp, skip=skip)
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
        self._analysis = T1AnalysisRecord(source, options, analysis, plots)
        return analysis

    def save(
        self,
        filepath: str | Path,
        result: T1Result | None = None,
        comment: str | None = None,
        tag: str | None = None,
        *,
        server_ip: str | None = None,
        port: int = 4999,
    ) -> None:
        selected = self.last_result if result is None else result
        if selected is None:
            raise ValueError("No T1 result to save")
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
    ) -> T1Result:
        result = self._core.load(Path(filepath), server_ip=server_ip, port=port)
        self.last_result = result
        self.run_plots = None
        self._analysis = None
        return result
