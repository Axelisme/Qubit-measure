"""Typed Notebook records and operation ownership over an experiment instance."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Generic, TypeVar

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.interfaces import RecordExperiment, SynchronousExperiment
from zcu_tools.experiment.records import AnalysisRecord, RunRecord
from zcu_tools.notebook.plotting import NotebookPlotHost
from zcu_tools.plotting.plots import PlotHost, Plots

CoreT = TypeVar("CoreT", bound=RecordExperiment[Any, Any], covariant=True)
CfgT = TypeVar("CfgT", bound=ExpCfgModel)
ResultT = TypeVar("ResultT")
OptionsT = TypeVar("OptionsT")
AnalysisT = TypeVar("AnalysisT")


class NotebookAdapter(Generic[CoreT]):
    """Bind handles and presentation while retaining only successful records.

    Analyze is statically available only for SynchronousExperiment instances.
    Load and analysis work without hardware. Run requires both handles before
    starting an operation. Each operation owns fresh plots; retaining a prior
    record or presentation handle does not let later operations close its figures.
    """

    def __init__(
        self,
        experiment: CoreT,
        *,
        soc: Any = None,
        soccfg: Any = None,
        host: PlotHost | None = None,
    ) -> None:
        self._core = experiment
        self._soc = soc
        self._soccfg = soccfg
        self._host = NotebookPlotHost() if host is None else host
        self._last_run: RunRecord[Any, Any] | None = None
        self._analysis: AnalysisRecord[Any, Any, Any, Any] | None = None
        self.run_presentation: Plots | None = None
        self.analysis_presentation: Plots | None = None

    @property
    def last_run(
        self: NotebookAdapter[RecordExperiment[CfgT, ResultT]],
    ) -> RunRecord[CfgT, ResultT] | None:
        return self._last_run

    @property
    def analysis(
        self: NotebookAdapter[
            SynchronousExperiment[CfgT, ResultT, OptionsT, AnalysisT]
        ],
    ) -> AnalysisRecord[CfgT, ResultT, OptionsT, AnalysisT] | None:
        return self._analysis

    def run(
        self: NotebookAdapter[RecordExperiment[CfgT, ResultT]],
        cfg: CfgT,
    ) -> RunRecord[CfgT, ResultT]:
        raise NotImplementedError("Notebook record acquisition is not implemented")

    def analyze(
        self: NotebookAdapter[
            SynchronousExperiment[CfgT, ResultT, OptionsT, AnalysisT]
        ],
        options: OptionsT,
        *,
        source: RunRecord[CfgT, ResultT] | None = None,
    ) -> AnalysisRecord[CfgT, ResultT, OptionsT, AnalysisT]:
        selected = self.last_run if source is None else source
        if selected is None:
            raise ValueError("No run record to analyze")
        retained_options = deepcopy(options)
        working_source = RunRecord(cfg=selected.cfg, result=selected.result)
        plots = Plots(self._host)
        result = self._core.analyze(
            working_source,
            deepcopy(retained_options),
            plots=plots,
        )
        figures = plots.finish()
        record = AnalysisRecord(
            source=selected,
            options=retained_options,
            result=result,
            figures=figures,
        )
        self._analysis = record
        self.analysis_presentation = plots
        return record

    def load(
        self: NotebookAdapter[RecordExperiment[CfgT, ResultT]],
        source: Path,
        *,
        server_ip: str | None = None,
        port: int = 4999,
    ) -> RunRecord[CfgT, ResultT]:
        record = self._core.load(source, server_ip=server_ip, port=port)
        self._last_run = record
        self._analysis = None
        self.run_presentation = None
        self.analysis_presentation = None
        return record

    def save(  # noqa: PLR0913 - explicit uniqueness plus core persistence options
        self: NotebookAdapter[RecordExperiment[CfgT, ResultT]],
        source: RunRecord[CfgT, ResultT],
        destination: Path,
        *,
        unique: bool = False,
        comment: str | None = None,
        tag: str | None = None,
        server_ip: str | None = None,
        port: int = 4999,
    ) -> Path:
        raise NotImplementedError("Notebook explicit record saving is not implemented")
