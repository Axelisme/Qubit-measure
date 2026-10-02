"""Typed Notebook records and operation ownership over an experiment instance."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Generic, TypeVar

from zcu_tools.datafile import format_ext, reserve_labber_filepath
from zcu_tools.device.base import BaseDevice
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.interfaces import RecordExperiment, SynchronousExperiment
from zcu_tools.experiment.records import AnalysisRecord, RunRecord
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.notebook.plotting import NotebookPlotHost, finish_failed_plots
from zcu_tools.plotting.plots import PlotHost, Plots

CoreT = TypeVar("CoreT", bound=RecordExperiment[Any, Any], covariant=True)
CfgT = TypeVar("CfgT", bound=ExpCfgModel)
ResultT = TypeVar("ResultT")
OptionsT = TypeVar("OptionsT")
AnalysisT = TypeVar("AnalysisT")


class NotebookAdapter(Generic[CoreT]):
    """Bind handles and presentation while retaining only successful records.

    Analyze is statically available only for SynchronousExperiment instances.
    Load and analysis work without hardware. Run requires handles and devices before
    starting an operation. Each operation owns fresh plots; retaining a prior
    record or presentation handle does not let later operations close its figures.
    """

    def __init__(
        self,
        experiment: CoreT,
        *,
        soc: Any = None,
        soccfg: Any = None,
        devices: Mapping[str, BaseDevice[Any]] | None = None,
        host: PlotHost | None = None,
    ) -> None:
        self._core = experiment
        self._soc = soc
        self._soccfg = soccfg
        self._devices = devices
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
        if self._soc is None or self._soccfg is None:
            raise ValueError("Run requires both soc and soccfg handles")
        if self._devices is None:
            raise ValueError(
                "Run requires an explicit devices mapping; use {} for no devices"
            )
        retained_cfg = deepcopy(cfg)
        plots = Plots(self._host)
        context = RunContext(
            soc=self._soc,
            soccfg=self._soccfg,
            plots=plots,
            devices=self._devices,
            cancel_signal=StopSignal(),
        )
        try:
            result = self._core.run(deepcopy(retained_cfg), context=context)
            context.cancel_signal.raise_if_error()
            plots.finish()
            record = RunRecord(cfg=retained_cfg, result=result)
        except BaseException as error:
            finish_failed_plots(plots, error, operation="Run")
            raise
        self._last_run = record
        self._analysis = None
        self.run_presentation = plots
        self.analysis_presentation = None
        return record

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
        try:
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
        except BaseException as error:
            finish_failed_plots(plots, error, operation="Analysis")
            raise
        self._analysis = record
        self.analysis_presentation = plots
        return record

    def load(
        self: NotebookAdapter[RecordExperiment[CfgT, ResultT]],
        source: Path,
    ) -> RunRecord[CfgT, ResultT]:
        record = self._core.load(source)
        self._last_run = record
        self._analysis = None
        self.run_presentation = None
        self.analysis_presentation = None
        return record

    def save(
        self: NotebookAdapter[RecordExperiment[CfgT, ResultT]],
        source: RunRecord[CfgT, ResultT],
        destination: Path,
        *,
        unique: bool = False,
        comment: str | None = None,
        tag: str | None = None,
    ) -> Path:
        path = Path(
            reserve_labber_filepath(str(destination))
            if unique
            else format_ext(str(destination))
        )
        self._core.save(
            source,
            path,
            comment=comment,
            tag=tag,
        )
        return path
