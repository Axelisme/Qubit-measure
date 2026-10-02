"""Typed Notebook records and operation ownership over an experiment instance."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Generic, TypeVar

from zcu_tools.datafile import format_ext, reserve_labber_filepath
from zcu_tools.device import DeviceManager
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


class NotebookAdapter:
    """Bind an environment once, then wrap independent experiment instances.

    Construction only retains references. Each run takes one current driver
    mapping from the bound manager; it does not refresh cfg or reconnect devices.
    Reassigning a caller's variables does not rebind this environment.
    """

    def __init__(
        self,
        *,
        soc: Any = None,
        soccfg: Any = None,
        device_manager: DeviceManager | None = None,
        host: PlotHost | None = None,
    ) -> None:
        self._soc = soc
        self._soccfg = soccfg
        self._device_manager = device_manager
        self._host = NotebookPlotHost() if host is None else host

    def __call__(self, experiment: CoreT) -> NotebookExperiment[CoreT]:
        return NotebookExperiment(
            experiment,
            soc=self._soc,
            soccfg=self._soccfg,
            device_manager=self._device_manager,
            host=self._host,
        )


class NotebookExperiment(Generic[CoreT]):
    """Retain successful records for one experiment in a bound environment.

    Analyze is statically available only for SynchronousExperiment instances.
    Load and analysis work without hardware. Run requires handles and a manager
    before starting an operation. Run and analyze own fresh plots; retaining a
    prior record does not let later operations close its figures.
    """

    def __init__(
        self,
        experiment: CoreT,
        *,
        soc: Any = None,
        soccfg: Any = None,
        device_manager: DeviceManager | None = None,
        host: PlotHost | None = None,
    ) -> None:
        self._core = experiment
        self._soc = soc
        self._soccfg = soccfg
        self._device_manager = device_manager
        self._host = NotebookPlotHost() if host is None else host
        self._last_run: RunRecord[Any, Any] | None = None
        self._analysis: AnalysisRecord[Any, Any, Any, Any] | None = None
        self.run_presentation: Plots | None = None
        self.analysis_presentation: Plots | None = None

    @property
    def last_run(
        self: NotebookExperiment[RecordExperiment[CfgT, ResultT]],
    ) -> RunRecord[CfgT, ResultT] | None:
        return self._last_run

    @property
    def analysis(
        self: NotebookExperiment[
            SynchronousExperiment[CfgT, ResultT, OptionsT, AnalysisT]
        ],
    ) -> AnalysisRecord[CfgT, ResultT, OptionsT, AnalysisT] | None:
        return self._analysis

    def run(
        self: NotebookExperiment[RecordExperiment[CfgT, ResultT]],
        cfg: CfgT,
    ) -> RunRecord[CfgT, ResultT]:
        if self._soc is None or self._soccfg is None:
            raise ValueError("Run requires both soc and soccfg handles")
        if self._device_manager is None:
            raise ValueError(
                "Run requires an explicit device manager; use DeviceManager() for no devices"
            )
        devices = self._device_manager.get_all_devices()
        retained_cfg = deepcopy(cfg)
        plots = Plots(self._host)
        context = RunContext(
            soc=self._soc,
            soccfg=self._soccfg,
            plots=plots,
            devices=devices,
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
        self: NotebookExperiment[
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
        self: NotebookExperiment[RecordExperiment[CfgT, ResultT]],
        source: Path,
    ) -> RunRecord[CfgT, ResultT]:
        record = self._core.load(source)
        self._last_run = record
        self._analysis = None
        self.run_presentation = None
        self.analysis_presentation = None
        return record

    def save(
        self: NotebookExperiment[RecordExperiment[CfgT, ResultT]],
        destination: str | Path,
        *,
        source: RunRecord[CfgT, ResultT] | None = None,
        unique: bool = False,
        comment: str | None = None,
        tag: str | None = None,
    ) -> Path:
        selected = self.last_run if source is None else source
        if selected is None:
            raise ValueError("No run record to save")
        path = Path(
            reserve_labber_filepath(str(destination))
            if unique
            else format_ext(str(destination))
        )
        self._core.save(
            selected,
            path,
            comment=comment,
            tag=tag,
        )
        return path
