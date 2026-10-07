"""Typed Notebook records and operation ownership over an experiment instance."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Generic, TypeVar

from matplotlib.figure import Figure

from zcu_tools.datafile import AnalysisDocument, format_ext, reserve_labber_filepath
from zcu_tools.device import DeviceManager
from zcu_tools.experiment.acquisition import (
    AcquisitionEvidence,
    capture_software_provenance,
    new_run_id,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.interfaces import (
    PersistableRecordExperiment,
    RecordExperiment,
    SynchronousExperiment,
)
from zcu_tools.experiment.records import AnalysisRecord, RunRecord
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.notebook.plotting import NotebookPlotHost, finish_failed_plots
from zcu_tools.notebook.saving import SavedRun
from zcu_tools.plotting.plots import PlotHost, Plots
from zcu_tools.resources.entry import PointView, RoleView

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
        self._acquisition_evidence: AcquisitionEvidence | None = None
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

    def acquisition_for(
        self: NotebookExperiment[RecordExperiment[CfgT, ResultT]],
        source: RunRecord[CfgT, ResultT],
    ) -> AcquisitionEvidence:
        """Return isolated facts for this adapter's most recent successful run.

        source must be that exact RunRecord object, not a value-equal record,
        another adapter's record, a stale run or loaded data. Raise ValueError
        before any I/O when there is no matching acquisition. Analysis and
        failed operations retain evidence; the next successful run replaces it
        and successful load clears it. Returned version mappings are detached.
        """
        evidence = self._acquisition_evidence
        if source is not self._last_run or evidence is None:
            raise ValueError(
                "No acquisition evidence for this source; use this adapter's latest run"
            )
        return replace(evidence)

    def run(
        self: NotebookExperiment[RecordExperiment[CfgT, ResultT]],
        cfg: CfgT,
    ) -> RunRecord[CfgT, ResultT]:
        """Acquire a copied cfg and publish its record/evidence after plots finish.

        Require soc, soccfg and a DeviceManager before starting. Capture
        software-only provenance before acquisition; propagate capture errors.
        Time core.run and its error check in UTC, excluding plot finalization.
        Normal stopped or interrupted returns publish stopped evidence and a
        saveable record. Failed signals and exceptions escaping core.run raise.
        This includes KeyboardInterrupt before a Schedule scope (device setup)
        or after it. Any acquisition or presentation failure
        retains the previous record, analysis and acquisition evidence.
        """
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
            working_cfg = deepcopy(retained_cfg)
            provenance = capture_software_provenance(context=context)
            provenance = replace(
                provenance, software_versions=dict(provenance.software_versions)
            )
            started_at = datetime.now(UTC)
            run_id = new_run_id(started_at)
            result = self._core.run(working_cfg, context=context)
            error = context.cancel_signal.error
            if error is not None and error.status != "interrupted":
                context.cancel_signal.raise_if_error()
            finished_at = datetime.now(UTC)
            evidence = AcquisitionEvidence(
                run_id=run_id,
                started_at=started_at,
                finished_at=finished_at,
                completion="stopped" if context.cancel_signal.is_set() else "complete",
                provenance=provenance,
            )
            plots.finish()
            record = RunRecord(cfg=retained_cfg, result=result)
        except BaseException as error:
            finish_failed_plots(plots, error, operation="Run")
            raise
        self._last_run = record
        self._acquisition_evidence = evidence
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
        self._acquisition_evidence = None
        self._analysis = None
        self.run_presentation = None
        self.analysis_presentation = None
        return record

    def save_run(
        self: NotebookExperiment[PersistableRecordExperiment[CfgT, ResultT]],
        ctx: PointView,
        *,
        source: RunRecord[CfgT, ResultT],
        roles: RoleView | None = None,
        figures: Mapping[str, Figure] | None = None,
        analysis: tuple[AnalysisDocument, ...] = (),
        comment: str = "",
    ) -> SavedRun[CfgT, ResultT]:
        """Save this adapter's latest acquired record into a bound point's run.

        ctx provides cached save-time context and exact entry storage. source
        must be the same object returned by the most recent successful run;
        loaded, stale, manual and other-adapter records fail before output I/O.
        roles, if provided, must come from ctx.resolve; None records no roles.
        Named figures create PNGs/copies, analysis supplies JSON-safe pane
        results, and comment remains raw user text. A returned SavedRun can
        explicitly rewrite the same run after later adapter operations.
        PartialSaveError retains that handle and latest output report; ledger
        errors propagate directly. Do not retry or guess a historical source.
        """
        saved = SavedRun(
            ctx,
            source=source,
            spec=self._core.persistence_spec(),
            acquisition=self.acquisition_for(source),
            roles=roles,
        )
        saved.save(figures=figures, analysis=analysis, comment=comment)
        return saved

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
