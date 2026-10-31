"""Typed Notebook records and operation ownership over an experiment instance."""

from __future__ import annotations

import _thread
import signal
import threading
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from types import FrameType, TracebackType
from typing import Any, Generic, Self, TypeVar

from matplotlib.figure import Figure

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import format_ext, reserve_labber_filepath
from zcu_tools.device import DeviceManager
from zcu_tools.experiment.acquisition import (
    AcquisitionEvidence,
    capture_device_snapshot,
    capture_software_provenance,
    new_run_id,
)
from zcu_tools.experiment.artifacts import AnalysisDocument
from zcu_tools.experiment.axes_spec import AxesSpec, GroupedAxesSpec
from zcu_tools.experiment.base import Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.native_persistence import load_run_with_metadata, save_labber
from zcu_tools.experiment.records import AnalysisRecord, RunRecord
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.notebook.plotting import NotebookPlotHost, finish_failed_plots
from zcu_tools.notebook.saving import SavedRun
from zcu_tools.plotting.plots import PlotHost, Plots
from zcu_tools.resources.entry import PointView, RoleView

CoreT = TypeVar("CoreT", bound=Experiment[Any, Any], covariant=True)
CfgT = TypeVar("CfgT", bound=ConfigBase)
ResultT = TypeVar("ResultT")
OptionsT = TypeVar("OptionsT")


def _validate_interrupt_timeout(seconds: float) -> None:
    if not isfinite(seconds) or seconds <= 0:
        raise ValueError("interrupt_timeout must be finite and positive (seconds)")


class _RunInterrupts:
    """Own the temporary main-thread handler and the frontend stop-request fact."""

    def __init__(self, stop: StopSignal, timeout: float) -> None:
        self.stop_requested = False
        self._stop = stop
        self._timeout = timeout
        self._previous = signal.getsignal(signal.SIGINT)
        self._main_thread = threading.current_thread() is threading.main_thread()
        self._thread_id = threading.get_ident()
        self._active = False
        self._timer: threading.Timer | None = None

    def __enter__(self) -> Self:
        if self._main_thread:
            self._active = True
            signal.signal(signal.SIGINT, self._handle)
        return self

    def _handle(self, _signum: int, _frame: FrameType | None) -> None:
        if not self._active:
            return
        if self.stop_requested:
            raise KeyboardInterrupt
        self.stop_requested = True
        self._stop.set(reason="Notebook SIGINT")
        self._timer = threading.Timer(self._timeout, self._interrupt_main)
        self._timer.daemon = True
        self._timer.start()

    def _interrupt_main(self) -> None:
        if self._active:
            if hasattr(signal, "pthread_kill"):
                # Target the main thread to interrupt blocking POSIX C calls.
                signal.pthread_kill(self._thread_id, signal.SIGINT)
            else:
                _thread.interrupt_main()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._active = False
        try:
            if self._timer is not None:
                self._timer.cancel()
                # Drain an already-running callback before returning the handler.
                self._timer.join()
        finally:
            if self._main_thread:
                signal.signal(signal.SIGINT, self._previous)


class NotebookAdapter:
    """Bind an environment once, then wrap independent experiment instances.

    Construction only retains references. Each run takes one current driver
    mapping from the bound manager; it does not refresh cfg or reconnect devices.
    Reassigning a caller's variables does not rebind this environment.
    interrupt_timeout is the positive, finite SIGINT grace period in seconds
    (default 10.0). Invalid values raise ValueError at construction.
    """

    def __init__(
        self,
        *,
        soc: Any = None,
        soccfg: Any = None,
        device_manager: DeviceManager | None = None,
        host: PlotHost | None = None,
        interrupt_timeout: float = 10.0,
    ) -> None:
        _validate_interrupt_timeout(interrupt_timeout)
        self._interrupt_timeout = interrupt_timeout
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
            interrupt_timeout=self._interrupt_timeout,
        )


class NotebookExperiment(Generic[CoreT]):
    """Retain successful records for one experiment in a bound environment.

    Analyze delegates to the experiment's optional synchronous operation.
    Load and analysis work without hardware. Run requires handles and a manager
    before starting an operation. Run and analyze own fresh plots; retaining a
    prior record does not let later operations close its figures.
    interrupt_timeout is the positive, finite SIGINT grace period in seconds
    (default 10.0); invalid values raise ValueError at construction.
    """

    def __init__(
        self,
        experiment: CoreT,
        *,
        soc: Any = None,
        soccfg: Any = None,
        device_manager: DeviceManager | None = None,
        host: PlotHost | None = None,
        interrupt_timeout: float = 10.0,
    ) -> None:
        _validate_interrupt_timeout(interrupt_timeout)
        self._interrupt_timeout = interrupt_timeout
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
        self: NotebookExperiment[Experiment[ResultT, CfgT]],
    ) -> RunRecord[CfgT, ResultT] | None:
        return self._last_run

    @property
    def analysis(
        self: NotebookExperiment[Experiment[ResultT, CfgT]],
    ) -> AnalysisRecord[CfgT, ResultT, Any, Any] | None:
        return self._analysis

    def acquisition_for(
        self: NotebookExperiment[Experiment[ResultT, CfgT]],
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
        self: NotebookExperiment[Experiment[ResultT, CfgT]],
        cfg: CfgT,
    ) -> RunRecord[CfgT, ResultT]:
        """Acquire a copied cfg and publish its record/evidence after plots finish.

        Require soc, soccfg and a DeviceManager before starting. Capture
        every connected driver's get_info once after RunContext construction,
        before core.run, without a lease (resources belong to the synchronous
        caller). Propagate any read error before writes/acquisition; no cached
        fallback. Retain that pre-Run snapshot unchanged through sweeps and Save.
        Capture software-only provenance before acquisition; propagate errors.
        Time core.run in UTC, excluding plot finalization. On the main thread,
        temporarily handle SIGINT during core.run: the first interrupt requests
        stop with a reason, the second or interrupt_timeout deadline raises
        KeyboardInterrupt. Cancel the timer before restoring the prior handler.
        Worker-thread runs leave signal handling unchanged. Publish every normal
        return; completion is stopped only if this adapter sent a stop request.
        Propagate all escaping exceptions unchanged, including KeyboardInterrupt.
        Any acquisition or presentation failure retains the previous record,
        analysis and acquisition evidence.
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
            scope=Scope("run"),
        )
        try:
            working_cfg = deepcopy(retained_cfg)
            device_snapshot = capture_device_snapshot(context.devices)
            provenance = capture_software_provenance(context=context)
            provenance = replace(
                provenance, software_versions=dict(provenance.software_versions)
            )
            started_at = datetime.now(UTC)
            run_id = new_run_id(started_at)
            with _RunInterrupts(context.scope.stop, self._interrupt_timeout) as interrupts:
                result = self._core.run(working_cfg, context=context)
            finished_at = datetime.now(UTC)
            evidence = AcquisitionEvidence(
                run_id=run_id,
                started_at=started_at,
                finished_at=finished_at,
                completion=(
                    "stopped" if interrupts.stop_requested else "complete"
                ),
                provenance=provenance,
                device_snapshot=device_snapshot,
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
        self: NotebookExperiment[Experiment[ResultT, CfgT]],
        options: OptionsT,
        *,
        source: RunRecord[CfgT, ResultT] | None = None,
    ) -> AnalysisRecord[CfgT, ResultT, OptionsT, Any]:
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
        self: NotebookExperiment[Experiment[ResultT, CfgT]],
        source: Path,
    ) -> RunRecord[CfgT, ResultT]:
        """Load a native Path through the core and publish only after success.

        No hardware is required. Return the core RunRecord, including cfg=None
        and its warning when historical cfg is incompatible. Successful load
        clears analysis, presentation references and acquisition evidence;
        failure retains all prior results. Labber-only files must be converted
        offline with the migrations tools before using the default core loader.
        """
        record = load_run_with_metadata(source, spec=self._declared_spec())
        self._last_run = record
        self._acquisition_evidence = None
        self._analysis = None
        self.run_presentation = None
        self.analysis_presentation = None
        return record

    def _declared_spec(
        self: NotebookExperiment[Experiment[ResultT, CfgT]],
    ) -> AxesSpec[ResultT, CfgT] | GroupedAxesSpec[ResultT, CfgT]:
        spec = self._core.AXES_SPEC
        if spec is None:
            raise NotImplementedError(f"{type(self._core).__name__} has no AXES_SPEC")
        return spec

    def save_run(
        self: NotebookExperiment[Experiment[ResultT, CfgT]],
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
        roles must be resolved from the same bound entry, point and storage
        roots as ctx; another handle of that point is valid. None records no
        roles. Wrong role origins raise ValueError before outputs or
        acquired/saved events. Named figures create PNGs/copies, analysis supplies
        JSON-safe pane results, and comment remains raw user text. A returned SavedRun can
        explicitly rewrite the same run after later adapter operations.
        core.AXES_SPEC.save_stem receives captured SaveContext on each save.
        Its result is used verbatim. The default joins dotted tag, roles and
        the complete run_id.
        Callback failures or invalid stems propagate; no prefix is appended.
        PartialSaveError retains that handle and latest output report; ledger
        errors propagate directly. Do not retry or guess a historical source.
        """
        saved = SavedRun(
            ctx,
            source=source,
            spec=self._declared_spec(),
            acquisition=self.acquisition_for(source),
            roles=roles,
        )
        saved.save(figures=figures, analysis=analysis, comment=comment)
        return saved

    def save(
        self: NotebookExperiment[Experiment[ResultT, CfgT]],
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
        save_labber(
            selected,
            path,
            spec=self._declared_spec(),
            comment=comment,
            tag=tag,
        )
        return path
