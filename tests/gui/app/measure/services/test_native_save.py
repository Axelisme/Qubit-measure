"""Native measurement-data Save through real state, workers, writers and ledger."""

import json
import threading
from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock

import numpy as np
import pytest
from qtpy.QtCore import QEventLoop, QTimer
from qtpy.QtWidgets import QApplication
from zcu_tools.datafile import (
    CfgSnapshot,
    ExperimentPayload,
    NativeExtensions,
    RunMetadata,
    RunSnapshot,
    SoftwareProvenance,
    load_run_data,
)
from zcu_tools.experiment.acquisition import new_run_id
from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.run_saving import ArtifactRef
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    AnalyzeRequest,
    PostAnalyzeRequest,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.adapter.types import (
    AnalyzeResultBase,
    PostAnalyzeResultBase,
)
from zcu_tools.gui.app.measure.artifact_tracker import SaveStatus
from zcu_tools.gui.app.measure.events.completion import (
    SaveArtifactsFinishedPayload,
    SaveDataFinishedPayload,
)
from zcu_tools.gui.app.measure.events.run import RunFinishedPayload
from zcu_tools.gui.app.measure.services.analyze import AnalyzeService
from zcu_tools.gui.app.measure.services.guard import GuardService
from zcu_tools.gui.app.measure.services.load import LoadService
from zcu_tools.gui.app.measure.services.operation_gate import OperationGate
from zcu_tools.gui.app.measure.services.post_analyze import PostAnalyzeService
from zcu_tools.gui.app.measure.services.run import RunService
from zcu_tools.gui.app.measure.services.run_analyze_control import (
    RunAnalyzeControlFacet,
)
from zcu_tools.gui.app.measure.services.save import SaveService
from zcu_tools.gui.app.measure.services.save_control import SaveControlFacet
from zcu_tools.gui.app.measure.services.tab import TabService
from zcu_tools.gui.app.measure.state import Session, State
from zcu_tools.gui.cfg import CfgSchema, CfgSectionSpec, CfgSectionValue
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import CommandOrigin, PointUseRequest
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.entry import ResultEntry, SavedPayload, SaveLayout
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult
from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui.app.measure._cfg_fakes import make_cfg
from tests.gui.app.measure.adapter._cfg_fakes import StaticCfgDefinition

GUI_ORIGIN = CommandOrigin("gui", None)


@dataclass
class PaneParams:
    gain: float = 2.0


@dataclass
class PaneSummary(AnalyzeResultBase):
    mean: float = 3.0


@dataclass
class PostSummary(PostAnalyzeResultBase):
    correction: float = 0.5


class SavingAdapter(
    BaseAdapter[SavingCfg, RunRecord[SavingCfg, SavingResult], PaneSummary, PaneParams]
):
    """Real BaseAdapter mapping with an offline typed experiment declaration."""

    exp_cls = SavingCore
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE, load_data=False, requires_soc=False
    )
    experiment_tag: ClassVar[str | None] = SavingCore().persistence_spec().tag

    @classmethod
    def cfg_definition(cls) -> StaticCfgDefinition:
        return StaticCfgDefinition()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return "saving"

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> RunRecord[SavingCfg, SavingResult]:
        """Acquire a typed record using only the offline fake backend."""
        cfg = SavingCfg.model_validate(raw_cfg)
        return RunRecord(cfg, SavingCore().run(cfg, context=context))


class AnalysisSavingAdapter(SavingAdapter):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT,
        post_analysis=True,
        load_data=True,
        requires_soc=False,
    )

    def analyze(
        self,
        req: AnalyzeRequest[RunRecord[SavingCfg, SavingResult], PaneParams],
        *,
        plots: Plots,
    ) -> PaneSummary:
        return PaneSummary()

    def get_post_analyze_params(
        self, analyze_result: PaneSummary, ctx: SessionEnv
    ) -> PaneParams:
        return PaneParams()

    def post_analyze(
        self,
        req: PostAnalyzeRequest[
            RunRecord[SavingCfg, SavingResult], PaneSummary, PaneParams
        ],
        *,
        plots: Plots,
    ) -> PostSummary:
        return PostSummary()


@dataclass
class NativeSaveSession:
    entry: ResultEntry
    context: ContextService
    state: State
    control: SaveControlFacet
    handles: OperationHandles
    bus: BaseEventBus
    metadata: RunMetadata
    completions: list[SaveDataFinishedPayload]
    artifact_completions: list[SaveArtifactsFinishedPayload]
    background: BackgroundRunner

    def save(self, *, origin: CommandOrigin = GUI_ORIGIN) -> SaveDataFinishedPayload:
        """Wait for the public completion fact, never inspecting private workers."""
        loop = QEventLoop()
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        subscription = self.bus.subscribe(
            SaveDataFinishedPayload, lambda _payload: loop.quit()
        )
        before = len(self.completions)
        try:
            submission = self.control.save_data(
                "tab", comment="raw annotation", origin=origin
            )
            assert self.state.is_tab_busy("tab")
            with pytest.raises(FailedPreconditionError, match="busy"):
                self.control.save_data("tab", origin=origin)
            assert Path(submission.data_path) == (
                self.entry.database_path / "runs" / self.metadata.run_id / "data.h5"
            )
            timer.start(5000)
            loop.exec()
        finally:
            timer.stop()
            subscription.unsubscribe()
        assert len(self.completions) == before + 1
        outcome = self.handles.known_outcome(submission.operation_id)
        assert outcome is not None
        completed = self.completions[-1]
        assert outcome.status == ("failed" if completed.error else "finished")
        assert not self.state.is_tab_busy("tab")
        return completed

    def save_artifacts(
        self, refs: tuple[ArtifactRef, ...] | None = None
    ) -> SaveArtifactsFinishedPayload:
        """Wait at the batch terminal seam and inspect the public settled handle."""
        loop = QEventLoop()
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        before = len(self.artifact_completions)
        observed: list[tuple[bool, bool, bool]] = []

        def terminal(payload: SaveArtifactsFinishedPayload) -> None:
            observed.append(
                (
                    not self.state.is_tab_busy("tab"),
                    self.handles.known_outcome(submission.operation_id) is not None,
                    payload.report is None
                    or all(
                        any(
                            member.key == item.key
                            and any(
                                output.path == item.output.path
                                and output.status == item.status
                                and output.error == item.error
                                for output in member.outputs
                            )
                            for section in self.state.get_artifact_snapshots("tab")
                            for artifact in section.artifacts
                            for member in artifact.members
                        )
                        for item in payload.report.outputs
                    ),
                )
            )
            loop.quit()

        subscription = self.bus.subscribe(SaveArtifactsFinishedPayload, terminal)
        try:
            submission = self.control.save_artifacts(
                "tab", artifacts=refs, origin=GUI_ORIGIN
            )
            assert self.state.is_tab_busy("tab")
            with pytest.raises(FailedPreconditionError, match="busy"):
                self.control.save_artifacts("tab", origin=GUI_ORIGIN)
            with pytest.raises(FailedPreconditionError, match="busy"):
                self.control.set_comment("tab", "not admitted")
            timer.start(5000)
            loop.exec()
        finally:
            timer.stop()
            subscription.unsubscribe()
        assert len(self.artifact_completions) == before + 1
        assert observed == [(True, True, True)]
        completed = self.artifact_completions[-1]
        outcome = self.handles.known_outcome(submission.operation_id)
        assert outcome is not None
        assert outcome.status == ("failed" if completed.error else "finished")
        return completed


@pytest.fixture
def native_session(tmp_path: Path, qapp: QApplication) -> Generator[NativeSaveSession]:
    entry = ResultEntry.create(
        "original", result_root=tmp_path / "Result", database_root=tmp_path / "Database"
    )
    entry.new_point("p")
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    bus = BaseEventBus()
    context = ContextService(state, bus, bind_assets=ArbWaveformDatabase.info_at)
    context.install_entry(entry)
    context_control = ContextControlFacet(context=context)
    context_control.use_point(PointUseRequest(entry.entry_id, "p"))
    captured = context_control.read_context()
    storage = captured.capture_storage()
    cfg = SavingCfg()
    record = RunRecord(cfg, SavingCore().run(cfg, context=None))
    metadata = RunMetadata(
        run_id="20261008T000000Z-a1b2c3",
        experiment=SavingCore().persistence_spec().tag,
        started_at="2026-10-08T00:00:00+00:00",
        finished_at="2026-10-08T00:01:00+00:00",
        completion="complete",
        labber_path=None,
        snapshot=RunSnapshot(
            entry_id=storage.entry_id,
            entry_name=storage.entry_name,
            point=storage.point,
            description=storage.description,
            roles=storage.roles,
        ),
        provenance=SoftwareProvenance(
            software_versions={},
            git_commit=None,
            git_dirty=None,
            qick_version=None,
            soc_fingerprint=None,
            hostname=None,
        ),
    )
    state.add_tab(
        "tab",
        Session(
            adapter_name="saving",
            adapter=AnalysisSavingAdapter(),
            cfg=make_cfg(CfgSchema(CfgSectionSpec(), CfgSectionValue()), state=state),
            parameters=captured,
            save_context=captured,
        ),
    )
    state.update_tab_result("tab", record, metadata=metadata)
    handles = OperationHandles()
    background = BackgroundRunner()
    service = SaveService(
        state,
        OperationRunner(None, handles, None, background, bus),
        bus,
        owner_scheduler=QtOwnerScheduler(),
    )
    control = SaveControlFacet(state=state, guard=GuardService(state), save=service)
    completions: list[SaveDataFinishedPayload] = []
    subscription = bus.subscribe(SaveDataFinishedPayload, completions.append)
    artifact_completions: list[SaveArtifactsFinishedPayload] = []
    batch_subscription = bus.subscribe(
        SaveArtifactsFinishedPayload, artifact_completions.append
    )
    try:
        yield NativeSaveSession(
            entry,
            context,
            state,
            control,
            handles,
            bus,
            metadata,
            completions,
            artifact_completions,
            background,
        )
    finally:
        subscription.unsubscribe()
        batch_subscription.unsubscribe()
        background.quiesce()
        background.deleteLater()
        qapp.processEvents()


def test_data_save_writes_both_formats_and_resave_keeps_run_identity(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    first = session.save()
    assert first.error is None and first.report is not None
    native, labber = first.report.outputs
    assert native.status == labber.status == "saved"
    assert (
        native.output.path
        == session.entry.database_path / "runs" / session.metadata.run_id / "data.h5"
    )
    stored = load_run_data(native.output.path)
    assert stored.metadata.run_id == session.metadata.run_id
    assert stored.metadata.started_at == session.metadata.started_at
    assert stored.metadata.snapshot == session.metadata.snapshot
    variable = next(iter(stored.payload.variables.values()))
    np.testing.assert_allclose(variable.axes[0].values, [1e9, 1.001e9, 1.002e9])
    second = session.save(origin=CommandOrigin("mcp", "save-again"))
    assert second.error is None and second.report is not None
    assert second.report.run_id == first.report.run_id
    assert second.report.outputs[0].output.path == native.output.path
    assert second.report.outputs[1].output.path != labber.output.path
    events = session.entry.ledger.events()
    assert len(events) == 2
    assert [event.origin for event in events] == ["gui", "mcp"]
    assert events[-1].call_id == "save-again"
    assert isinstance(events[-1].payload, SavedPayload)
    assert [output.status for output in events[-1].payload.outputs] == [
        "saved",
        "saved",
    ]


@pytest.mark.parametrize("blocked", ["native", "labber"])
def test_each_output_failure_preserves_the_other_and_reports_both(
    native_session: NativeSaveSession, blocked: str
) -> None:
    session = native_session
    if blocked == "native":
        path = (
            session.entry.database_path / "runs" / session.metadata.run_id / "data.h5"
        )
        path.mkdir(parents=True)
    else:
        (session.entry.database_path / "Labber").write_text("not a directory")
    completed = session.save()
    assert completed.error is not None and completed.report is not None
    native, labber = completed.report.outputs
    assert [native.status, labber.status] == (
        ["failed", "saved"] if blocked == "native" else ["saved", "failed"]
    )
    failed = native if blocked == "native" else labber
    saved = labber if blocked == "native" else native
    assert saved.output.path.is_file()
    assert failed.error and str(failed.output.path) in completed.error
    event = session.entry.ledger.events()[-1]
    assert isinstance(event.payload, SavedPayload)
    assert [output.status for output in event.payload.outputs] == [
        native.status,
        labber.status,
    ]


def test_save_uses_tab_storage_after_live_entry_switch(
    native_session: NativeSaveSession, tmp_path: Path
) -> None:
    session = native_session
    other = ResultEntry.create(
        "other", result_root=tmp_path / "Result", database_root=tmp_path / "Database"
    )
    other.new_point("q")
    session.context.install_entry(other)
    ContextControlFacet(context=session.context).use_point(
        PointUseRequest(other.entry_id, "q")
    )
    completed = session.save()
    assert completed.error is None and completed.report is not None
    assert all(
        item.output.path.is_relative_to(session.entry.database_path)
        for item in completed.report.outputs
    )
    assert len(session.entry.ledger.events()) == 1
    assert other.ledger.events() == ()


@pytest.mark.parametrize("missing", ["metadata", "storage"])
def test_missing_native_evidence_rejects_before_operation_or_files(
    native_session: NativeSaveSession, missing: str
) -> None:
    session = native_session
    if missing == "metadata":
        session.state.update_tab_result("tab", session.state.get_tab("tab").run.result)
    else:
        session.state.get_tab("tab").save_context = None
    with pytest.raises(FailedPreconditionError):
        session.control.save_data("tab", origin=CommandOrigin("gui", None))
    assert not session.state.is_tab_busy("tab")
    assert session.completions == []
    assert session.entry.ledger.events() == ()
    assert not (session.entry.database_path / "runs").exists()


def test_cancel_during_metadata_preserves_completed_acquisition_when_saved(
    native_session: NativeSaveSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = native_session
    owner = QtOwnerScheduler()
    gate = OperationGate(session.bus)
    progress = ProgressService(DirectProgressTransport())
    run = RunService(
        session.state,
        OperationRunner(
            gate, session.handles, progress, session.background, session.bus
        ),
        session.bus,
        session.handles,
        MagicMock(),
        gate=gate,
        devices=lambda: {},
    )

    def historical_provenance(*, context: RunContext) -> SoftwareProvenance:
        # Host/Git evidence is not the target of the Run -> Save contract.
        return session.metadata.provenance

    def identity_after_cancel(started_at: datetime) -> str:
        # This collaborator runs after the offline acquisition has returned.
        assert owner.call(run.cancel_run)
        return new_run_id(started_at)

    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.services.run.capture_software_provenance",
        historical_provenance,
    )
    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.services.run.new_run_id", identity_after_cancel
    )
    cfg = session.state.get_tab("tab").cfg
    permit = GuardService(session.state).acquire_run_permit(
        "tab", expected_revision=cfg.observe().ref.revision
    )
    loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    finished: list[RunFinishedPayload] = []

    def terminal(payload: RunFinishedPayload) -> None:
        finished.append(payload)
        loop.quit()

    subscription = session.bus.subscribe(RunFinishedPayload, terminal)
    try:
        token = run.start_run(permit, plots=Plots(NonPresentingHost()))
        timer.start(5000)
        loop.exec()
    finally:
        timer.stop()
        subscription.unsubscribe()
    assert len(finished) == 1 and finished[0].outcome == "cancelled"
    outcome = session.handles.known_outcome(token)
    assert outcome is not None and outcome.status == "cancelled"
    metadata = session.state.get_tab("tab").run.metadata
    assert metadata is not None and metadata.completion == "complete"
    session.metadata = metadata
    saved = session.save()
    assert saved.error is None and saved.report is not None
    stored = load_run_data(Path(saved.data_path))
    assert stored.metadata.completion == "complete"
    assert stored.metadata.run_id == metadata.run_id
    assert stored.metadata.started_at == metadata.started_at
    assert stored.metadata.finished_at == metadata.finished_at


def test_invalid_record_fails_in_worker_without_writing_or_saved_event(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    session.state.update_tab_result("tab", object(), metadata=session.metadata)
    completed = session.save()
    assert completed.report is None
    assert completed.error and "RunRecord" in completed.error
    assert session.entry.ledger.events() == ()
    assert not (session.entry.database_path / "runs").exists()


def declare_panes(
    session: NativeSaveSession, names: tuple[str, ...] = ("fit", "residual")
) -> Plots:
    """Publish finished pane resources through the same State terminal commands."""
    plots = Plots(NonPresentingHost())
    for name in names:
        _, axes = plots.subplots(name)
        axes.plot([0, 1], [1, 2])
    plots.finish(present=False)
    session.state.update_tab_analyze(
        "tab", PaneSummary(), plots, analyze_params_instance=PaneParams(2.0)
    )
    session.state.update_tab_post_analyze(
        "tab", PostSummary(), None, post_analyze_params_instance=PaneParams(4.0)
    )
    return plots


def test_save_all_writes_all_members_and_resaves_saved_artifacts(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    declare_panes(session)
    first = session.save_artifacts()
    assert first.error is None and first.report is not None
    assert len(first.report.outputs) == 8  # data ×2, two figures ×2, two JSON
    assert all(
        item.status == "saved" and item.output.path.is_file()
        for item in first.report.outputs
    )
    sections = session.state.get_artifact_snapshots("tab")
    assert [section.section for section in sections] == [
        "run",
        "analysis",
        "post_analysis",
    ]
    assert all(
        artifact.status is SaveStatus.SAVED
        for section in sections
        for artifact in section.artifacts
    )
    primary_json = next(
        item.output.path
        for item in first.report.outputs
        if item.key.section == "analysis" and item.output.format == "json"
    )
    post_json = next(
        item.output.path
        for item in first.report.outputs
        if item.key.section == "post_analysis"
    )
    assert json.loads(primary_json.read_text())["parameters"] == {"gain": 2.0}
    assert json.loads(post_json.read_text())["summary"] == {"correction": 0.5}
    assert len(session.entry.ledger.events()) == 1
    saved_event = session.entry.ledger.events()[0].payload
    assert isinstance(saved_event, SavedPayload)
    assert len(saved_event.outputs) == 8
    second = session.save_artifacts()
    assert second.error is None and second.report is not None
    assert second.report.run_id == first.report.run_id
    assert second.report.outputs[0].output.path == first.report.outputs[0].output.path
    assert second.report.outputs[1].output.path != first.report.outputs[1].output.path
    metadata = session.state.get_tab("tab").run.metadata
    assert metadata is not None
    assert metadata.labber_path == str(second.report.outputs[1].output.path)
    assert len(session.entry.ledger.events()) == 2


def test_json_only_save_uses_result_inputs_and_preserves_unselected_dirty_run(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    session.save()
    declare_panes(session, ())
    session.state.update_tab_analyze_param_instance("tab", PaneParams(99.0))
    session.control.set_comment("tab", "changed")
    completed = session.save_artifacts((ArtifactRef("analysis", "results"),))
    assert completed.error is None and completed.report is not None
    assert len(completed.report.outputs) == 1
    document = json.loads(completed.report.outputs[0].output.path.read_text())
    assert document["parameters"] == {"gain": 2.0}
    assert document["summary"] == {"mean": 3.0}
    sections = session.state.get_artifact_snapshots("tab")
    assert sections[0].artifacts[0].status is SaveStatus.UNSAVED_CHANGES
    assert sections[1].artifacts[0].status is SaveStatus.SAVED
    assert sections[2].artifacts[0].status is SaveStatus.NOT_SAVED


def test_owner_encoding_once_and_worker_io_with_member_failure_isolation(
    native_session: NativeSaveSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    from matplotlib.figure import Figure
    from zcu_tools.experiment import run_saving
    from zcu_tools.gui.app.measure.services import save as saving

    session = native_session
    plots = declare_panes(session)
    owner_id = threading.get_ident()
    encoded: list[Figure] = []
    io_threads: list[int] = []
    render = saving.render_save_figure_png
    native_write = run_saving.save_run_data

    def encode(figure: Figure) -> bytes:
        assert threading.get_ident() == owner_id
        encoded.append(figure)
        if figure is plots["fit"]:
            raise RuntimeError("encoder failed")
        return render(figure)

    def write(
        destination: Path,
        payload: ExperimentPayload,
        metadata: RunMetadata,
        *,
        cfg: CfgSnapshot,
        replace: bool = False,
        extensions: NativeExtensions | None = None,
    ) -> None:
        io_threads.append(threading.get_ident())
        native_write(
            destination,
            payload,
            metadata,
            cfg=cfg,
            replace=replace,
            extensions=extensions,
        )

    monkeypatch.setattr(saving, "render_save_figure_png", encode)
    monkeypatch.setattr(run_saving, "save_run_data", write)
    completed = session.save_artifacts()
    assert completed.report is not None and completed.error is not None
    assert encoded == [plots["fit"], plots["residual"]]
    assert io_threads and all(thread != owner_id for thread in io_threads)
    failed = [item for item in completed.report.outputs if item.status == "failed"]
    assert len(failed) == 2
    assert all(
        item.key.member_name == "fit" and "encoder failed" in (item.error or "")
        for item in failed
    )
    assert all(
        item.output.path.is_file()
        for item in completed.report.outputs
        if item.status == "saved"
    )
    sections = session.state.get_artifact_snapshots("tab")
    assert sections[1].artifacts[0].status is SaveStatus.PARTIALLY_SAVED
    assert (
        sections[0].artifacts[0].status
        is sections[2].artifacts[0].status
        is SaveStatus.SAVED
    )
    assert len(session.entry.ledger.events()) == 1


def test_copy_path_failure_preserves_canonical_png_and_other_members(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    declare_panes(session, ("fit",))
    planned = (
        session.state.get_artifact_snapshots("tab")[1].artifacts[0].members[0].outputs
    )
    planned[1].path.mkdir(parents=True)
    completed = session.save_artifacts()
    assert completed.report is not None and completed.error is not None
    figures = [item for item in completed.report.outputs if item.key.member == "figure"]
    assert [item.status for item in figures] == ["saved", "failed"]
    assert figures[0].output.path.is_file()
    assert figures[1].error
    assert (
        session.state.get_artifact_snapshots("tab")[1].artifacts[0].status
        is SaveStatus.PARTIALLY_SAVED
    )
    assert len(session.entry.ledger.events()) == 1


@pytest.mark.parametrize(
    "refs",
    [
        (),
        (ArtifactRef("missing", "results"),),
        (ArtifactRef("run", "measurement_data"),) * 2,
    ],
)
def test_invalid_artifact_selection_preserves_annotation_and_admission(
    native_session: NativeSaveSession, refs: tuple[ArtifactRef, ...]
) -> None:
    session = native_session
    previous = session.state.get_artifact_snapshots("tab")
    with pytest.raises(FailedPreconditionError):
        session.control.save_artifacts(
            "tab", artifacts=refs, comment="rejected", origin=GUI_ORIGIN
        )
    assert session.state.get_tab("tab").save.comment == ""
    assert session.state.get_artifact_snapshots("tab") == previous
    assert not session.state.is_tab_busy("tab")
    assert session.artifact_completions == []
    assert session.entry.ledger.events() == ()
    assert not (session.entry.database_path / "runs").exists()


@pytest.mark.parametrize("missing", ["metadata", "storage"])
@pytest.mark.parametrize("batch", [False, True])
def test_missing_evidence_rejects_comment_before_operation(
    native_session: NativeSaveSession, missing: str, batch: bool
) -> None:
    session = native_session
    if missing == "metadata":
        session.state.update_tab_result("tab", session.state.get_tab("tab").run.result)
    else:
        session.state.get_tab("tab").save_context = None
    command = session.control.save_artifacts if batch else session.control.save_data
    with pytest.raises(FailedPreconditionError):
        command("tab", comment="rejected", origin=GUI_ORIGIN)
    assert session.state.get_tab("tab").save.comment == ""
    assert not session.state.is_tab_busy("tab")
    assert not session.completions and not session.artifact_completions
    assert session.entry.ledger.events() == ()
    assert not (session.entry.database_path / "runs").exists()


def test_analysis_replacement_and_new_run_retire_saved_sections(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    declare_panes(session)
    completed = session.save_artifacts()
    assert completed.report is not None
    session.state.update_tab_post_analyze(
        "tab", PostSummary(0.8), None, post_analyze_params_instance=PaneParams(5.0)
    )
    sections = session.state.get_artifact_snapshots("tab")
    assert sections[1].artifacts[0].status is SaveStatus.SAVED
    assert sections[2].artifacts[0].status is SaveStatus.NOT_SAVED
    session.state.update_tab_analyze(
        "tab", PaneSummary(7.0), None, analyze_params_instance=PaneParams(8.0)
    )
    sections = session.state.get_artifact_snapshots("tab")
    assert [section.section for section in sections] == ["run", "analysis"]
    assert sections[0].artifacts[0].status is SaveStatus.SAVED
    assert [member.key.member for member in sections[1].artifacts[0].members] == [
        "analysis"
    ]
    session.state.clear_tab_results("tab")
    assert session.state.get_artifact_snapshots("tab") == ()


def test_resave_all_failed_replaces_previous_success_evidence(
    native_session: NativeSaveSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zcu_tools.experiment import run_saving

    session = native_session
    first = session.save()
    assert first.report is not None and first.error is None
    native_path = first.report.outputs[0].output.path
    native_path.unlink()
    native_path.mkdir()

    def fail_labber(*args: object, **kwargs: object) -> None:
        raise OSError("Labber denied")

    monkeypatch.setattr(run_saving, "write_labber", fail_labber)
    second = session.save()
    assert second.report is not None and second.error is not None
    assert [item.status for item in second.report.outputs] == ["failed", "failed"]
    sections = session.state.get_artifact_snapshots("tab")
    data = sections[0].artifacts[0]
    assert data.status is SaveStatus.NOT_SAVED
    assert [output.status for output in data.members[0].outputs] == ["failed", "failed"]
    assert data.members[0].outputs[1].path == second.report.outputs[1].output.path
    assert data.members[0].outputs[1].path != first.report.outputs[1].output.path
    metadata = session.state.get_tab("tab").run.metadata
    assert metadata is not None
    assert metadata.labber_path == str(first.report.outputs[1].output.path)


def test_invalid_analysis_json_rejects_before_annotation_or_files(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    session.state.update_tab_analyze(
        "tab", PaneSummary(float("nan")), None, analyze_params_instance=PaneParams()
    )
    with pytest.raises(ValueError, match="strict JSON"):
        session.control.save_artifacts("tab", comment="rejected", origin=GUI_ORIGIN)
    assert session.state.get_tab("tab").save.comment == ""
    assert not session.state.is_tab_busy("tab")
    assert session.entry.ledger.events() == ()
    assert not (session.entry.database_path / "runs").exists()


def test_ledger_failure_has_no_complete_report_and_settles_operation(
    native_session: NativeSaveSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = native_session
    declare_panes(session, ())

    def fail_ledger(*args: object, **kwargs: object) -> None:
        raise OSError("ledger denied")

    monkeypatch.setattr(type(session.entry.ledger), "append", fail_ledger)
    completed = session.save_artifacts()
    assert completed.report is None
    assert completed.error and "ledger denied" in completed.error
    assert session.entry.ledger.events() == ()
    assert all(
        artifact.status is SaveStatus.NOT_SAVED
        for section in session.state.get_artifact_snapshots("tab")
        for artifact in section.artifacts
    )


def test_tab_snapshot_reads_committed_sections_without_payload_or_path_planning(
    native_session: NativeSaveSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = native_session
    declare_panes(session)
    expected = session.state.get_artifact_snapshots("tab")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("read model attempted capture or planning")

    monkeypatch.setattr(SaveLayout, "outputs", forbidden)
    monkeypatch.setattr(
        session.state.get_tab("tab").adapter, "capture_analysis_document", forbidden
    )
    tab_service = TabService(
        session.state,
        registry=MagicMock(),
        writeback=MagicMock(),
        cfg_resources=MagicMock(),
        cfg_resolution=MagicMock(),
        captured_parameters=MagicMock(),
    )
    actual = tab_service.get_snapshot("tab")
    assert actual.artifacts == expected
    assert actual.save is not None and actual.save.comment == ""
    assert actual.analysis is not None and actual.analysis.result_params == PaneParams(
        2.0
    )
    assert (
        actual.post_analysis is not None
        and actual.post_analysis.result_params == PaneParams(4.0)
    )
    assert session.state.get_artifact_snapshots("tab") == expected


class StartupBackground:
    """Deliver work inline or reject submit at the runner's public background seam."""

    def __init__(self, on_submit: Callable[[], None], *, fail: bool) -> None:
        self.on_submit = on_submit
        self.fail = fail

    def submit(
        self,
        work: Callable[[], object],
        *,
        run_in_pool: bool,
        on_done: Callable[[object], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        self.on_submit()
        if self.fail:
            raise RuntimeError("submission rejected")
        try:
            result = work()
        except (OSError, ValueError, TypeError, RuntimeError) as error:
            on_error(error)
        else:
            on_done(result)


@pytest.mark.parametrize("command_kind", ["data", "artifacts"])
@pytest.mark.parametrize("fail_submit", [False, True])
def test_inline_terminal_and_submission_failure_clear_admission_once(
    native_session: NativeSaveSession, command_kind: str, fail_submit: bool
) -> None:
    session = native_session
    tokens: list[int] = []

    def opened() -> None:
        active = service.active_save_operations()
        assert len(active) == 1 and active[0].tab_id == "tab"
        assert session.state.is_tab_busy("tab")
        tokens.append(active[0].operation_id)

    service = SaveService(
        session.state,
        OperationRunner(
            None,
            session.handles,
            None,
            StartupBackground(opened, fail=fail_submit),
            session.bus,
        ),
        session.bus,
        owner_scheduler=QtOwnerScheduler(),
    )
    control = SaveControlFacet(
        state=session.state, guard=GuardService(session.state), save=service
    )
    observations: list[tuple[bool, bool, bool]] = []

    def terminal(
        _payload: SaveDataFinishedPayload | SaveArtifactsFinishedPayload,
    ) -> None:
        observations.append(
            (
                not session.state.is_tab_busy("tab"),
                service.active_save_operations() == (),
                session.handles.known_outcome(tokens[0]) is not None,
            )
        )

    subscription = (
        session.bus.subscribe(SaveDataFinishedPayload, terminal)
        if command_kind == "data"
        else session.bus.subscribe(SaveArtifactsFinishedPayload, terminal)
    )
    command = control.save_data if command_kind == "data" else control.save_artifacts
    try:
        if fail_submit:
            with pytest.raises(RuntimeError, match="submission rejected"):
                command("tab", origin=GUI_ORIGIN)
        else:
            submission = command("tab", origin=GUI_ORIGIN)
            assert submission.operation_id == tokens[0]
    finally:
        subscription.unsubscribe()
    assert observations == [(True, True, True)]
    assert service.active_save_operations() == ()
    assert not session.state.is_tab_busy("tab")
    assert session.handles.live_count() == 0
    outcome = session.handles.known_outcome(tokens[0])
    assert outcome is not None
    assert outcome.status == ("failed" if fail_submit else "finished")
    completed = (
        session.completions if command_kind == "data" else session.artifact_completions
    )
    assert len(completed) == 1
    if fail_submit:
        assert completed[0].report is None
        assert completed[0].error == "submission rejected"
        assert session.entry.ledger.events() == ()
    else:
        assert completed[0].report is not None and completed[0].error is None
        assert len(session.entry.ledger.events()) == 1


def test_pending_save_rejects_driving_run_load_and_analysis_without_replacing_panes(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    declare_panes(session, ())
    runner = OperationRunner(
        None, session.handles, None, session.background, session.bus
    )
    writeback = MagicMock()
    tab_service = TabService(
        session.state,
        registry=MagicMock(),
        writeback=writeback,
        cfg_resources=MagicMock(),
        cfg_resolution=MagicMock(),
        captured_parameters=MagicMock(),
    )
    run = RunService(
        session.state,
        runner,
        session.bus,
        session.handles,
        writeback,
        gate=OperationGate(session.bus),
        devices=lambda: {},
    )
    control = RunAnalyzeControlFacet(
        state=session.state,
        bus=session.bus,
        guard=GuardService(session.state),
        tab=tab_service,
        load=LoadService(session.state, writeback, provide_options=lambda _name: ()),
        run=run,
        analyze=AnalyzeService(
            session.state, runner, session.bus, writeback, session.handles
        ),
        post_analyze=PostAnalyzeService(
            session.state, runner, session.bus, session.handles, writeback
        ),
        render_host=lambda: None,
        owner_scheduler=QtOwnerScheduler(),
    )
    before = session.state.get_artifact_snapshots("tab")
    loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    subscription = session.bus.subscribe(
        SaveArtifactsFinishedPayload, lambda _payload: loop.quit()
    )
    try:
        submission = session.control.save_artifacts("tab", origin=GUI_ORIGIN)
        cfg_ref = session.state.get_tab("tab").cfg.observe().ref
        with pytest.raises(FailedPreconditionError, match="busy"):
            control.start_run("tab", cfg_ref)
        with pytest.raises(FailedPreconditionError, match="busy"):
            control.load_tab_result("tab", "never-read.h5")
        with pytest.raises(FailedPreconditionError, match="busy"):
            control.analyze("tab", PaneParams(99.0))
        with pytest.raises(FailedPreconditionError, match="busy"):
            control.start_post_analyze("tab", PaneParams(99.0))
        assert session.state.get_artifact_snapshots("tab") == before
        assert session.state.get_tab("tab").analysis.result_params == PaneParams(2.0)
        session.handles.cancel(submission.operation_id)
        timer.start(5000)
        loop.exec()
    finally:
        timer.stop()
        subscription.unsubscribe()
    assert len(session.artifact_completions) == 1
    assert session.artifact_completions[0].report is not None
    assert session.artifact_completions[0].error is None
    outcome = session.handles.known_outcome(submission.operation_id)
    assert outcome is not None and outcome.status == "finished"
    assert session.handles.live_count() == 0
