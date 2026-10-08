"""Native measurement-data Save through real state, workers, writers and ledger."""

from collections.abc import Generator, Mapping
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
    RunMetadata,
    RunSnapshot,
    SoftwareProvenance,
    load_run_data,
)
from zcu_tools.experiment.acquisition import new_run_id
from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.events.completion import SaveDataFinishedPayload
from zcu_tools.gui.app.measure.events.run import RunFinishedPayload
from zcu_tools.gui.app.measure.services.guard import GuardService
from zcu_tools.gui.app.measure.services.operation_gate import OperationGate
from zcu_tools.gui.app.measure.services.run import RunService
from zcu_tools.gui.app.measure.services.save import SaveService
from zcu_tools.gui.app.measure.services.save_control import SaveControlFacet
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
from zcu_tools.resources.entry import ResultEntry, SavedPayload
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult
from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui.app.measure._cfg_fakes import make_cfg
from tests.gui.app.measure.adapter._cfg_fakes import StaticCfgDefinition

GUI_ORIGIN = CommandOrigin("gui", None)


class SavingAdapter(BaseAdapter[SavingCfg, RunRecord[SavingCfg, SavingResult]]):
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
            adapter=SavingAdapter(),
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
    # TabService participates only in legacy path commands, not native Data Save.
    # Passing the real assembled tab owner is unnecessary for this entry seam.
    control = SaveControlFacet(
        state=state,
        bus=bus,
        guard=GuardService(state),
        tab=MagicMock(),
        save=service,
        notify_info=lambda _message: None,
    )
    completions: list[SaveDataFinishedPayload] = []
    subscription = bus.subscribe(SaveDataFinishedPayload, completions.append)
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
            background,
        )
    finally:
        subscription.unsubscribe()
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
