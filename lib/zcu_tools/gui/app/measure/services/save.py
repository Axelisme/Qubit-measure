"""Native artifact saving through one worker and one State terminal commit."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from zcu_tools.datafile import RunMetadata
from zcu_tools.experiment.run_saving import (
    ArtifactRef,
    DataMember,
    FileMember,
    SaveArtifact,
    SaveReport,
    save_artifacts,
)
from zcu_tools.gui.app.measure.adapter import SaveDataRequest
from zcu_tools.gui.app.measure.catalog import ExperimentAccess
from zcu_tools.gui.app.measure.events.completion import (
    SaveArtifactsFinishedPayload,
    SaveDataFinishedPayload,
)
from zcu_tools.gui.app.measure.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.app.measure.figure_export import render_save_figure_png
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.context_types import CommandOrigin
from zcu_tools.gui.session.operation_handles import OperationOutcome
from zcu_tools.gui.session.operation_runner import BgResult, OperationSpec, SettleFn
from zcu_tools.resources.entry import ArtifactKey, SaveLayout

from .guard import SavePermit
from .ports import (
    ActiveSaveOperation,
    SaveArtifactsSubmission,
    SaveDataSubmission,
    SaveDestination,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from zcu_tools.gui.app.measure.state import State
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus
    from zcu_tools.gui.session.operation_runner import OperationRunner
    from zcu_tools.gui.session.ports import OwnerScheduler
    from zcu_tools.resources.entry import PointStorageBinding


@dataclass(frozen=True)
class _CapturedArtifact:
    ref: ArtifactRef
    members: SaveDataRequest[object] | tuple[FileMember, ...]


@dataclass(frozen=True)
class _SavePlan:
    artifacts: tuple[_CapturedArtifact, ...]
    layout: SaveLayout
    metadata: RunMetadata
    storage: PointStorageBinding
    origin: CommandOrigin
    comment: str
    destinations: tuple[SaveDestination, ...]


class SaveService:
    """Capture fixed pane content, then save it without a hardware lease.

    Public commands run on State's owner thread. Data mapping and file/ledger
    I/O belong to the worker. PNG encoding returns to the Figure owner through
    the injected scheduler. Save commands are non-cancellable and never retry.
    """

    def __init__(
        self,
        state: State,
        runner: OperationRunner,
        bus: EventBus,
        *,
        owner_scheduler: OwnerScheduler,
        access: ExperimentAccess | None = None,
    ) -> None:
        """Inject the operation runner, terminal bus and Figure owner scheduler.

        access is the shared catalog reload/shutdown gate for driving-entry
        preflight. None creates an independent available gate. Direct start
        commands keep their existing same-tab busy-only exclusion policy.
        """
        self._state = state
        self._runner = runner
        self._bus = bus
        self._owner_scheduler = owner_scheduler
        self._access = access if access is not None else ExperimentAccess()
        self._active_operations: dict[str, int] = {}

    def require_save_available(self, tab_id: str) -> None:
        """Reject reload/shutdown or busy tabs without changing drafts or files.

        tab_id must identify an existing tab. Raise FailedPreconditionError for
        unavailable commands; unknown-tab errors propagate.
        """
        self._access.require_available()
        self._require_tab_idle(tab_id)

    def active_save_operations(self) -> tuple[ActiveSaveOperation, ...]:
        """Return the operation IDs and owning tabs of currently active saves."""
        return tuple(
            ActiveSaveOperation(token, tab_id)
            for tab_id, token in self._active_operations.items()
        )

    def start_save_data(
        self, permit: SavePermit, comment: str | None = None, *, origin: CommandOrigin
    ) -> SaveDataSubmission:
        """Submit native HDF5 and a fresh Labber file for the captured Run.

        permit identifies the guarded tab. comment=None retains its annotation;
        origin attributes this command. Missing historical storage/metadata or
        busy state raises FailedPreconditionError before comment mutation,
        operation admission or I/O. The returned data_path is a reservation.
        The data terminal contains both outputs after State/handle settle.
        """
        plan = self._capture(
            permit.tab_id, (ArtifactRef("run", "measurement_data"),), comment, origin
        )
        data_path = str(plan.destinations[0].output.path)
        token = self._begin_save(permit.tab_id, plan, data_path=data_path)
        return SaveDataSubmission(token, data_path)

    def start_save_artifacts(
        self,
        permit: SavePermit,
        artifacts: tuple[ArtifactRef, ...] | None = None,
        comment: str | None = None,
        *,
        origin: CommandOrigin,
    ) -> SaveArtifactsSubmission:
        """Save whole artifacts in one non-cancellable, lease-free operation.

        artifacts=None includes all declarations, including Saved. Explicit refs
        retain their order; empty, repeated or unknown refs raise
        FailedPreconditionError. Missing historical storage/metadata and busy
        state also fail before comment mutation, operation admission or I/O.
        JSON capture uses each completed pane's historical inputs, not its next
        draft. Invalid pane content propagates TypeError/ValueError before
        admission. comment=None retains the annotation. origin identifies this
        Save command. Destinations are reservations, not proof of success.
        Every output is reported independently; terminal State commit precedes
        handle settle, which precedes the completion fact.
        """
        plan = self._capture(permit.tab_id, artifacts, comment, origin)
        token = self._begin_save(permit.tab_id, plan)
        return SaveArtifactsSubmission(token, plan.destinations)

    def _require_tab_idle(self, tab_id: str) -> None:
        if self._state.is_tab_busy(tab_id):
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")

    def _capture(
        self,
        tab_id: str,
        refs: tuple[ArtifactRef, ...] | None,
        comment: str | None,
        origin: CommandOrigin,
    ) -> _SavePlan:
        self._require_tab_idle(tab_id)
        tab = self._state.get_tab(tab_id)
        metadata = tab.run.metadata
        if metadata is None or tab.save_context is None:
            raise FailedPreconditionError(
                "Run has no historical native metadata or captured storage",
                reason_code="storage_unavailable",
            )
        try:
            selected = tab.artifacts.select(refs)
        except ValueError as exc:
            raise FailedPreconditionError(str(exc)) from exc
        storage = tab.save_context.capture_storage()
        layout = SaveLayout(
            result_path=storage.result_path,
            database_path=storage.database_path,
            run_id=metadata.run_id,
            point=storage.point,
            saved_at=datetime.now().astimezone(),
        )
        annotation = tab.save.comment if comment is None else comment
        destinations = tuple(
            SaveDestination(key, output)
            for artifact in selected
            for key in artifact.members
            for output in layout.outputs(key)
        )
        captured: list[_CapturedArtifact] = []
        for artifact in selected:
            ref = artifact.ref
            if ref.section == "run":
                data_output = next(
                    destination.output
                    for destination in destinations
                    if destination.key in artifact.members
                    and destination.output.format == "data_h5"
                )
                captured.append(
                    _CapturedArtifact(
                        ref,
                        SaveDataRequest(
                            run_result=tab.run.result,
                            data_path=str(data_output.path),
                            parameters=tab.parameters,
                            storage=storage,
                            metadata=metadata,
                            comment=annotation,
                        ),
                    )
                )
            else:
                if ref.section == "analysis":
                    pane = tab.analysis
                    pane_name = "analysis"
                elif ref.section == "post_analysis":
                    pane = tab.post_analysis
                    pane_name = "post_analysis"
                else:
                    raise ValueError(f"Unsupported pane section {ref.section!r}")
                members: list[FileMember] = []
                for key in artifact.members:
                    if key.member == "figure":
                        if pane.plots is None or key.member_name is None:
                            raise ValueError("Declared figure has no pane content")
                        content = pane.plots[key.member_name]
                    else:
                        content = tab.adapter.capture_analysis_document(
                            pane.result, pane.result_params, pane=pane_name
                        )
                    members.append(FileMember(key, content))
                captured.append(_CapturedArtifact(ref, tuple(members)))
        return _SavePlan(
            tuple(captured), layout, metadata, storage, origin, annotation, destinations
        )

    def _begin_save(
        self, tab_id: str, plan: _SavePlan, *, data_path: str | None = None
    ) -> int:
        """Admit captured inputs, run shared saver, commit report, settle and emit."""
        adapter = self._state.get_tab(tab_id).adapter
        self._state.update_tab_comment(tab_id, plan.comment)
        self._state.set_tab_saving_data(tab_id, saving_data=True)
        self._bus.emit(
            TabInteractionChangedPayload(
                tab_id=tab_id, fact=TabInteractionFact.SAVE_STARTED
            )
        )
        terminal_delivered = False

        def encode_figure(figure: Figure) -> bytes:
            return self._owner_scheduler.call(lambda: render_save_figure_png(figure))

        def work(_factory: object) -> SaveReport:
            artifacts: list[SaveArtifact] = []
            for captured in plan.artifacts:
                if isinstance(captured.members, SaveDataRequest):
                    request = captured.members
                    prepared = adapter.prepare_save_data(
                        request.run_result, comment=request.comment
                    )
                    members = (
                        DataMember(
                            ArtifactKey(
                                captured.ref.section, captured.ref.name, "data"
                            ),
                            prepared,
                        ),
                    )
                else:
                    members = captured.members
                artifacts.append(SaveArtifact(captured.ref, members))
            return save_artifacts(
                tuple(artifacts),
                layout=plan.layout,
                metadata=plan.metadata,
                ledger=plan.storage.ledger,
                origin=plan.origin.origin,
                call_id=plan.origin.call_id,
                figure_encoder=encode_figure,
            )

        def on_opened(token: int) -> None:
            self._active_operations[tab_id] = token

        def emit_finished(report: SaveReport | None, error: str | None) -> None:
            self._bus.emit(
                TabInteractionChangedPayload(
                    tab_id=tab_id,
                    fact=(
                        TabInteractionFact.SAVE_SUCCEEDED
                        if error is None
                        else TabInteractionFact.SAVE_FAILED
                    ),
                )
            )
            if data_path is None:
                self._bus.emit(
                    SaveArtifactsFinishedPayload(
                        tab_id=tab_id, error=error, report=report
                    )
                )
            else:
                self._bus.emit(
                    SaveDataFinishedPayload(
                        tab_id=tab_id, data_path=data_path, error=error, report=report
                    )
                )

        def on_terminal(bg: BgResult, settle: SettleFn) -> None:
            nonlocal terminal_delivered
            report: SaveReport | None = bg.result if bg.ok else None
            error = None if bg.ok else str(bg.error)
            if report is not None:
                self._state.commit_save_report(tab_id, report)
                error = (
                    "\n".join(
                        f"{item.output.path}: {item.error}"
                        for item in report.outputs
                        if item.status == "failed"
                    )
                    or None
                )
            self._state.set_tab_saving_data(tab_id, saving_data=False)
            self._active_operations.pop(tab_id, None)
            settle(OperationOutcome("finished" if error is None else "failed", error))
            terminal_delivered = True
            emit_finished(report, error)

        try:
            return self._runner.begin(
                OperationSpec(
                    exclusion=None,
                    owner_id=tab_id,
                    wants_progress=False,
                    cancel_hook=None,
                    work=work,
                    run_in_pool=False,
                    on_terminal=on_terminal,
                    on_opened=on_opened,
                )
            )
        except Exception as exc:
            # Runner settles startup failures; inline completion may already emit.
            if not terminal_delivered:
                self._state.set_tab_saving_data(tab_id, saving_data=False)
                self._active_operations.pop(tab_id, None)
                emit_finished(None, str(exc))
            raise
