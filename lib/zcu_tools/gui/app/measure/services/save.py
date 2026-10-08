from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from zcu_tools.datafile import RunMetadata
from zcu_tools.datafile.paths import reserve_labber_filepath
from zcu_tools.experiment.run_saving import (
    ArtifactRef,
    DataMember,
    SaveArtifact,
    SaveReport,
    save_artifacts,
)
from zcu_tools.gui.app.measure.adapter import SaveDataRequest
from zcu_tools.gui.app.measure.artifact_paths import require_distinct_paths
from zcu_tools.gui.app.measure.artifact_tracker import (
    ArtifactKey,
    ArtifactKind,
    SaveAttempt,
)
from zcu_tools.gui.app.measure.catalog import ExperimentAccess
from zcu_tools.gui.app.measure.events.completion import (
    SaveArtifactsFinishedPayload,
    SaveDataFinishedPayload,
)
from zcu_tools.gui.app.measure.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.app.measure.figure_export import (
    resolve_figure_path,
    save_figure_to_path,
)
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.context_types import CommandOrigin
from zcu_tools.gui.session.operation_handles import OperationOutcome
from zcu_tools.gui.session.operation_runner import (
    BgResult,
    OperationRunner,
    OperationSpec,
    SettleFn,
)
from zcu_tools.resources.entry import ArtifactKey as NativeArtifactKey
from zcu_tools.resources.entry import SaveLayout

from .guard import SavePermit
from .ports import (
    ActiveSaveOperation,
    SaveArtifactsSubmission,
    SaveDataSubmission,
    SaveDestination,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from zcu_tools.gui.app.measure.state import State
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus
    from zcu_tools.gui.session.ports import OwnerScheduler


def resolve_artifact_destinations(
    destinations: tuple[SaveDestination, ...],
) -> tuple[SaveDestination, ...]:
    """Resolve output names and reject collisions without writing files.

    The facade checks before changing drafts; the service checks again at its
    independent submission boundary. Data naming only inspects existing paths.
    """
    for destination in destinations:
        name = destination.key.figure_name
        if name is not None:
            try:
                name.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise FailedPreconditionError(
                    "Figure name must be UTF-8 encodable"
                ) from exc
    resolved = tuple(
        SaveDestination(
            d.key,
            reserve_labber_filepath(d.path)
            if d.key.kind is ArtifactKind.DATA
            else resolve_figure_path(d.path),
        )
        for d in destinations
    )
    try:
        require_distinct_paths(tuple(d.path for d in resolved))
    except ValueError as exc:
        raise FailedPreconditionError(str(exc)) from exc
    return resolved


class SaveService:
    def __init__(
        self,
        state: State,
        runner: OperationRunner,
        bus: EventBus,
        *,
        owner_scheduler: OwnerScheduler,
        access: ExperimentAccess | None = None,
    ) -> None:
        """Use tab state, operation runner and owner scheduler for save commands.

        ``access`` is the shared catalog reload/shutdown gate used only by
        driving-entry preflight. Omitting it creates an available independent
        gate. Direct save operations retain their separate busy-only policy.
        """
        self._access = access if access is not None else ExperimentAccess()
        self._state = state
        self._runner = runner
        self._bus = bus
        self._owner_scheduler = owner_scheduler
        self._active_paths: dict[str, str] = {}
        self._active_operations: dict[str, int] = {}

    def require_save_available(self, tab_id: str) -> None:
        """Check driving-entry catalog availability, then same-tab exclusion.

        ``tab_id`` names an existing tab, as established by SavePermit. Raise
        FailedPreconditionError during reload/shutdown or while the tab is busy.
        Unknown-tab errors propagate. This command changes no draft, artifact,
        operation, path or event state; direct save commands remain busy-only.
        """
        self._access.require_available()
        self._require_tab_idle(tab_id)

    def active_save_operations(self) -> tuple[ActiveSaveOperation, ...]:
        return tuple(
            ActiveSaveOperation(token, tab_id)
            for tab_id, token in self._active_operations.items()
        )

    def _start_save(
        self,
        tab_id: str,
        req: SaveDataRequest[object],
        attempt: SaveAttempt,
        *,
        layout: SaveLayout,
        metadata: RunMetadata,
        origin: CommandOrigin,
    ) -> int:
        """Run both writers and one ledger event in the non-cancellable worker."""
        adapter = self._state.get_tab(tab_id).adapter

        def work(_factory: object) -> SaveReport:
            prepared = adapter.prepare_save_data(req.run_result, comment=req.comment)
            artifact = SaveArtifact(
                ArtifactRef("run", "measurement_data"),
                (
                    DataMember(
                        NativeArtifactKey("run", "measurement_data", "data"), prepared
                    ),
                ),
            )
            return save_artifacts(
                (artifact,),
                layout=layout,
                metadata=metadata,
                ledger=req.storage.ledger,
                origin=origin.origin,
                call_id=origin.call_id,
            )

        def on_terminal(result: BgResult, settle: SettleFn) -> None:
            report: SaveReport | None = None
            error = result.error
            if result.ok:
                if isinstance(result.result, SaveReport):
                    report = result.result
                    failures = [
                        f"{item.output.path}: {item.error}"
                        for item in report.outputs
                        if item.status == "failed"
                    ]
                    if failures:
                        error = RuntimeError("\n".join(failures))
                else:
                    error = TypeError("Native Save worker returned no SaveReport")
            elif error is None:
                error = RuntimeError("save failed without an error")
            payload = self._finish_save_state(tab_id, error, attempt, report=report)
            settle(
                OperationOutcome(
                    "finished" if error is None else "failed", payload.error
                )
            )
            self._emit_save_finished(payload)

        return self._runner.begin(
            OperationSpec(
                exclusion=None,
                owner_id=tab_id,
                wants_progress=False,
                cancel_hook=None,
                work=work,
                run_in_pool=False,
                on_terminal=on_terminal,
            )
        )

    def start_save_data(
        self, permit: SavePermit, comment: str = "", *, origin: CommandOrigin
    ) -> SaveDataSubmission:
        """Submit both native and Labber outputs for the pane's captured Run.

        permit identifies the guarded existing tab. comment is its annotation;
        origin identifies this Save command. Missing historical Run metadata or
        captured entry storage raises FailedPreconditionError before submission.
        SaveLayout reserves destinations without mkdir or writes. Preparation,
        both writers and ledger append run in the worker; output failures are
        isolated by the shared saver and reported at terminal. No retry/rollback,
        acquisition identity generation, live-context lookup or path override.
        """
        tab_id = permit.tab_id
        self._require_tab_idle(tab_id)
        req = self._make_save_data_request(tab_id, "", comment=comment)
        metadata = req.metadata
        if metadata is None:
            raise FailedPreconditionError(
                "Run has no native acquisition metadata",
                reason_code="storage_unavailable",
            )
        layout = SaveLayout(
            result_path=req.storage.result_path,
            database_path=req.storage.database_path,
            run_id=metadata.run_id,
            point=req.storage.point,
            saved_at=datetime.now().astimezone(),
        )
        outputs = layout.outputs(NativeArtifactKey("run", "measurement_data", "data"))
        data_path = str(outputs[0].path)
        req = replace(req, data_path=data_path)
        logger.info("start_save_data: tab_id=%r path=%r", tab_id, data_path)
        self._state.get_artifact_snapshots(tab_id)
        tracker = self._state.get_tab(tab_id).artifacts
        attempt = tracker.started(ArtifactKey(ArtifactKind.DATA))
        self._active_paths[tab_id] = data_path
        self._mark_saving(tab_id, TabInteractionFact.SAVE_STARTED)
        try:
            token = self._start_save(
                tab_id, req, attempt, layout=layout, metadata=metadata, origin=origin
            )
        except Exception as error:
            # OperationRunner has already settled a synchronous submission failure.
            if attempt.pending:
                self._emit_save_finished(
                    self._finish_save_state(tab_id, error, attempt)
                )
            raise
        # Inline executors may have delivered terminal before begin returns.
        if tab_id in self._active_paths:
            self._active_operations[tab_id] = token
        return SaveDataSubmission(token, data_path)

    def start_save_artifacts(
        self,
        permit: SavePermit,
        destinations: tuple[SaveDestination, ...],
        comment: str = "",
    ) -> SaveArtifactsSubmission:
        """Save an ordered subset under one non-cancellable, lease-free handle."""
        tab_id = permit.tab_id
        self._require_tab_idle(tab_id)
        destinations = self._prepare_destinations(tab_id, destinations)
        tab = self._state.get_tab(tab_id)
        data_path = next(
            (d.path for d in destinations if d.key.kind is ArtifactKind.DATA), None
        )
        req = (
            self._make_save_data_request(tab_id, data_path, comment)
            if data_path is not None
            else None
        )
        adapter = tab.adapter
        owner = self._owner_scheduler
        figures = {
            d.key: self._capture_image_figure(tab_id, d.key)
            for d in destinations
            if d.key.kind is not ArtifactKind.DATA
        }
        attempts = self._start_artifact_attempts(tab_id, destinations)

        def work(_factory: object) -> None:
            for destination in destinations:
                attempt = attempts[destination.key]
                if destination.key.kind is ArtifactKind.DATA:
                    if req is None:
                        raise RuntimeError("Data save has no prepared request")
                    self._ensure_parent_directory(destination.path)
                    adapter.save(req)
                    owner.call(lambda a=attempt, d=destination: a.succeed(d.path))
                else:
                    owner.call(
                        lambda a=attempt, d=destination: self._export_image(
                            d, figure=figures[d.key], attempt=a
                        )
                    )

        def on_terminal(result: BgResult, settle: SettleFn) -> None:
            error = None if result.ok else str(result.error or "save failed")
            self._finish_artifact_save(tab_id, tuple(attempts.values()))
            settle(OperationOutcome("finished" if result.ok else "failed", error))
            self._emit_artifact_save_finished(tab_id, error)

        # Later GUI draft edits must not be promoted by this batch's success.
        self._active_paths[tab_id] = data_path or ""
        self._mark_saving(tab_id, TabInteractionFact.SAVE_STARTED)
        try:
            token = self._runner.begin(
                OperationSpec(
                    exclusion=None,
                    owner_id=tab_id,
                    wants_progress=False,
                    cancel_hook=None,
                    work=work,
                    run_in_pool=False,
                    on_terminal=on_terminal,
                )
            )
        except Exception as error:
            self._finish_artifact_save(tab_id, tuple(attempts.values()))
            self._emit_artifact_save_finished(tab_id, str(error))
            raise
        if tab_id in self._active_paths:
            self._active_operations[tab_id] = token
        return SaveArtifactsSubmission(token, destinations)

    def _capture_image_figure(self, tab_id: str, key: ArtifactKey) -> Figure:
        """Keep the original Figure when a later pane swap reuses its key."""
        pane = self._state.get_tab(tab_id).image_pane(key)
        name = key.figure_name
        if name is None or pane.plots is None:
            raise FailedPreconditionError(f"No figure for {key!r}")
        return pane.plots[name]

    def _start_artifact_attempts(
        self, tab_id: str, destinations: tuple[SaveDestination, ...]
    ) -> dict[ArtifactKey, SaveAttempt]:
        """Capture all records or undo reservations before starting the worker."""
        tracker = self._state.get_tab(tab_id).artifacts
        attempts: dict[ArtifactKey, SaveAttempt] = {}
        try:
            for destination in destinations:
                attempts[destination.key] = tracker.started(destination.key)
        except Exception:
            for attempt in attempts.values():
                attempt.fail()
            raise
        return attempts

    def _finish_artifact_save(
        self, tab_id: str, attempts: tuple[SaveAttempt, ...]
    ) -> None:
        for attempt in attempts:
            if attempt.pending:
                attempt.fail()
        self._active_paths.pop(tab_id, None)
        self._active_operations.pop(tab_id, None)
        self._state.set_tab_saving_data(tab_id, saving_data=False)

    def _emit_artifact_save_finished(self, tab_id: str, error: str | None) -> None:
        self._bus.emit(
            TabInteractionChangedPayload(
                tab_id,
                TabInteractionFact.SAVE_SUCCEEDED
                if error is None
                else TabInteractionFact.SAVE_FAILED,
            )
        )
        self._bus.emit(SaveArtifactsFinishedPayload(tab_id, error))

    def _prepare_destinations(
        self, tab_id: str, destinations: tuple[SaveDestination, ...]
    ) -> tuple[SaveDestination, ...]:
        selected = {d.key: d.path for d in destinations}
        if not selected or len(selected) != len(destinations):
            raise FailedPreconditionError(
                "Save requires a nonempty unique artifact set"
            )
        available = {a.key: a for a in self._state.get_artifact_snapshots(tab_id)}
        for key, path in selected.items():
            if key not in available or not available[key].is_saveable:
                raise FailedPreconditionError(f"Artifact {key!r} is not saveable")
            if not path.strip():
                raise FailedPreconditionError(f"Artifact {key!r} has an empty path")
        return resolve_artifact_destinations(
            tuple(
                SaveDestination(key, selected[key])
                for kind in (
                    ArtifactKind.ANALYSIS,
                    ArtifactKind.POST_ANALYSIS,
                    ArtifactKind.DATA,
                )
                for key in available
                if key.kind is kind and key in selected
            )
        )

    def _export_image(
        self, destination: SaveDestination, *, figure: Figure, attempt: SaveAttempt
    ) -> None:
        try:
            self._ensure_parent_directory(destination.path)
            save_figure_to_path(figure, destination.path)
        except Exception:
            if attempt.pending:
                attempt.fail()
            raise
        attempt.succeed(destination.path)

    def save_image_sync(
        self, permit: SavePermit, key: ArtifactKey, image_path: str
    ) -> None:
        """Explicitly re-export one current named figure on the owner thread."""
        tab_id = permit.tab_id
        self._require_tab_idle(tab_id)
        (destination,) = self._prepare_destinations(
            tab_id, (SaveDestination(key, image_path),)
        )
        figure = self._capture_image_figure(tab_id, key)
        attempt = self._state.get_tab(tab_id).artifacts.started(key)
        self._export_image(destination, figure=figure, attempt=attempt)

    def _require_tab_idle(self, tab_id: str) -> None:
        """Reject every save entry point while another tab operation owns it.

        ``GuardService`` owns static preconditions; the tab's busy state is a
        dynamic resource check and must happen before path reservation or image
        export so a Save permit cannot bypass same-tab operation exclusion.
        """
        if self._state.is_tab_busy(tab_id):
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")

    @staticmethod
    def _ensure_parent_directory(path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _make_save_data_request(
        self, tab_id: str, data_path: str, comment: str = ""
    ) -> SaveDataRequest[object]:
        # Run-result presence is proven by the SavePermit; tab-busy is the
        # dynamic check that stays at the operation boundary.
        if self._state.is_tab_busy(tab_id):
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")

        tab = self._state.get_tab(tab_id)
        if tab.save_context is None:
            raise FailedPreconditionError(
                "Tab has no captured save context", reason_code="storage_unavailable"
            )
        return SaveDataRequest(
            run_result=tab.run.result,
            data_path=data_path,
            parameters=tab.parameters,
            storage=tab.save_context.capture_storage(),
            metadata=tab.run.metadata,
            comment=comment,
        )

    def _mark_saving(
        self,
        tab_id: str,
        fact: TabInteractionFact,
    ) -> None:
        self._state.set_tab_saving_data(tab_id, saving_data=True)
        self._bus.emit(
            TabInteractionChangedPayload(tab_id=tab_id, fact=fact),
        )

    def _finish_save_state(
        self,
        tab_id: str,
        error: Exception | None,
        attempt: SaveAttempt,
        *,
        report: SaveReport | None = None,
    ) -> SaveDataFinishedPayload:
        path = self._active_paths.pop(tab_id, "")
        self._active_operations.pop(tab_id, None)
        if error is None:
            logger.info("save finished: tab_id=%r path=%r", tab_id, path)
            attempt.succeed(path)
        else:
            logger.warning(
                "save failed: tab_id=%r path=%r error=%r", tab_id, path, error
            )
            if attempt.pending:
                attempt.fail()
        self._state.set_tab_saving_data(tab_id, saving_data=False)
        return SaveDataFinishedPayload(
            tab_id=tab_id,
            data_path=path,
            error=None if error is None else str(error),
            report=report,
        )

    def _emit_save_finished(self, payload: SaveDataFinishedPayload) -> None:
        """Publish only after State and the shared handle agree on terminal state."""
        fact = (
            TabInteractionFact.SAVE_SUCCEEDED
            if payload.error is None
            else TabInteractionFact.SAVE_FAILED
        )
        self._bus.emit(TabInteractionChangedPayload(tab_id=payload.tab_id, fact=fact))
        self._bus.emit(payload)
