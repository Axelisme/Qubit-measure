from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from zcu_tools.gui.app.main.adapter import SaveDataRequest
from zcu_tools.gui.app.main.artifact_tracker import ArtifactKind
from zcu_tools.gui.app.main.events.completion import (
    SaveArtifactsFinishedPayload,
    SaveDataFinishedPayload,
)
from zcu_tools.gui.app.main.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.app.main.figure_export import (
    resolve_figure_path,
    save_figure_to_path,
)
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.operation_handles import OperationOutcome
from zcu_tools.gui.session.operation_runner import (
    BgResult,
    OperationRunner,
    OperationSpec,
    SettleFn,
)
from zcu_tools.utils.datasaver import reserve_labber_filepath

from .guard import SavePermit
from .ports import (
    ActiveSaveOperation,
    SaveArtifactsSubmission,
    SaveDataSubmission,
    SaveDestination,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.state import State
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus
    from zcu_tools.gui.session.ports import OwnerScheduler


def resolve_artifact_destinations(
    destinations: tuple[SaveDestination, ...],
) -> tuple[SaveDestination, ...]:
    """Resolve output names and reject collisions without writing files.

    The facade checks before changing drafts; the service checks again at its
    independent submission boundary. Data naming only inspects existing paths.
    """
    resolved = tuple(
        SaveDestination(
            d.kind,
            reserve_labber_filepath(d.path)
            if d.kind is ArtifactKind.DATA
            else resolve_figure_path(d.path),
        )
        for d in destinations
    )
    paths = [Path(d.path).resolve() for d in resolved]
    if len(set(paths)) != len(paths):
        raise FailedPreconditionError("Save destinations must have distinct paths")
    return resolved


class SaveService:
    def __init__(
        self,
        state: State,
        runner: OperationRunner,
        bus: EventBus,
        *,
        owner_scheduler: OwnerScheduler,
    ) -> None:
        self._state = state
        self._runner = runner
        self._bus = bus
        self._owner_scheduler = owner_scheduler
        self._active_paths: dict[str, str] = {}
        self._active_operations: dict[str, int] = {}

    def active_save_operations(self) -> tuple[ActiveSaveOperation, ...]:
        return tuple(
            ActiveSaveOperation(token, tab_id)
            for tab_id, token in self._active_operations.items()
        )

    def _start_save(self, tab_id: str, req: SaveDataRequest) -> int:
        """Submit non-cancellable I/O to the shared handle lifecycle, without a lease."""
        adapter = self._state.get_tab(tab_id).adapter

        def on_terminal(result: BgResult, settle: SettleFn) -> None:
            error = (
                None
                if result.ok
                else result.error or RuntimeError("save failed without an error")
            )
            payload = self._finish_save_state(tab_id, error)
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
                work=lambda _factory: adapter.save(req),
                run_in_pool=False,
                on_terminal=on_terminal,
            )
        )

    def start_save_data(
        self, permit: SavePermit, data_path: str, comment: str = ""
    ) -> SaveDataSubmission:
        tab_id = permit.tab_id
        self._require_tab_idle(tab_id)
        # Reserve the final data path in the GUI orchestration layer so the
        # worker receives the exact file it must write.
        data_path = reserve_labber_filepath(data_path)
        req = self._make_save_data_request(tab_id, data_path, comment=comment)
        logger.info("start_save_data: tab_id=%r path=%r", tab_id, data_path)
        self._ensure_parent_directory(data_path)
        self._state.get_artifact_snapshots(tab_id)
        tracker = self._state.get_tab(tab_id).artifacts
        tracker.started(ArtifactKind.DATA)
        self._active_paths[tab_id] = data_path
        self._mark_saving(tab_id, True, TabInteractionFact.SAVE_STARTED)
        try:
            token = self._start_save(tab_id, req)
        except Exception as error:
            # OperationRunner has already settled a synchronous submission failure.
            self._emit_save_finished(self._finish_save_state(tab_id, error))
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
            (d.path for d in destinations if d.kind is ArtifactKind.DATA), None
        )
        req = (
            self._make_save_data_request(tab_id, data_path, comment)
            if data_path is not None
            else None
        )
        adapter = tab.adapter
        tracker = tab.artifacts
        owner = self._owner_scheduler

        def work(_factory: object) -> None:
            for destination in destinations:
                if destination.kind is ArtifactKind.DATA:
                    if req is None:
                        raise RuntimeError("Data save has no prepared request")
                    self._ensure_parent_directory(destination.path)
                    adapter.save(req)
                    owner.call(lambda d=destination: tracker.succeeded(d.kind, d.path))
                else:
                    owner.call(
                        lambda d=destination: self._export_image(
                            tab_id, d, capture_signature=False
                        )
                    )

        def on_terminal(result: BgResult, settle: SettleFn) -> None:
            error = None if result.ok else str(result.error or "save failed")
            self._finish_artifact_save(tab_id, destinations)
            settle(OperationOutcome("finished" if result.ok else "failed", error))
            self._emit_artifact_save_finished(tab_id, error)

        # Freeze every signature alongside the prepared payload and destinations.
        # Later GUI draft edits must not be promoted by this batch's success.
        for destination in destinations:
            tracker.started(destination.kind)
        self._active_paths[tab_id] = data_path or ""
        self._mark_saving(tab_id, True, TabInteractionFact.SAVE_STARTED)
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
            self._finish_artifact_save(tab_id, destinations)
            self._emit_artifact_save_finished(tab_id, str(error))
            raise
        if tab_id in self._active_paths:
            self._active_operations[tab_id] = token
        return SaveArtifactsSubmission(token, destinations)

    def _finish_artifact_save(
        self, tab_id: str, destinations: tuple[SaveDestination, ...]
    ) -> None:
        tracker = self._state.get_tab(tab_id).artifacts
        for destination in destinations:
            tracker.failed(destination.kind)
        self._active_paths.pop(tab_id, None)
        self._active_operations.pop(tab_id, None)
        self._state.set_tab_saving_data(tab_id, False)

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
        selected = {d.kind: d.path for d in destinations}
        if not selected or len(selected) != len(destinations):
            raise FailedPreconditionError(
                "Save requires a nonempty unique artifact set"
            )
        available = {a.kind: a for a in self._state.get_artifact_snapshots(tab_id)}
        for kind, path in selected.items():
            if kind not in available or not available[kind].is_saveable:
                raise FailedPreconditionError(f"Artifact {kind.value} is not saveable")
            if not path.strip():
                raise FailedPreconditionError(
                    f"Artifact {kind.value} has an empty path"
                )
        return resolve_artifact_destinations(
            tuple(
                SaveDestination(kind, selected[kind])
                for kind in (
                    ArtifactKind.ANALYSIS,
                    ArtifactKind.POST_ANALYSIS,
                    ArtifactKind.DATA,
                )
                if kind in selected
            )
        )

    def _export_image(
        self,
        tab_id: str,
        destination: SaveDestination,
        *,
        capture_signature: bool = True,
    ) -> None:
        tab = self._state.get_tab(tab_id)
        figure = (
            tab.analysis.figure
            if destination.kind is ArtifactKind.ANALYSIS
            else tab.post_analysis.figure
        )
        if figure is None:
            label = (
                "post-analysis figure"
                if destination.kind is ArtifactKind.POST_ANALYSIS
                else "figure"
            )
            raise FailedPreconditionError(f"No {label} available to save")
        self._ensure_parent_directory(destination.path)
        tracker = tab.artifacts
        if capture_signature:
            tracker.started(destination.kind)
        try:
            save_figure_to_path(figure, destination.path)
        except Exception:
            tracker.failed(destination.kind)
            raise
        tracker.succeeded(destination.kind, destination.path)

    def save_image_sync(self, permit: SavePermit, image_path: str) -> None:
        tab_id = permit.tab_id
        self._require_tab_idle(tab_id)
        self._state.get_artifact_snapshots(tab_id)
        self._export_image(tab_id, SaveDestination(ArtifactKind.ANALYSIS, image_path))

    def save_post_image_sync(self, permit: SavePermit, image_path: str) -> None:
        """Synchronously export the independently owned post-analysis figure."""
        tab_id = permit.tab_id
        self._require_tab_idle(tab_id)
        self._state.get_artifact_snapshots(tab_id)
        self._export_image(
            tab_id, SaveDestination(ArtifactKind.POST_ANALYSIS, image_path)
        )

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
    ) -> SaveDataRequest:
        # Run-result presence is proven by the SavePermit; tab-busy is the
        # dynamic check that stays at the operation boundary.
        if self._state.is_tab_busy(tab_id):
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")

        tab = self._state.get_tab(tab_id)
        ctx = self._state.exp_context
        req = SaveDataRequest(
            run_result=tab.run.result,
            data_path=data_path,
            md=ctx.md,
            ml=ctx.ml,
            chip_name=ctx.chip_name,
            qub_name=ctx.qub_name,
            res_name=ctx.res_name,
            active_label=ctx.active_label,
            comment=comment,
        )
        return req

    def _mark_saving(
        self,
        tab_id: str,
        saving_data: bool,
        fact: TabInteractionFact,
    ) -> None:
        self._state.set_tab_saving_data(tab_id, saving_data)
        self._bus.emit(
            TabInteractionChangedPayload(tab_id=tab_id, fact=fact),
        )

    def _finish_save_state(
        self, tab_id: str, error: Exception | None
    ) -> SaveDataFinishedPayload:
        path = self._active_paths.pop(tab_id, "")
        self._active_operations.pop(tab_id, None)
        tracker = self._state.get_tab(tab_id).artifacts
        if error is None:
            logger.info("save finished: tab_id=%r path=%r", tab_id, path)
            tracker.succeeded(ArtifactKind.DATA, path)
        else:
            logger.warning(
                "save failed: tab_id=%r path=%r error=%r", tab_id, path, error
            )
            tracker.failed(ArtifactKind.DATA)
        self._state.set_tab_saving_data(tab_id, False)
        return SaveDataFinishedPayload(
            tab_id=tab_id, data_path=path, error=None if error is None else str(error)
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
