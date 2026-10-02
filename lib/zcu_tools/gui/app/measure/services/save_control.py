"""App-facing save control facet for UI and remote driving adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol

from zcu_tools.gui.app.measure.artifact_tracker import ArtifactKind
from zcu_tools.gui.app.measure.catalog import ExperimentAccess
from zcu_tools.gui.app.measure.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.app.measure.figure_export import resolve_figure_path
from zcu_tools.gui.expected_error import FailedPreconditionError

from .ports import SaveDestination
from .save import resolve_artifact_destinations

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.state import State
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus

    from .guard import GuardService
    from .ports import SaveArtifactsSubmission, SaveDataSubmission
    from .save import SaveService
    from .tab import TabService


class SaveControlPort(Protocol):
    """App-facing save surface for driving adapters."""

    def has_tab(self, tab_id: str) -> bool: ...

    def set_comment(self, tab_id: str, comment: str) -> None: ...

    def save_data(
        self, tab_id: str, data_path: str | None = None, comment: str | None = None
    ) -> SaveDataSubmission: ...

    def save_artifacts(
        self,
        tab_id: str,
        *,
        artifacts: tuple[ArtifactKind, ...] | None = None,
        paths: Mapping[ArtifactKind, str] | None = None,
        comment: str | None = None,
    ) -> SaveArtifactsSubmission: ...

    def save_image(
        self,
        tab_id: str,
        image_path: str | None = None,
        *,
        operation_id: int | None = None,
    ) -> str: ...

    def save_post_image(
        self,
        tab_id: str,
        image_path: str | None = None,
        *,
        operation_id: int | None = None,
    ) -> str: ...


class SaveControlFacet:
    """Composite adapter over save guards, save service, and save path state."""

    def __init__(
        self,
        *,
        state: State,
        bus: EventBus,
        guard: GuardService,
        tab: TabService,
        save: SaveService,
        notify_info: Callable[[str], None],
        access: ExperimentAccess | None = None,
    ) -> None:
        self._state = state
        self._bus = bus
        self._guard = guard
        self._tab = tab
        self._save = save
        self._notify_info = notify_info
        self._access = access if access is not None else ExperimentAccess()

    def has_tab(self, tab_id: str) -> bool:
        return self._state.has_tab(tab_id)

    def set_comment(self, tab_id: str, comment: str) -> None:
        self._state.update_tab_comment(tab_id, comment)

    def save_data(
        self, tab_id: str, data_path: str | None = None, comment: str | None = None
    ) -> SaveDataSubmission:
        if data_path is not None and not data_path.strip():
            raise FailedPreconditionError(f"Tab {tab_id!r} has an empty data path")
        permit = self._guard.acquire_save_permit(tab_id)
        self._require_tab_idle(tab_id)
        if data_path is not None:
            self._tab.update_tab_data_path_override(tab_id, data_path)
        if comment is not None:
            self._state.update_tab_comment(tab_id, comment)
        resolved = self._tab.get_tab_data_path(tab_id)
        if not resolved:
            raise FailedPreconditionError(f"Tab {tab_id!r} has no data path configured")
        draft_comment = self._state.get_tab(tab_id).save.comment
        return self._save.start_save_data(permit, resolved, comment=draft_comment)

    def save_artifacts(
        self,
        tab_id: str,
        *,
        artifacts: tuple[ArtifactKind, ...] | None = None,
        paths: Mapping[ArtifactKind, str] | None = None,
        comment: str | None = None,
    ) -> SaveArtifactsSubmission:
        permit = self._guard.acquire_save_permit(tab_id)
        self._require_tab_idle(tab_id)
        available = {a.kind: a for a in self._state.get_artifact_snapshots(tab_id)}
        selected = (
            tuple(kind for kind, a in available.items() if a.is_saveable)
            if artifacts is None
            else artifacts
        )
        if not selected or len(set(selected)) != len(selected):
            raise FailedPreconditionError(
                "Save requires a nonempty unique artifact set"
            )
        overrides = paths if paths is not None else {}
        if set(overrides) - set(selected):
            raise FailedPreconditionError("Save paths must name selected artifacts")
        destinations = []
        for kind in selected:
            if kind not in available or not available[kind].is_saveable:
                raise FailedPreconditionError(f"Artifact {kind.value} is not saveable")
            path = overrides.get(kind, available[kind].default_path)
            if path is None or not path.strip():
                raise FailedPreconditionError(
                    f"Artifact {kind.value} has an empty path"
                )
            destinations.append(SaveDestination(kind, path))
        resolve_artifact_destinations(tuple(destinations))
        setters = {
            ArtifactKind.DATA: self._tab.update_tab_data_path_override,
            ArtifactKind.ANALYSIS: self._tab.update_tab_analysis_image_path_override,
            ArtifactKind.POST_ANALYSIS: self._tab.update_tab_post_analysis_image_path_override,
        }
        for kind, path in overrides.items():
            setters[kind](tab_id, path)
        if comment is not None:
            self._state.update_tab_comment(tab_id, comment)
        if overrides or comment is not None:
            self._bus.emit(
                TabInteractionChangedPayload(
                    tab_id, TabInteractionFact.SAVE_DRAFT_COMMITTED
                )
            )
        return self._save.start_save_artifacts(
            permit, tuple(destinations), self._state.get_tab(tab_id).save.comment
        )

    def save_image(
        self,
        tab_id: str,
        image_path: str | None = None,
        *,
        operation_id: int | None = None,
    ) -> str:
        if operation_id is not None:
            self._state.require_analysis_operation(tab_id, "analysis", operation_id)
        if image_path is not None and not image_path.strip():
            raise FailedPreconditionError(f"Tab {tab_id!r} has an empty image path")
        permit = self._guard.acquire_save_permit(tab_id)
        self._require_tab_idle(tab_id)
        if image_path is not None:
            self._tab.update_tab_analysis_image_path_override(tab_id, image_path)
            self._bus.emit(
                TabInteractionChangedPayload(
                    tab_id, TabInteractionFact.SAVE_DRAFT_COMMITTED
                )
            )
        resolved = self._tab.get_tab_analysis_image_path(tab_id)
        if resolved is None:
            raise FailedPreconditionError(
                f"Tab {tab_id!r} has no analysis image path configured"
            )
        resolved = resolve_figure_path(resolved)
        self._save.save_image_sync(permit, resolved)
        self._notify_info(f"Image saved to {resolved}")
        return resolved

    def save_post_image(
        self,
        tab_id: str,
        image_path: str | None = None,
        *,
        operation_id: int | None = None,
    ) -> str:
        if operation_id is not None:
            self._state.require_analysis_operation(
                tab_id, "post_analysis", operation_id
            )
        if image_path is not None and not image_path.strip():
            raise FailedPreconditionError(
                f"Tab {tab_id!r} has an empty post image path"
            )
        permit = self._guard.acquire_save_permit(tab_id)
        self._require_tab_idle(tab_id)
        if image_path is not None:
            self._tab.update_tab_post_analysis_image_path_override(tab_id, image_path)
            self._bus.emit(
                TabInteractionChangedPayload(
                    tab_id, TabInteractionFact.SAVE_DRAFT_COMMITTED
                )
            )
        resolved = self._tab.get_tab_post_analysis_image_path(tab_id)
        if resolved is None:
            raise FailedPreconditionError(
                f"Tab {tab_id!r} has no post-analysis image path configured"
            )
        resolved = resolve_figure_path(resolved)
        self._save.save_post_image_sync(permit, resolved)
        self._notify_info(f"Post-analysis image saved to {resolved}")
        return resolved

    def _require_tab_idle(self, tab_id: str) -> None:
        self._access.require_available()
        if self._state.is_tab_busy(tab_id):
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")
