"""App-facing save control facet for UI and remote driving adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol

from zcu_tools.gui.app.measure.artifact_tracker import ArtifactKey, ArtifactKind
from zcu_tools.gui.app.measure.catalog import ExperimentAccess
from zcu_tools.gui.app.measure.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
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
        artifacts: tuple[ArtifactKey, ...] | None = None,
        paths: Mapping[ArtifactKey, str] | None = None,
        comment: str | None = None,
    ) -> SaveArtifactsSubmission: ...

    def save_image(
        self,
        tab_id: str,
        key: ArtifactKey,
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
        artifacts: tuple[ArtifactKey, ...] | None = None,
        paths: Mapping[ArtifactKey, str] | None = None,
        comment: str | None = None,
    ) -> SaveArtifactsSubmission:
        permit = self._guard.acquire_save_permit(tab_id)
        self._require_tab_idle(tab_id)
        available = {a.key: a for a in self._state.get_artifact_snapshots(tab_id)}
        overrides = paths if paths is not None else {}
        data_key = ArtifactKey(ArtifactKind.DATA)
        if artifacts is None:
            data_draft_changed = (
                data_key in overrides
                and overrides[data_key] != available[data_key].default_path
            ) or (
                comment is not None
                and comment != self._state.get_tab(tab_id).save.comment
            )
            selected = tuple(
                key
                for key, artifact in available.items()
                if artifact.needs_save
                or (key == data_key and artifact.is_saveable and data_draft_changed)
            )
        else:
            selected = artifacts
        if not selected or len(set(selected)) != len(selected):
            raise FailedPreconditionError(
                "Save requires a nonempty unique artifact set"
            )
        if set(overrides) - set(selected):
            raise FailedPreconditionError("Save paths must name selected artifacts")
        destinations = []
        for key in selected:
            if key not in available or not available[key].is_saveable:
                raise FailedPreconditionError(f"Artifact {key!r} is not saveable")
            path = overrides.get(key, available[key].default_path)
            if path is None or not path.strip():
                raise FailedPreconditionError(f"Artifact {key!r} has an empty path")
            destinations.append(SaveDestination(key, path))
        resolve_artifact_destinations(tuple(destinations))
        for key, path in overrides.items():
            if key.kind is ArtifactKind.DATA:
                self._tab.update_tab_data_path_override(tab_id, path)
            else:
                self._tab.update_tab_image_path_override(tab_id, key, path)
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
        key: ArtifactKey,
        image_path: str | None = None,
        *,
        operation_id: int | None = None,
    ) -> str:
        if key.kind is ArtifactKind.DATA:
            raise FailedPreconditionError("Data is not an image artifact")
        if operation_id is not None:
            self._state.require_analysis_operation(
                tab_id,
                "analysis" if key.kind is ArtifactKind.ANALYSIS else "post_analysis",
                operation_id,
            )
        if image_path is not None and not image_path.strip():
            raise FailedPreconditionError(f"Tab {tab_id!r} has an empty image path")
        permit = self._guard.acquire_save_permit(tab_id)
        self._require_tab_idle(tab_id)
        artifacts = {a.key: a for a in self._state.get_artifact_snapshots(tab_id)}
        if key not in artifacts or not artifacts[key].is_saveable:
            raise FailedPreconditionError(f"Artifact {key!r} is not saveable")
        path = image_path if image_path is not None else artifacts[key].default_path
        if path is None or not path.strip():
            raise FailedPreconditionError(
                f"Tab {tab_id!r} has no image path configured"
            )
        resolved = resolve_artifact_destinations(
            (SaveDestination(key=key, path=path),)
        )[0].path
        if image_path is not None:
            self._tab.update_tab_image_path_override(tab_id, key, image_path)
            self._bus.emit(
                TabInteractionChangedPayload(
                    tab_id, TabInteractionFact.SAVE_DRAFT_COMMITTED
                )
            )
        self._save.save_image_sync(permit, key, resolved)
        self._notify_info(f"Image saved to {resolved}")
        return resolved

    def _require_tab_idle(self, tab_id: str) -> None:
        self._access.require_available()
        if self._state.is_tab_busy(tab_id):
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")
