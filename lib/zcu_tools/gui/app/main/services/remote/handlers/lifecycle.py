"""Noninteractive close policy evaluated within one GUI owner dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zcu_tools.gui.app.main.artifact_tracker import ArtifactKind, SaveStatus
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


def require_idle(adapter: RemoteControlAdapter, tab_id: str | None = None) -> None:
    active = [
        {"tab": op.tab, "kind": op.kind}
        for op in adapter.operation_control.active_operations()
        if tab_id is None or op.tab == tab_id
    ]
    if active:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            "Wait for active operations before closing.",
            reason="busy",
            data={"operations": active},
        )


def require_saved(adapter: RemoteControlAdapter, tab_ids: list[str]) -> None:
    keys = {
        ArtifactKind.DATA: "data",
        ArtifactKind.ANALYSIS: "analysis",
        ArtifactKind.POST_ANALYSIS: "post",
    }
    unsaved = []
    descriptions = []
    for tab_id in tab_ids:
        artifacts = [
            keys[item.kind]
            for item in adapter.tab_control.get_tab_snapshot(tab_id).artifacts
            if item.status in (SaveStatus.NOT_SAVED, SaveStatus.UNSAVED_CHANGES)
        ]
        if artifacts:
            unsaved.append({"tab": tab_id, "artifacts": artifacts})
            descriptions.append(f"{tab_id}: {', '.join(artifacts)}")
    if unsaved:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"Unsaved artifacts ({'; '.join(descriptions)}). "
            "Save them or explicitly set discard_unsaved=true.",
            reason="unsaved",
            data={"unsaved": unsaved},
        )
