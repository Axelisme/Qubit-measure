"""Entry/point wire projection through the session context owner."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, TypedDict

from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.context_types import PointUseRequest

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


class PointEntry(TypedDict):
    """Opened entry: UUID, directory name and exact absolute owner roots."""

    entry_id: str
    name: str
    result_path: str
    database_path: str


class PointSummary(TypedDict):
    """Persisted point label and description; absent descriptions are empty."""

    label: str
    description: str


class PointListReply(TypedDict):
    """entry is None before opening; points lists refreshed persisted metadata."""

    entry: PointEntry | None
    points: list[PointSummary]


class SavedPointReply(TypedDict):
    """Persisted destination, not a binding generation or writable handle.

    kind is always saved_point. entry_id is the opened entry UUID.
    point is the owner's validated single-segment label.
    """

    kind: Literal["saved_point"]
    entry_id: str
    point: str


def h_point_list(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> PointListReply:
    """Return entry roots and refreshed label/descriptions, not context content.

    params is empty. No entry returns entry=None and points=[].
    Owner refresh/schema/I/O errors propagate; never fall back to cached success.
    This metadata-only read establishes no full-context observation.
    """
    info = adapter.context_control.get_entry_info()
    points = adapter.context_control.list_points(refresh=True)
    entry: PointEntry | None = None
    if info is not None:
        entry = {
            "entry_id": info.entry_id,
            "name": info.name,
            "result_path": str(info.result_path),
            "database_path": str(info.database_path),
        }
    return {
        "entry": entry,
        "points": [
            {"label": point.label, "description": point.description} for point in points
        ],
    }


def h_point_new(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> SavedPointReply:
    """Create and select label in the opened entry identified by entry_id.

    Required entry_id/label are strings. description defaults to empty text,
    explicitly replacing seed/clone metadata. clone_from is None or a same-entry
    label; None copies setup, never the unbound DRAFT.
    Return the SavedPoint destination. Malformed wire input is INVALID_PARAMS.
    Owner entry mismatch/schema/source/I/O failures retain their classification.
    Created-but-uninstalled PointBindingError becomes CONTROLLER_ERROR with
    reason point_binding_failed and data entry_id/point/created=True. Keep the
    original cause and point; Use is recovery, never retry or delete New.
    Existing tabs retain their parameter capture; new tabs use this point.
    """
    raise NotImplementedError("point.new owner wiring")


def h_point_use(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> SavedPointReply:
    """Select required label in the opened entry identified by entry_id.

    Both params are strings. Return the selected SavedPoint destination.
    Malformed wire input is INVALID_PARAMS. Owner entry/schema/source/I/O
    failures propagate without fallback and leave the old binding intact.
    Existing tabs retain their parameter capture; new tabs use this point.
    """
    entry_id = params.get("entry_id")
    label = params.get("label")
    if not isinstance(entry_id, str):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "'entry_id' must be a string")
    if not isinstance(label, str):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "'label' must be a string")
    point = adapter.context_control.use_point(PointUseRequest(entry_id, label))
    return {"kind": point.kind, "entry_id": point.entry_id, "point": point.point}
