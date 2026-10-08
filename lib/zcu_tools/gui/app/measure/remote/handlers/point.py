"""Entry/point wire projection through the session context owner."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, TypedDict

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
    raise NotImplementedError("point.list owner wiring")


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
    raise NotImplementedError("point.use owner wiring")
