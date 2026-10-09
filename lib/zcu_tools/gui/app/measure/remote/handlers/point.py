"""Entry/point wire projection through the session context owner."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, TypedDict

from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    PointBindingError,
    PointCreateRequest,
    PointUseRequest,
)

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


class PointEntry(TypedDict):
    """Opened entry: UUID, directory name and exact absolute owner roots."""

    entry_id: str
    name: str
    result_path: str
    database_path: str


class EntryListReply(TypedDict):
    """entries contains refreshed persisted entry directory names."""

    entries: list[str]


class EntryInfoReply(TypedDict):
    """entry is the cached opened identity, or None before opening."""

    entry: PointEntry | None


def h_entry_list(
    adapter: RemoteControlAdapter, _params: Mapping[str, object]
) -> EntryListReply:
    """Refresh entry names; no binding change or full-context observation.

    params is empty. Owner discovery/schema/I/O failures propagate.
    """
    raise NotImplementedError("entry.list wiring")


def h_entry_info(
    adapter: RemoteControlAdapter, _params: Mapping[str, object]
) -> EntryInfoReply:
    """Read cached EntryInfo as {entry}, with None before opening.

    params is empty. Paths are absolute owner roots converted to strings.
    This metadata-only read establishes no full-context observation.
    """
    raise NotImplementedError("entry.info wiring")


def h_entry_new(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> PointEntry:
    """Create required string name through SetupControlPort and publish DRAFT.

    Return the exact EntryInfo fields with path strings. Invalid wire types
    raise INVALID_PARAMS. Owner name/existence/I/O errors propagate and preserve
    the prior binding. Post-creation installation failure retains the entry and
    names Open as recovery; never retry or delete it. No hardware side effects.
    """
    raise NotImplementedError("entry.new wiring")


def h_entry_open(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> PointEntry:
    """Open required string name through SetupControlPort and publish DRAFT.

    Return the exact EntryInfo fields with path strings. Invalid wire types
    raise INVALID_PARAMS. Owner invalid/missing/I/O/install errors propagate and
    preserve the prior binding. Never creates resources or touches hardware.
    """
    raise NotImplementedError("entry.open wiring")


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
    adapter: RemoteControlAdapter, _params: Mapping[str, object]
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
    """Create and select label in the currently opened entry.

    Required label is a string. description defaults to empty text,
    explicitly replacing seed/clone metadata. clone_from is None or a same-entry
    label; None copies setup, never the unbound DRAFT.
    Return the SavedPoint destination. Malformed wire input is INVALID_PARAMS.
    Owner no_entry/schema/source/I/O failures retain their classification.
    Created-but-uninstalled PointBindingError becomes CONTROLLER_ERROR with
    reason point_binding_failed and data entry_id/point/created=True. Keep the
    original cause and point; Use is recovery, never retry or delete New.
    Existing tabs retain their parameter capture; new tabs use this point.
    """
    label = params.get("label")
    description = params.get("description", "")
    clone_from = params.get("clone_from")
    if not isinstance(label, str):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "'label' must be a string")
    if not isinstance(description, str):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "'description' must be a string")
    if clone_from is not None and not isinstance(clone_from, str):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, "'clone_from' must be a string or null"
        )
    request = PointCreateRequest(
        label=label,
        description=description,
        origin=CommandOrigin("mcp"),
        clone_from=clone_from,
    )
    try:
        point = adapter.context_control.new_point(request)
    except PointBindingError as exc:
        raise RemoteError(
            ErrorCode.CONTROLLER_ERROR,
            str(exc),
            reason="point_binding_failed",
            data={"entry_id": exc.entry_id, "point": exc.label, "created": True},
        ) from exc
    return {"kind": point.kind, "entry_id": point.entry_id, "point": point.point}


def h_point_use(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> SavedPointReply:
    """Select required label in the currently opened entry.

    label is a string. Return the selected SavedPoint destination.
    Malformed wire input is INVALID_PARAMS. Owner no_entry/schema/source/I/O
    failures propagate without fallback and leave the old binding intact.
    Existing tabs retain their parameter capture; new tabs use this point.
    """
    label = params.get("label")
    if not isinstance(label, str):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "'label' must be a string")
    point = adapter.context_control.use_point(PointUseRequest(label))
    return {"kind": point.kind, "entry_id": point.entry_id, "point": point.point}
