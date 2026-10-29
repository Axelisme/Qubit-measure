"""Entry/point wire projection through the session context owner."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, TypedDict

from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryInfo,
    EntryRequest,
    PointBindingError,
    PointCreateRequest,
    PointUseRequest,
)

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


class PointEntry(TypedDict):
    """Opened entry: directory name and exact absolute owner roots."""

    name: str
    result_path: str
    database_path: str


class EntryListReply(TypedDict):
    """entries contains refreshed persisted entry directory names."""

    entries: list[str]


class PointFile(TypedDict):
    """Installed point file: document kind, absolute path and raw-byte SHA-256.

    revision is None for an absent library or aliases file.
    """

    document: Literal["parameters", "library", "aliases"]
    path: str
    revision: str | None


class EntryInfoReply(TypedDict):
    """entry is the opened identity or None; files lists installed point versions."""

    entry: PointEntry | None
    files: list[PointFile]


def h_entry_list(
    adapter: RemoteControlAdapter, _params: Mapping[str, object]
) -> EntryListReply:
    """Refresh entry names; no binding change or full-context observation.

    params is empty. Owner discovery/schema/I/O failures propagate.
    """
    return {"entries": list(adapter.ctrl.setup_control.list_entries(refresh=True))}


def h_entry_info(
    adapter: RemoteControlAdapter, _params: Mapping[str, object]
) -> EntryInfoReply:
    """Read cached entry identity and installed point files as {entry, files}.

    params is empty. entry is None before opening; files is [] without a point.
    Paths are absolute owner paths converted to strings. File revisions describe
    installed bytes, not unvalidated disk edits. No I/O or implicit refresh occurs.
    """
    info = adapter.ctrl.setup_control.get_entry_info()
    return {
        "entry": _entry_projection(info) if info is not None else None,
        "files": [
            {
                "document": item.document,
                "path": str(item.path),
                "revision": item.revision,
            }
            for item in adapter.context_control.get_file_revisions()
        ],
    }


def h_entry_new(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> PointEntry:
    """Create required string name through SetupControlPort and leave point unselected.

    Return the exact EntryInfo fields with path strings. Invalid wire types
    raise INVALID_PARAMS. Owner name/existence/I/O errors propagate and preserve
    the prior binding. Post-creation installation failure retains the entry and
    names Open as recovery; never retry or delete it. No hardware side effects.
    """
    name = params.get("name")
    if not isinstance(name, str):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "'name' must be a string")
    info = adapter.ctrl.setup_control.new_entry(EntryRequest(name))
    return _entry_projection(info)


def h_entry_open(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> PointEntry:
    """Open required string name through SetupControlPort and leave point unselected.

    Return the exact EntryInfo fields with path strings. Invalid wire types
    raise INVALID_PARAMS. Owner invalid/missing/I/O/install errors propagate and
    preserve the prior binding. Never creates resources or touches hardware.
    """
    name = params.get("name")
    if not isinstance(name, str):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "'name' must be a string")
    info = adapter.ctrl.setup_control.open_entry(EntryRequest(name))
    return _entry_projection(info)


def _entry_projection(info: EntryInfo) -> PointEntry:
    return {
        "name": info.name,
        "result_path": str(info.result_path),
        "database_path": str(info.database_path),
    }


class PointSummary(TypedDict):
    """Point label/description and independent parameter availability.

    available is True when the parameter-only owner probe passed. error is None
    on success, otherwise the failed document path and diagnostic. Library and
    alias usability is not probed. Absent/unreadable descriptions are empty.
    """

    label: str
    description: str
    available: bool
    error: str | None


class PointListReply(TypedDict):
    """entry is None before opening; points lists refreshed persisted metadata."""

    entry: PointEntry | None
    points: list[PointSummary]


class PointSelectionReply(TypedDict):
    """Selected documents: entry_name is the opened directory name.

    point is the selected single-segment label, or None for Global.
    This projection carries no binding generation or writable handle.
    """

    entry_name: str
    point: str | None


def h_point_list(
    adapter: RemoteControlAdapter, _params: Mapping[str, object]
) -> PointListReply:
    """Return entry roots, refreshed descriptions and parameter availability.

    params is empty. No entry returns entry=None and points=[].
    Point errors are isolated in each row; discovery failures propagate.
    This metadata-only read establishes no full-context observation.
    """
    info = adapter.context_control.get_entry_info()
    points = adapter.context_control.list_points(refresh=True)
    entry: PointEntry | None = None
    if info is not None:
        entry = _entry_projection(info)
    return {
        "entry": entry,
        "points": [
            {
                "label": point.label,
                "description": point.description,
                "available": point.available,
                "error": point.error,
            }
            for point in points
        ],
    }


def h_point_new(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> PointSelectionReply:
    """Create and select label in the currently opened entry.

    Required label is a string. description defaults to empty text, replacing
    inherited/clone metadata. mode defaults to inherit, copying Global documents.
    empty creates minimal parameters without library/aliases. clone requires a
    same-entry clone_from label; other modes reject clone_from.
    Return {entry_name, point}. Malformed wire input is INVALID_PARAMS.
    Unsafe destination labels raise the owner's typed input error, projected by
    shared dispatch as INVALID_PARAMS with reason point_label_invalid. Setup and
    clone rejection precedes source reads/writes and preserves selection/files.
    Existing destinations raise the owner's typed input error, projected as
    INVALID_PARAMS with the original exists message and no structured data.
    Collision rejection preserves selection/files for setup and clone alike.
    Owner no_entry/schema/source/other I/O failures retain their classification.
    Created-but-uninstalled PointBindingError becomes CONTROLLER_ERROR with
    reason point_binding_failed and data entry_name/point/created=True. Keep the
    original cause and point; Use is recovery, never retry or delete New.
    Existing tabs retain their parameter capture; new tabs use this point.
    """
    mode = params.get("mode", "inherit")
    if mode not in ("inherit", "empty", "clone"):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, "mode must be inherit, empty or clone"
        )
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
        mode=mode,
        clone_from=clone_from,
    )
    try:
        point = adapter.context_control.new_point(request)
    except PointBindingError as exc:
        raise RemoteError(
            ErrorCode.CONTROLLER_ERROR,
            str(exc),
            reason="point_binding_failed",
            data={
                "entry_name": exc.entry_name,
                "point": exc.point,
                "created": exc.created,
            },
        ) from exc
    return {"entry_name": point.entry_name, "point": point.point}


def h_point_use(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> PointSelectionReply:
    """Select required string label or None for the opened entry Global set.

    Return {entry_name, point}, with point=None for Global. Missing label or
    malformed wire input is INVALID_PARAMS. Owner no_entry/schema/source/I/O
    failures propagate without fallback and leave the old binding intact.
    Existing tabs retain their parameter capture; new tabs use this selection.
    """
    if "label" not in params:
        raise RemoteError(ErrorCode.INVALID_PARAMS, "missing 'label'")
    label = params["label"]
    if label is None:
        target = adapter.context_control.use_global()
        return {"entry_name": target.entry_name, "point": None}
    if not isinstance(label, str):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "'label' must be a string or null")
    point = adapter.context_control.use_point(PointUseRequest(label))
    return {"entry_name": point.entry_name, "point": point.point}
