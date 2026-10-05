"""Private point identity lookup; no notebook-facing PointView method."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

if TYPE_CHECKING:
    from .points import PointView


@dataclass(frozen=True)
class PointOrigin:
    """Owning entry identity and caller roots for a ResultEntry-created view.

    source is point.yaml, entry_id the immutable UUID, result_root and
    database_root the resolved roots used to open the owning entry.
    """

    source: Path
    entry_id: str
    result_root: Path
    database_root: Path


# Weak keys keep identity only for the lifetime of the corresponding view.
_sources: WeakKeyDictionary[PointView, PointOrigin | Path] = WeakKeyDictionary()


def register(view: PointView, origin: PointOrigin | Path) -> None:
    """Associate view with its point.yaml Path or full ResultEntry PointOrigin.

    Construction records only the path, permitting same-entry clone. ResultEntry
    replaces that path with full identity to permit foreign clone. No I/O runs.
    """
    _sources[view] = origin


def clone_source(
    view: PointView,
    *,
    result_root: Path,
    database_root: Path,
    entry_path: Path,
    entry_id: str,
) -> PointOrigin:
    """Return a verified source identity for clone into entry_path/entry_id.

    A constructor-only path can clone within that same resolved entry. Foreign
    views require ResultEntry identity and both roots must match. The caller
    reloads the published point instead of using the view snapshot.
    """
    origin = _sources.get(view)
    if isinstance(origin, Path):
        if origin.resolve().parents[2] != entry_path.resolve():
            raise ValueError("PointView has no ResultEntry source identity")
        return PointOrigin(
            origin, entry_id, result_root.resolve(), database_root.resolve()
        )
    if origin is None:
        raise ValueError("PointView has no ResultEntry source identity")
    if (
        origin.result_root != result_root.resolve()
        or origin.database_root != database_root.resolve()
    ):
        raise ValueError("PointView result/database roots do not match")
    return origin
