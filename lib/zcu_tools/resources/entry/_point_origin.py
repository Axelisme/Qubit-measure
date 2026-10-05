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
_sources: WeakKeyDictionary[PointView, PointOrigin] = WeakKeyDictionary()


def register(view: PointView, origin: PointOrigin) -> None:
    """Record ResultEntry's source identity, without exposing a snapshot API."""
    _sources[view] = origin


def clone_source(
    view: PointView, *, result_root: Path, database_root: Path
) -> PointOrigin:
    """Return the original identity if both resolved roots match.

    Manually constructed views lack this identity and raise ValueError, as do
    views bound to another pair of roots. The caller reloads the published point.
    """
    origin = _sources.get(view)
    if origin is None:
        raise ValueError("PointView has no ResultEntry source identity")
    if (
        origin.result_root != result_root.resolve()
        or origin.database_root != database_root.resolve()
    ):
        raise ValueError("PointView result/database roots do not match")
    return origin
