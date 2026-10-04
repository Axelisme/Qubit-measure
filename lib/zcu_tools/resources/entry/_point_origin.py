"""Private point identity lookup; no notebook-facing PointView method."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

if TYPE_CHECKING:
    from .points import PointView

# Weak keys keep identity only for the lifetime of the corresponding view.
_sources: WeakKeyDictionary[PointView, Path] = WeakKeyDictionary()


def register(view: PointView, source: Path) -> None:
    """Called only by PointView construction."""
    _sources[view] = source


def clone_source(view: PointView, entry: Path) -> Path:
    """ResultEntry's clone caller holds entry's lock before this identity check."""
    source = _sources[view]
    if source.resolve().parents[2] != entry.resolve():
        raise ValueError("Cross-entry cloning is not supported")
    return source.parent
