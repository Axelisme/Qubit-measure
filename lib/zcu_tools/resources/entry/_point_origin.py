"""Immutable point identity and clone checks; no view methods or identity table."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .points import PointView


@dataclass(frozen=True)
class PointOrigin:
    """Complete owning identity supplied once when constructing a PointView.

    source is the bound point.yaml path and entry_id its owning entry's immutable
    UUID. result_root and database_root are the resolved roots supplied by the
    constructor. Identity does not depend on later ResultEntry registration.
    """

    source: Path
    entry_id: str
    result_root: Path
    database_root: Path


def clone_source(
    view: PointView, *, result_root: Path, database_root: Path
) -> PointOrigin:
    """Return this view's constructor identity if both destination roots match.

    Root mismatch raises ValueError without I/O. The caller reloads the published
    point and checks the source entry's UUID before cloning; no cached view data
    or destination identity is used to invent an origin.
    """
    # Import at use time: views owns bindings and imports the identity type.
    from .views import point_origin

    origin = point_origin(view)
    if (
        origin.result_root != result_root.resolve()
        or origin.database_root != database_root.resolve()
    ):
        raise ValueError("PointView result/database roots do not match")
    return origin
