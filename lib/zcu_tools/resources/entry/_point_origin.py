"""Captured point location and clone checks; no view methods or identity table."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .points import PointView


@dataclass(frozen=True)
class PointOrigin:
    """Owning identity and output configuration bound when constructing a view.

    source is the bound point.yaml path.
    result_root is the resolved root used for clone identity. database_root
    is the resolved Save output root, not a clone boundary. Identity does not
    depend on later ResultEntry registration.
    """

    source: Path
    result_root: Path
    database_root: Path


def clone_source(view: PointView, *, result_root: Path) -> PointOrigin:
    """Return this view's constructor identity if the Result root matches.

    Root mismatch raises ValueError without I/O. The caller reloads the published
    point at its current source location before cloning; no cached view data
    or destination identity is used to invent an origin.
    """
    # Import at use time: views owns bindings and imports the identity type.
    from .views import point_origin

    origin = point_origin(view)
    if origin.result_root != result_root.resolve():
        raise ValueError("PointView result roots do not match")
    return origin
