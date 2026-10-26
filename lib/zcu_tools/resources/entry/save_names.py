"""Caller-controlled stems shared by save frontends."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from zcu_tools.datafile import JsonObject

from .aliases import AliasValue


@dataclass(frozen=True, kw_only=True)
class SaveContext:
    """Already-captured facts available to a save_stem callback.

    name is the full GUI adapter path, notebook tag or workflow name.
    run_id is the complete historical acquisition/workflow identity.
    point is the historical work-point label, or None.
    roles maps captured aliases to names or ordered name lists.
    cfg holds JSON configuration values, or None when unavailable.
    saved_at is the timezone-aware save instant, not acquisition time.
    Callbacks return a nonempty safe filename stem, without extension;
    SaveLayout rejects invalid path segments rather than normalizing them.
    No callback may mutate the captured facts.
    """

    name: str
    run_id: str
    point: str | None
    roles: Mapping[str, AliasValue]
    cfg: JsonObject | None
    saved_at: datetime


def default_save_stem(context: SaveContext) -> str:
    """Join dotted name, ordered role names and the complete run_id.

    Flatten ordered role lists in place, join names with hyphens, and omit
    the role token when no names are bound. Do not add collision suffixes.
    The caller passes the returned stem unchanged to SaveLayout.
    """
    names = [
        name
        for value in context.roles.values()
        for name in ([value] if isinstance(value, str) else value)
        if name
    ]
    parts = [context.name.replace("/", ".")]
    if names:
        parts.append("-".join(names))
    parts.append(context.run_id)
    return "_".join(parts)
