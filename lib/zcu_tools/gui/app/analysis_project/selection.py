"""Unresolved entry choices shared by analysis GUI composition and CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalysisSelection:
    """Requested target before component registration or filesystem access.

    project_root contains result/ and Database/ and may be relative to cwd.
    entry names an existing Result entry; empty leaves the GUI unopened.
    component names a fluxonium component, or None resolves its qubit alias.
    Construction raises ValueError("--component requires --entry") when a
    component is provided without an entry. No registration or file access occurs.
    """

    project_root: str | Path
    entry: str = ""
    component: str | None = None

    def __post_init__(self) -> None:
        if not self.entry and self.component is not None:
            raise ValueError("--component requires --entry")
