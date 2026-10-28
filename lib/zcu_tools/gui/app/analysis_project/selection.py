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
    A component without an entry is invalid.
    """

    project_root: str | Path
    entry: str = ""
    component: str | None = None
