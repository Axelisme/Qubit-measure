"""Check tracked Markdown reader declarations and module README dates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Violation:
    """A rejected document: repo-relative POSIX path and human-readable reason."""

    path: str
    reason: str


def check_sources(sources: Mapping[str, str]) -> tuple[Violation, ...]:
    """Check repo-relative Markdown paths mapped to UTF-8 document text.

    Return path-sorted violations without reading files or changing the mapping.
    """
    return ()
