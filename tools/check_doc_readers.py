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
    found: list[Violation] = []
    for path, source in sorted(sources.items()):
        header = "\n".join(source.splitlines()[:8])
        if "**讀者：**" not in header and "**Readers:**" not in header:
            found.append(
                Violation(path, "missing reader declaration in first 8 body lines")
            )
    return tuple(found)
