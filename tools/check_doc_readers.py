"""Check tracked Markdown reader declarations and module README dates."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

_ALLOWED_READERS: Final = frozenset({"operator", "lab-author", "framework-dev", "agent"})
_DECLARATION: Final = re.compile(r"^\*\*(?:讀者：|Readers:)\*\*[ \t]*(.*)", re.MULTILINE)


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
        declaration = _DECLARATION.search(header)
        if declaration is None:
            found.append(
                Violation(path, "missing reader declaration in first 8 body lines")
            )
        else:
            readers = re.split(r"[。.]", declaration[1], maxsplit=1)[0].strip()
            tokens = [token.strip() for token in re.split(r"[,、，]", readers)]
            if not all(token in _ALLOWED_READERS for token in tokens):
                found.append(Violation(path, "invalid reader list: " + repr(readers)))
    return tuple(found)
