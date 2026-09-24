"""Check that no Python source file exceeds the module size limit.

Ruff's C901 and PLR0915 constrain functions, not files: a module can hold five
hundred simple functions and satisfy every per-function rule while being
impossible to navigate. This is the only check that guards file size, so it is
the only thing standing between the tree and another five-thousand-line module.

The limit states the standard rather than describing the tree. Existing files
above it are read through the ratchet (tools/check_ratchet.py), which compares a
candidate against its base rather than against zero.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

import _support

LINE_LIMIT: Final = 1000


@dataclass(frozen=True)
class OversizeFile:
    """A Python file longer than the limit."""

    path: PurePosixPath
    lines: int


def line_count(path: Path) -> int:
    """Return the number of lines in `path`, counting a final unterminated line."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def oversize_files(
    root: Path, *, limit: int = LINE_LIMIT, paths: Iterable[str] | None = None
) -> tuple[OversizeFile, ...]:
    """Return every checked Python file longer than `limit`, longest first.

    `paths` narrows the walk to named files. The ratchet passes the candidate's
    changed files: a file can only cross the limit by being edited, so scanning
    the rest costs time and finds nothing new.
    """
    found = [
        OversizeFile(
            path=PurePosixPath(path.relative_to(root).as_posix()),
            lines=line_count(path),
        )
        for path in _support.python_files(root, paths=paths)
    ]
    return tuple(
        sorted(
            (item for item in found if item.lines > limit),
            key=lambda item: (-item.lines, str(item.path)),
        )
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    found = oversize_files(root)
    receipt = {
        "limit": LINE_LIMIT,
        "status": "FAIL" if found else "PASS",
        "violation_count": len(found),
        "violations": [{"lines": item.lines, "path": str(item.path)} for item in found],
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    for item in found:
        print(f"{item.path}: {item.lines} lines > {LINE_LIMIT}", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
