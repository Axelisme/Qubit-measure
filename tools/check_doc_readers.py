"""Check tracked Markdown reader declarations and module README dates."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Final

_ALLOWED_READERS: Final = frozenset({"operator", "lab-author", "framework-dev", "agent"})
_LAST_UPDATED: Final = re.compile(
    r"^\*\*Last updated:\*\* \d{4}-\d{2}-\d{2}\b", re.MULTILINE
)
_DECLARATION: Final = re.compile(r"^\*\*(?:讀者：|Readers:)\*\*[ \t]*(.*)", re.MULTILINE)


@dataclass(frozen=True)
class Violation:
    """A rejected document: repo-relative POSIX path and human-readable reason."""

    path: str
    reason: str


def check_sources(sources: Mapping[str, str]) -> tuple[Violation, ...]:
    """Check repo-relative Markdown paths mapped to UTF-8 document text.

    Skip ADR/discussion snapshots, except docs/adr/README.md and
    docs/discussions/INDEX.md. After optional --- YAML front matter, check the
    first eight body lines for a Chinese or English declaration. Reader names
    must be operator, lab-author, framework-dev or agent, separated by commas or
    Chinese enumeration commas, before the first period. Module READMEs under
    lib/ or tests/ also need **Last updated:** YYYY-MM-DD somewhere in the body.

    Return path-sorted violations without reading files or changing the mapping.
    Raise ValueError if opening YAML front matter has no closing --- line.
    """
    found: list[Violation] = []
    for path, source in sorted(sources.items()):
        if (
            path.startswith("docs/adr/")
            and path != "docs/adr/README.md"
            or path.startswith("docs/discussions/")
            and path != "docs/discussions/INDEX.md"
        ):
            continue
        lines = source.splitlines()
        if lines and lines[0] == "---":
            lines = lines[lines.index("---", 1) + 1 :]
        document = PurePosixPath(path)
        if (
            document.parts[0] in {"lib", "tests"}
            and document.name == "README.md"
            and _LAST_UPDATED.search("\n".join(lines)) is None
        ):
            found.append(
                Violation(path, "missing or malformed **Last updated:** YYYY-MM-DD")
            )
        header = "\n".join(lines[:8])
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


def main(argv: list[str] | None = None) -> int:
    """Scan tracked Markdown under --root, defaulting to this checkout.

    Print a check_file_size-style JSON receipt to stdout and path/reason lines
    to stderr. Return 0 for PASS or 1 for violations. argv=None reads sys.argv;
    Git, decoding and file-read errors propagate rather than reporting PASS.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args(argv).root
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", "*.md"],
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    sources = {
        path: (root / path).read_text(encoding="utf-8")
        for path in tracked.split("\0")
        if path
    }
    found = check_sources(sources)
    receipt = {
        "status": "FAIL" if found else "PASS",
        "violation_count": len(found),
        "violations": [asdict(item) for item in found],
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    for item in found:
        print(f"{item.path}: {item.reason}", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
