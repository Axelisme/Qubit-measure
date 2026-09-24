"""Count the escape hatches, so that using one is not free.

Every other check here counts violations. A writer who wants a green result has
two ways to get one: fix the code, or silence the finding. Silencing is cheaper,
and nothing was counting it -- a `# noqa: C901` removes a complexity finding and
leaves no trace, so the ratchet reads it as progress.

This counts the silencers themselves. Read through the ratchet, adding a `# noqa`
to clear a ruff finding trades one counted thing for another and the total does
not fall, which is the point: the escape hatch stops being free without being
forbidden. Suppression is sometimes the right answer -- Pyright cannot see
pytest's fixture registry, so conftest.py disables reportUnusedFunction on
purpose -- and a rule that banned it would only teach people to write worse code
to avoid the ban. Metering it makes the trade visible to whoever reviews the
candidate.

Configuration silences things too, and more broadly than any comment: one line in
pyproject.toml can turn a rule off for a glob of files, or for the repository.
Those are counted alongside the inline ones -- a `# noqa` per call site is at
least visible at the call site, while a per-file-ignore pattern is not visible
anywhere near the code it covers.

Comments are read with tokenize rather than matched against the file text, so a
string that merely contains "# noqa" is not counted. `cast` is resolved through
the module's imports for the same reason the capability check is: a local
function named cast is not typing.cast.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import re
import sys
import tokenize
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

import _support

# kind -> pattern anchored at the start of one comment token.
#
# Anchored, not searched: a directive only works where the tool reads it, at the
# head of its own comment. Searching anywhere in the comment would also count
# prose that merely mentions a directive -- this file's own explanation of
# `# pyright: reportFoo=false` counted as a use of it until the patterns moved.
COMMENT_SUPPRESSIONS: Final = {
    "noqa": re.compile(r"#\s*noqa", re.IGNORECASE),
    "type-ignore": re.compile(r"#\s*type:\s*ignore"),
    "pyright-ignore": re.compile(r"#\s*pyright:\s*ignore"),
    # `# pyright: reportFoo=false` turns a rule off for a whole file, which is the
    # broadest silencer available and the one most worth seeing grow.
    "pyright-file-rule": re.compile(r"#\s*pyright:\s*\w+\s*=\s*(?:false|none)", re.I),
}

CALL_SUPPRESSIONS: Final = {"typing.cast": "cast"}

_PYPROJECT: Final = "pyproject.toml"
_DISABLED_VALUES: Final = frozenset({"none", "false"})


@dataclass(frozen=True)
class Suppression:
    """How many times one file reaches for one kind of escape hatch."""

    path: PurePosixPath
    kind: str
    count: int


def comment_suppressions(source: str) -> dict[str, int]:
    """Return {kind: count} over the comments of one module."""
    found: dict[str, int] = {}
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    try:
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue
            for kind, pattern in COMMENT_SUPPRESSIONS.items():
                if pattern.match(token.string):
                    found[kind] = found.get(kind, 0) + 1
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Keep what was read. An unterminated bracket stops the tokenizer at the
        # end of the file, and discarding the whole result would let a broken file
        # hide the silencers it already carries.
        return found
    return found


def call_suppressions(tree: ast.AST) -> dict[str, int]:
    """Return {kind: count} for suppressing calls, resolved through imports."""
    bindings = _support.import_bindings(tree)
    found: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        resolved = _support.resolve_call(node.func, bindings)
        kind = CALL_SUPPRESSIONS.get(resolved) if resolved else None
        if kind is not None:
            found[kind] = found.get(kind, 0) + 1
    return found


def config_suppressions(
    root: Path, *, filename: str = _PYPROJECT
) -> tuple[Suppression, ...]:
    """Return every rule silenced by configuration rather than by a comment.

    Each entry is keyed by the setting it lives in, so adding a new glob pattern
    reads as a new entry and widening an existing one raises that entry's count.

    `filename` exists for the ratchet. It measures a base tree that has been given
    the candidate's configuration, so that enabling a rule does not read as a
    regression -- but that substitution would also hide a newly added
    per-file-ignore. The base tree keeps its own configuration under a second name
    and the ratchet points this at it.
    """
    path = root / filename
    if not path.is_file():
        return ()
    with path.open("rb") as stream:
        raw = tomllib.load(stream)

    found: list[Suppression] = []
    ruff = raw.get("tool", {}).get("ruff", {}).get("lint", {})

    ignored = ruff.get("ignore", [])
    if ignored:
        found.append(
            Suppression(
                path=PurePosixPath(f"{_PYPROJECT}[lint.ignore]"),
                kind="ruff-global-ignore",
                count=len(ignored),
            )
        )

    for pattern, codes in sorted(ruff.get("per-file-ignores", {}).items()):
        found.append(
            Suppression(
                path=PurePosixPath(f"{_PYPROJECT}[per-file-ignores:{pattern}]"),
                kind="ruff-per-file-ignore",
                count=len(codes),
            )
        )

    pyright = raw.get("tool", {}).get("pyright", {})
    disabled = sum(
        1
        for key, value in pyright.items()
        if key.startswith("report") and str(value).lower() in _DISABLED_VALUES
    )
    if disabled:
        found.append(
            Suppression(
                path=PurePosixPath(f"{_PYPROJECT}[tool.pyright]"),
                kind="pyright-global-rule",
                count=disabled,
            )
        )
    return tuple(found)


def suppressions(
    root: Path, *, paths: Iterable[str] | None = None
) -> tuple[Suppression, ...]:
    """Return every file's use of every escape hatch, most-used first.

    `paths` narrows the walk to named files. Tokenising and parsing the whole tree
    twice is the ratchet's dominant cost, and a file only gains a suppression by
    being edited. Configuration entries are always included: they are not keyed by
    a source file.
    """
    found: list[Suppression] = []
    for path in _support.python_files(root, paths=paths):
        source = path.read_text(encoding="utf-8", errors="replace")
        counts = comment_suppressions(source)
        with contextlib.suppress(SyntaxError):
            counts.update(call_suppressions(ast.parse(source, filename=str(path))))
        relative = PurePosixPath(path.relative_to(root).as_posix())
        found.extend(
            Suppression(path=relative, kind=kind, count=count)
            for kind, count in sorted(counts.items())
        )
    found.extend(config_suppressions(root))
    return tuple(
        sorted(found, key=lambda item: (-item.count, str(item.path), item.kind))
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    found = suppressions(root)
    totals: dict[str, int] = {}
    for item in found:
        totals[item.kind] = totals.get(item.kind, 0) + item.count
    receipt = {
        "files_with_suppressions": len({item.path for item in found}),
        "status": "REPORT",
        "total": sum(totals.values()),
        "totals_by_kind": dict(sorted(totals.items())),
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    for item in found[:20]:
        print(f"{item.path}: {item.kind} x{item.count}", file=sys.stderr)
    # Always zero. Nothing here is a violation on its own; the ratchet decides
    # whether a candidate reached for more escape hatches than its base did.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
