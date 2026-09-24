"""Check that every test directory names an existing module.

The rule (CLAUDE.md section 4, task tests-path-correspondence-discipline):

    A directory under tests/ that holds test_*.py must correspond to a real module
    directory in the source tree. Correspondence is at module level -- file names
    are unconstrained. When a reserved segment appears in the path, that segment
    and everything below it is exempt, but the prefix before it must still
    correspond.

Reserved segments are the only exemption mechanism. There is deliberately no
baseline or allow-list of existing violations: an exemption that means "no module
matched" would let the rule decay into describing whatever the tree already is.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

# Reserved segments name test groupings that legitimately span modules, so no single
# module owns them. THIS LIST IS OWNED BY THE USER. Adding a segment is an
# architecture decision and needs explicit user consent, for the same reason the
# .importlinter contracts do: a writer who may widen the exemption will
# reach for it whenever something is awkward to place, and the rule stops biting.
RESERVED_SEGMENTS: Final = frozenset({"contract", "parity"})

# Test directories live under lib/zcu_tools/ unless their top-level segment names a
# different source root. This is a mapping, not an exemption: the target must exist.
_DEFAULT_SOURCE_ROOT: Final = PurePosixPath("lib/zcu_tools")
_SOURCE_ROOT_OVERRIDES: Final = {
    "script": PurePosixPath("script"),
    "tools": PurePosixPath("tools"),
}

_TESTS_DIR: Final = "tests"
_TEST_FILE_GLOB: Final = "test_*.py"


@dataclass(frozen=True)
class Violation:
    """A test directory whose path names no existing module."""

    test_dir: PurePosixPath
    expected_source_dir: PurePosixPath


def expected_source_dir(relative: PurePosixPath) -> PurePosixPath | None:
    """Return the module directory `relative` must correspond to.

    `relative` is a test directory relative to tests/. Returns None when the path is
    exempt, which happens when a reserved segment appears with no prefix before it.
    """
    effective: list[str] = []
    for segment in relative.parts:
        if segment in RESERVED_SEGMENTS:
            break
        effective.append(segment)

    if not effective:
        return None

    head, *rest = effective
    override = _SOURCE_ROOT_OVERRIDES.get(head)
    if override is None:
        return _DEFAULT_SOURCE_ROOT.joinpath(*effective)
    return override.joinpath(*rest)


def test_directories(root: Path) -> tuple[PurePosixPath, ...]:
    """Return every directory under tests/ that holds at least one test module."""
    tests = root / _TESTS_DIR
    if not tests.is_dir():
        return ()
    found = {
        PurePosixPath(path.parent.relative_to(tests).as_posix())
        for path in tests.rglob(_TEST_FILE_GLOB)
        if path.is_file()
    }
    return tuple(sorted(found))


def violations(root: Path) -> tuple[Violation, ...]:
    """Return every test directory that corresponds to no existing module."""
    found = []
    for relative in test_directories(root):
        expected = expected_source_dir(relative)
        if expected is None or (root / expected).is_dir():
            continue
        found.append(
            Violation(
                test_dir=PurePosixPath(_TESTS_DIR).joinpath(relative),
                expected_source_dir=expected,
            )
        )
    return tuple(found)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    found = violations(root)
    receipt = {
        "reserved_segments": sorted(RESERVED_SEGMENTS),
        "status": "FAIL" if found else "PASS",
        "test_directories": len(test_directories(root)),
        "violation_count": len(found),
        "violations": [
            {
                "expected_source_dir": str(item.expected_source_dir),
                "test_dir": str(item.test_dir),
            }
            for item in found
        ],
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    for item in found:
        print(
            f"{item.test_dir}: no module at {item.expected_source_dir}",
            file=sys.stderr,
        )
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
