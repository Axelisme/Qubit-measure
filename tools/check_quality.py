"""Measure repository quality without importing the sources being checked.

Subcommands report file size, test paths, test structure, test capabilities and
suppression usage. Each owns its JSON receipt and stderr diagnostics. The ratchet
compares these measurements against a base rather than imposing zero counts.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import re
import sys
import tokenize
import tomllib
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal

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
    "apps": PurePosixPath("apps"),
    "migrations": PurePosixPath("migrations"),
    "mcp_servers": PurePosixPath("mcp_servers"),
    "scripts": PurePosixPath("scripts"),
    "tools": PurePosixPath("tools"),
    "zcu_lab": PurePosixPath("zcu_lab"),
}

_TESTS_DIR: Final = "tests"
_TEST_FILE_GLOB: Final = "test_*.py"


@dataclass(frozen=True)
class PathViolation:
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


def path_violations(root: Path) -> tuple[PathViolation, ...]:
    """Return every test directory that corresponds to no existing module."""
    found = []
    for relative in test_directories(root):
        expected = expected_source_dir(relative)
        if expected is None or (root / expected).is_dir():
            continue
        found.append(
            PathViolation(
                test_dir=PurePosixPath(_TESTS_DIR).joinpath(relative),
                expected_source_dir=expected,
            )
        )
    return tuple(found)


_HISTORY_NAME = re.compile(
    r"(?:^|_)(?:ticket_?[a-z]*\d+|phase_?\d+|part_?\d+|[bc]\d+|misc)(?:_|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class StructureFinding:
    """One structural diagnostic at repo-relative path and one-based line.

    rule identifies the check and message explains it. severity is blocking
    (affects exit status) or advisory (reported without failing the check).
    """

    path: str
    line: int
    rule: str
    message: str
    severity: Literal["blocking", "advisory"] = "blocking"


def _module_target(root: Path, parts: tuple[str, ...]) -> Path | None:
    """Resolve only explicit repository paths, not arbitrary sys.path entries."""
    if not parts:
        return None
    target = root.joinpath(*parts)
    package = target / "__init__.py"
    if package.is_file():
        return package
    source = target.with_suffix(".py")
    return source if source.is_file() else None


def _import_targets(
    root: Path, path: Path, node: ast.Import | ast.ImportFrom
) -> set[Path]:
    if isinstance(node, ast.Import):
        names = [tuple(alias.name.split(".")) for alias in node.names]
    else:
        prefix: tuple[str, ...] = ()
        if node.level:
            parent = path.relative_to(root).parent.parts
            if node.level > len(parent):
                return set()
            prefix = parent[: len(parent) - node.level + 1]
        module = prefix + tuple(node.module.split(".") if node.module else ())
        # Imported names may be package attributes, even when matching files
        # exist. Only the explicit module is certain without executing imports.
        names = [module]
    return {
        target for name in names if (target := _module_target(root, name)) is not None
    }


def _import_findings(
    root: Path, path: Path, tree: ast.Module
) -> list[StructureFinding]:
    found: list[StructureFinding] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for target in sorted(_import_targets(root, path, node)):
            relative = target.relative_to(root)
            if relative.parts[0] != "tests":
                continue
            if target.name == "conftest.py":
                rule = "conftest-import"
            elif target.name.startswith("test_") and target != path:
                rule = "test-module-import"
            else:
                continue
            found.append(
                StructureFinding(
                    path.relative_to(root).as_posix(),
                    node.lineno,
                    rule,
                    f"Import {relative.as_posix()}; share helpers through a support module",
                )
            )
    return found


def _duplicates(body: list[ast.stmt], path: str) -> list[StructureFinding]:
    """Inspect collection namespaces, without flattening conditional branches."""
    found: list[StructureFinding] = []
    seen: dict[str, int] = {}
    for node in body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        is_test = (
            node.name.startswith("Test")
            if isinstance(node, ast.ClassDef)
            else node.name.startswith("test_")
        )
        if not is_test:
            continue
        if node.name in seen:
            found.append(
                StructureFinding(
                    path,
                    node.lineno,
                    "duplicate-test-definition",
                    f"{node.name} overwrites definition at line {seen[node.name]}",
                )
            )
        seen[node.name] = node.lineno
        if isinstance(node, ast.ClassDef):
            found.extend(_duplicates(node.body, path))
    return found


def structure_findings(root: Path) -> tuple[StructureFinding, ...]:
    """Read tests only; propagate syntax and IO errors rather than reporting PASS."""
    paths = (
        path.relative_to(root).as_posix() for path in (root / "tests").rglob("*.py")
    )
    found: list[StructureFinding] = []
    for path in _support.python_files(root, paths=paths):
        relative = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        found.extend(_import_findings(root, path, tree))
        if path.name.startswith("test_"):
            found.extend(_duplicates(tree.body, relative))
            if _HISTORY_NAME.search(path.stem.removeprefix("test_")):
                found.append(
                    StructureFinding(
                        relative,
                        1,
                        "test-file-name",
                        "Review whether the filename names a stable domain or contract",
                        "advisory",
                    )
                )
    return tuple(sorted(found, key=lambda item: (item.path, item.line, item.rule)))


# capability -> (marker that declares it, fully qualified callables that use it)
CAPABILITIES: Final = {
    "wall-clock": ("uses_wall_clock", ("time.sleep", "asyncio.sleep")),
    "loopback": (
        "requires_loopback",
        (
            "socket.socket",
            "socket.create_connection",
            "socket.create_server",
            "socket.socketpair",
        ),
    ),
    "subprocess": (
        "requires_subprocess",
        (
            "subprocess.run",
            "subprocess.Popen",
            "subprocess.call",
            "subprocess.check_call",
            "subprocess.check_output",
            "os.system",
            "os.popen",
            "multiprocessing.Process",
        ),
    ),
}


@dataclass(frozen=True)
class UndeclaredCapability:
    """A test module that uses a capability without declaring its marker."""

    module: PurePosixPath
    capability: str
    marker: str
    uses: int


def capability_uses(tree: ast.AST) -> dict[str, int]:
    """Return {capability: number of call sites} for one parsed module."""
    wanted = {
        dotted: capability
        for capability, (_, dotted_names) in CAPABILITIES.items()
        for dotted in dotted_names
    }
    bindings = _support.import_bindings(tree)
    found: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        qualified = _support.resolve_call(node.func, bindings)
        capability = wanted.get(qualified) if qualified else None
        if capability is not None:
            found[capability] = found.get(capability, 0) + 1
    return found


def declared_markers(tree: ast.AST) -> set[str]:
    """Return every pytest marker named anywhere in one parsed module."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        dotted = _support.dotted_name(node)
        if dotted is not None and dotted.startswith("pytest.mark."):
            found.add(dotted.removeprefix("pytest.mark."))
    return found


def undeclared_capabilities(
    root: Path, *, paths: Iterable[str] | None = None
) -> tuple[UndeclaredCapability, ...]:
    """Return every test module using a capability it does not declare.

    `paths` narrows the walk to named files. The unit here is the module, and a
    module only gains a capability by being edited, so the ratchet passes the
    candidate's changed files rather than re-reading every test in the tree.
    """
    tests = root / _TESTS_DIR
    if not tests.is_dir():
        return ()
    if paths is None:
        candidates = sorted(tests.rglob("test_*.py"))
    else:
        candidates = [
            root / name
            for name in sorted(paths)
            if name.startswith(f"{_TESTS_DIR}/")
            and name.rpartition("/")[2].startswith("test_")
        ]
    found: list[UndeclaredCapability] = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        markers = declared_markers(tree)
        for capability, uses in sorted(capability_uses(tree).items()):
            marker = CAPABILITIES[capability][0]
            if marker in markers:
                continue
            found.append(
                UndeclaredCapability(
                    module=PurePosixPath(path.relative_to(root).as_posix()),
                    capability=capability,
                    marker=marker,
                    uses=uses,
                )
            )
    return tuple(found)


# Inline directives start a hash-delimited segment, not necessarily the token.
# File-level Pyright settings only work at the start of the comment token.
# Keep the hash in each pattern so prose that only mentions the words is skipped.
COMMENT_SUPPRESSIONS: Final = {
    "noqa": re.compile(r"#\s*noqa", re.IGNORECASE),
    "type-ignore": re.compile(r"#\s*type:\s*ignore"),
    "pyright-ignore": re.compile(r"#\s*pyright:\s*ignore"),
    # `# pyright: reportFoo=false` turns a rule off for a whole file, which is the
    # broadest silencer available and the one most worth seeing grow.
    "pyright-file-rule": re.compile(r"^#\s*pyright:\s*\w+\s*=\s*(?:false|none)", re.I),
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


def comment_directives(comment: str) -> dict[str, re.Match[str]]:
    """Return the first suppression match of each kind in a COMMENT token.

    Inline directives must start a hash-delimited segment; file-level Pyright
    settings must start the token itself. Match offsets refer to the original
    token so callers can parse bracketed scope and trailing text.
    """
    found: dict[str, re.Match[str]] = {}
    for kind, pattern in COMMENT_SUPPRESSIONS.items():
        match = pattern.search(comment)
        if match is not None:
            found[kind] = match
    return found


def comment_suppressions(source: str) -> dict[str, int]:
    """Return {kind: count} over the comments of one module."""
    found: dict[str, int] = {}
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    try:
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue
            for kind in comment_directives(token.string):
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


def _file_size(root: Path) -> tuple[dict[str, object], int]:
    found = oversize_files(root)
    receipt: dict[str, object] = {
        "limit": LINE_LIMIT,
        "status": "FAIL" if found else "PASS",
        "violation_count": len(found),
        "violations": [{"lines": item.lines, "path": str(item.path)} for item in found],
    }
    for item in found:
        print(f"{item.path}: {item.lines} lines > {LINE_LIMIT}", file=sys.stderr)
    return receipt, 1 if found else 0


def _test_paths(root: Path) -> tuple[dict[str, object], int]:
    found = path_violations(root)
    receipt: dict[str, object] = {
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
    for item in found:
        print(
            f"{item.test_dir}: no module at {item.expected_source_dir}",
            file=sys.stderr,
        )
    return receipt, 1 if found else 0


def _test_structure(root: Path) -> tuple[dict[str, object], int]:
    try:
        root = root.resolve(strict=True)
        if not (root / "tests").is_dir():
            raise ValueError(f"No tests directory at {root}")
        found = structure_findings(root)
    except (OSError, SyntaxError, UnicodeError, ValueError) as error:
        print(f"test structure check failed: {error}", file=sys.stderr)
        return {}, 2
    blocking = sum(item.severity == "blocking" for item in found)
    receipt: dict[str, object] = {
        "status": "FAIL" if blocking else "PASS",
        "violation_count": blocking,
        "advisory_count": len(found) - blocking,
        "findings": [asdict(item) for item in found],
    }
    for item in found:
        print(
            f"{item.path}:{item.line}: {item.severity} {item.rule}: {item.message}",
            file=sys.stderr,
        )
    return receipt, 1 if blocking else 0


def _test_capabilities(root: Path) -> tuple[dict[str, object], int]:
    found = undeclared_capabilities(root)
    receipt: dict[str, object] = {
        "capabilities": {name: marker for name, (marker, _) in CAPABILITIES.items()},
        "status": "FAIL" if found else "PASS",
        "violation_count": len(found),
        "violations": [
            {
                "capability": item.capability,
                "marker": item.marker,
                "module": str(item.module),
                "uses": item.uses,
            }
            for item in found
        ],
    }
    for item in found:
        print(
            f"{item.module}: uses {item.capability} "
            f"({item.uses} call sites) without @pytest.mark.{item.marker}",
            file=sys.stderr,
        )
    return receipt, 1 if found else 0


def _suppressions(root: Path) -> tuple[dict[str, object], int]:
    found = suppressions(root)
    totals: dict[str, int] = {}
    for item in found:
        totals[item.kind] = totals.get(item.kind, 0) + item.count
    receipt: dict[str, object] = {
        "files_with_suppressions": len({item.path for item in found}),
        "status": "REPORT",
        "total": sum(totals.values()),
        "totals_by_kind": dict(sorted(totals.items())),
    }
    for item in found[:20]:
        print(f"{item.path}: {item.kind} x{item.count}", file=sys.stderr)
    # Always zero. Nothing here is a violation on its own; the ratchet decides
    # whether a candidate reached for more escape hatches than its base did.
    return receipt, 0


_CHECKS: Final = {
    "file-size": _file_size,
    "test-paths": _test_paths,
    "test-structure": _test_structure,
    "test-capabilities": _test_capabilities,
    "suppressions": _suppressions,
}


def main(argv: list[str] | None = None) -> int:
    """Print the selected quality receipt and stderr diagnostics for this checkout.

    None reads sys.argv. Each check returns 0 or 1; test-structure also returns 2
    for invalid roots/read/parse failures. Suppression usage alone returns 0.
    all nests the five receipts (an empty object on structure failure), emits
    diagnostics in subcommand order and returns the highest exit code.
    argparse handles help and invalid syntax with SystemExit.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (*_CHECKS, "all"):
        subparser = subparsers.add_parser(name, help=f"Report {name} measurements")
        if name == "test-structure":
            subparser.add_argument(
                "--root",
                type=Path,
                default=Path(__file__).resolve().parents[1],
                help="checkout containing tests (default: this checkout)",
            )
    arguments = parser.parse_args(argv)
    root = (
        arguments.root
        if arguments.command == "test-structure"
        else Path(__file__).resolve().parents[1]
    )
    if arguments.command == "all":
        receipts = {}
        code = 0
        for name, check in _CHECKS.items():
            receipt, check_code = check(root)
            receipts[name] = receipt
            code = max(code, check_code)
        print(json.dumps(receipts, indent=2, sort_keys=True))
        return code
    receipt, code = _CHECKS[arguments.command](root)
    if not receipt:
        return code
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
