"""Report test-module coupling, overwritten definitions, and naming hints.

Only statically resolved repository imports and unconditional duplicate definitions
block. Naming hints need human review. No test modules are imported or executed.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import _support

_HISTORY_NAME = re.compile(
    r"(?:^|_)(?:ticket_?[a-z]*\d+|phase_?\d+|part_?\d+|[bc]\d+|misc)(?:_|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Finding:
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


def _import_findings(root: Path, path: Path, tree: ast.Module) -> list[Finding]:
    found: list[Finding] = []
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
                Finding(
                    path.relative_to(root).as_posix(),
                    node.lineno,
                    rule,
                    f"Import {relative.as_posix()}; share helpers through a support module",
                )
            )
    return found


def _duplicates(body: list[ast.stmt], path: str) -> list[Finding]:
    """Inspect collection namespaces, without flattening conditional branches."""
    found: list[Finding] = []
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
                Finding(
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


def findings(root: Path) -> tuple[Finding, ...]:
    """Read tests only; propagate syntax and IO errors rather than reporting PASS."""
    paths = (
        path.relative_to(root).as_posix() for path in (root / "tests").rglob("*.py")
    )
    found: list[Finding] = []
    for path in _support.python_files(root, paths=paths):
        relative = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        found.extend(_import_findings(root, path, tree))
        if path.name.startswith("test_"):
            found.extend(_duplicates(tree.body, relative))
            if _HISTORY_NAME.search(path.stem.removeprefix("test_")):
                found.append(
                    Finding(
                        relative,
                        1,
                        "test-file-name",
                        "Review whether the filename names a stable domain or contract",
                        "advisory",
                    )
                )
    return tuple(sorted(found, key=lambda item: (item.path, item.line, item.rule)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    arguments = parser.parse_args(argv)
    try:
        root = arguments.root.resolve(strict=True)
        if not (root / "tests").is_dir():
            raise ValueError(f"No tests directory at {root}")
        found = findings(root)
    except (OSError, SyntaxError, UnicodeError, ValueError) as error:
        print(f"test structure check failed: {error}", file=sys.stderr)
        return 2
    blocking = sum(item.severity == "blocking" for item in found)
    print(
        json.dumps(
            {
                "status": "FAIL" if blocking else "PASS",
                "violation_count": blocking,
                "advisory_count": len(found) - blocking,
                "findings": [asdict(item) for item in found],
            },
            indent=2,
            sort_keys=True,
        )
    )
    for item in found:
        print(
            f"{item.path}:{item.line}: {item.severity} {item.rule}: {item.message}",
            file=sys.stderr,
        )
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
