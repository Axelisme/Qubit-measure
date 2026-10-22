"""Check lab author await boundaries without importing or executing lab modules.

This is an AST guardrail, not a type checker: framework receivers use the literal
names env/tab/session/scope/iteration. Assignment aliases, dynamic dispatch and wrappers are not
tracked. Direct imports (including relative imports) resolve against definitions,
so an external callable is never accepted just for looking like a lab helper.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Final

from _support import SKIPPED_DIRS, dotted_name, import_bindings

_FORBIDDEN: Final = frozenset({"create_task", "gather", "wait_for", "TaskGroup"})
_RECEIVERS: Final = frozenset({"env", "tab", "session", "scope", "iteration"})


@dataclass(frozen=True)
class Violation:
    """One rejected expression: repo-relative path, one-based line and source target.

    target is the awaited expression or a forbidden direct scheduling call.
    """

    path: str
    line: int
    target: str


def _module_name(path: str) -> str:
    parts = PurePosixPath(path).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _bindings(tree: ast.Module, path: str) -> dict[str, str]:
    bindings = import_bindings(tree)
    module = _module_name(path)
    package = (
        module if PurePosixPath(path).stem == "__init__" else module.rpartition(".")[0]
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            parent = package.split(".")[: len(package.split(".")) - node.level + 1]
            prefix = ".".join([*parent, *([node.module] if node.module else [])])
            for alias in node.names:
                bindings[alias.asname or alias.name] = f"{prefix}.{alias.name}"
    return bindings


def _definitions(tree: ast.AST, prefix: str) -> set[str]:
    found: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef, ast.ClassDef)):
            qualified = f"{prefix}.{node.name}"
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found.add(qualified)
            found.update(_definitions(node, qualified))
        else:
            found.update(_definitions(node, prefix))
    return found


def _imported_target(name: str, bindings: Mapping[str, str]) -> str:
    head, separator, rest = name.partition(".")
    return bindings.get(head, head) + (separator + rest if separator else "")


def _forbidden(name: str) -> bool:
    return name.split(".")[0] == "asyncio" or name.split(".")[-1] in _FORBIDDEN


class _AwaitChecker(ast.NodeVisitor):
    def __init__(
        self, path: str, bindings: Mapping[str, str], definitions: set[str]
    ) -> None:
        self.path = path
        self.bindings = bindings
        self.definitions = definitions
        self.scope = [_module_name(path)]
        self.async_depth = 0
        self.violations: list[Violation] = []

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.async_depth += 1
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()
        self.async_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_Await(self, node: ast.Await) -> None:
        if self.async_depth and not self._allowed(node.value):
            self.violations.append(
                Violation(self.path, node.lineno, ast.unparse(node.value))
            )
        # A forbidden awaited call is already reported above. Still inspect its
        # arguments for additional direct scheduling calls.
        if isinstance(node.value, ast.Call):
            for child in ast.iter_child_nodes(node.value):
                self.visit(child)
        else:
            self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = dotted_name(node.func)
        if (
            self.async_depth
            and name
            and _forbidden(_imported_target(name, self.bindings))
        ):
            self.violations.append(Violation(self.path, node.lineno, ast.unparse(node)))
        self.generic_visit(node)

    def _allowed(self, expression: ast.expr) -> bool:
        if not isinstance(expression, ast.Call):
            return False
        name = dotted_name(expression.func)
        if not name:
            return False
        resolved = _imported_target(name, self.bindings)
        if _forbidden(resolved):
            return False
        head, separator, _rest = name.partition(".")
        if separator and head in _RECEIVERS:
            return True
        if head in self.bindings:
            return resolved in self.definitions
        return any(
            f"{'.'.join(self.scope[:depth])}.{name}" in self.definitions
            for depth in range(len(self.scope), 0, -1)
        )


def check_sources(sources: Mapping[str, str]) -> tuple[Violation, ...]:
    """Check repo-relative Python sources beneath zcu_lab using one uniform rule.

    Every async function is checked, whether decorated, called or unused. Await
    accepts env/tab/session/scope/iteration calls or statically resolved lab function definitions,
    whether defined with def or async def. Within async functions, direct asyncio
    calls and create_task/gather/wait_for/TaskGroup are rejected even without await.
    Other async contexts and iterators are not banned.
    Non-lab paths are ignored. SyntaxError propagates; source is never executed.
    Results are sorted by path and line. See the module docstring for limits.
    """
    trees = {
        path: ast.parse(source, filename=path)
        for path, source in sources.items()
        if PurePosixPath(path).parts[0] == "zcu_lab" and path.endswith(".py")
    }
    definitions: set[str] = set()
    for path, tree in trees.items():
        definitions.update(_definitions(tree, _module_name(path)))
    violations: list[Violation] = []
    for path, tree in trees.items():
        checker = _AwaitChecker(path, _bindings(tree, path), definitions)
        checker.visit(tree)
        violations.extend(checker.violations)
    return tuple(
        sorted(violations, key=lambda item: (item.path, item.line, item.target))
    )


def main(argv: list[str] | None = None) -> int:
    """Scan --root/zcu_lab and print a JSON receipt plus stderr diagnostics.

    --root defaults to this checkout. Exit 0 means no violations, 1 means rejected
    expressions, and 2 means source could not be read or parsed. No imports run.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    root = parser.parse_args(argv).root
    try:
        sources = {
            path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
            for path in sorted((root / "zcu_lab").rglob("*.py"))
            if SKIPPED_DIRS.isdisjoint(path.relative_to(root).parts)
        }
        violations = check_sources(sources)
    except (OSError, SyntaxError) as error:
        print(f"script awaits failed: {error}", file=sys.stderr)
        return 2
    for item in violations:
        print(f"{item.path}:{item.line}: rejected {item.target}", file=sys.stderr)
    print(
        json.dumps(
            {
                "violation_count": len(violations),
                "violations": [asdict(v) for v in violations],
            }
        )
    )
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
