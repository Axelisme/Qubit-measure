"""Shared pieces of the checks in this directory.

The checks stay independent in what they judge; this holds only the mechanics
they would otherwise each reimplement -- walking the tree, reading an attribute
chain, resolving a call through the module's imports, and loading a sibling
check.

Import resolution in particular is worth sharing rather than copying. Two checks
need to know that `from time import sleep` and `import typing as t` both bind a
name to something elsewhere, and a copy in each would mean every future
correction had to be made twice.

`tools/` is expected to be on sys.path. It is, three ways: running a check
directly puts its own directory there, `load_tool` relies on the same, and
pytest's `pythonpath` lists it.
"""

from __future__ import annotations

import ast
import importlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Final

# Where repository Python lives. `script` and `tools` sit at the root; everything
# else the checks care about is under lib/ or tests/.
CHECKED_ROOTS: Final = ("lib", "tests", "script", "tools")

# Generated or vendored trees. Nothing here is written by hand, so no check has
# an opinion about it.
SKIPPED_DIRS: Final = frozenset({"__pycache__", ".venv", ".pytest-tmp"})


def python_files(root: Path, *, paths: Iterable[str] | None = None) -> list[Path]:
    """Return the Python files a check should read.

    `paths` narrows the walk to named files, which is how the ratchet keeps its
    cost proportional to the change rather than to the repository. A named path
    that does not exist is skipped: a file added by the candidate is absent from
    the base tree, and one deleted by it is absent from the candidate.

    Skipping is decided on the path relative to `root`. An absolute path carries
    the directories above the repository, which are none of a check's business
    and would otherwise decide what gets skipped.
    """
    if paths is not None:
        candidates = [root / name for name in sorted(paths)]
    else:
        candidates = []
        for checked in CHECKED_ROOTS:
            base = root / checked
            if base.is_dir():
                candidates.extend(sorted(base.rglob("*.py")))
    return [
        path
        for path in candidates
        if path.is_file() and SKIPPED_DIRS.isdisjoint(path.relative_to(root).parts)
    ]


def dotted_name(node: ast.expr) -> str | None:
    """Return `a.b.c` for an attribute chain rooted in a plain name, else None."""
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def import_bindings(tree: ast.AST) -> dict[str, str]:
    """Return {local name: fully qualified name} for one module's imports.

    `import time` binds time -> time, `import time as t` binds t -> time, and
    `from time import sleep as nap` binds nap -> time.sleep. Relative imports are
    skipped: they name modules inside the repository, which own none of the
    callables these checks look for.
    """
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                bindings[local] = alias.name if alias.asname else local
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            for alias in node.names:
                bindings[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return bindings


def resolve_call(func: ast.expr, bindings: Mapping[str, str]) -> str | None:
    """Return the fully qualified name a call targets, as far as imports say.

    A name the module never imported resolves to None rather than to itself: a
    local function called `cast` is not typing.cast.
    """
    dotted = dotted_name(func)
    if dotted is None:
        return None
    head, _, rest = dotted.partition(".")
    qualified = bindings.get(head)
    if qualified is None:
        return None
    return f"{qualified}.{rest}" if rest else qualified


def load_tool(name: str) -> ModuleType:
    """Return a sibling check as a module, so one check can reuse another's query."""
    return importlib.import_module(name)
