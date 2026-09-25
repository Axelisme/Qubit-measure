"""Shared AST matcher for the existing context-version source checks."""

from __future__ import annotations

import ast
from pathlib import Path

_ML_WRITE_METHODS = frozenset(
    {"register_module", "delete_module", "register_waveform", "delete_waveform"}
)
_MD_WRITE_BUILTINS = frozenset({"setattr", "delattr"})


def _is_md_target(arg: ast.expr) -> bool:
    """True if ``arg`` names the MetaDict (``md`` or ``*.md``)."""
    return (isinstance(arg, ast.Name) and arg.id == "md") or (
        isinstance(arg, ast.Attribute) and arg.attr == "md"
    )


def is_md_ml_write(node: ast.Call) -> bool:
    func = node.func
    return (isinstance(func, ast.Attribute) and func.attr in _ML_WRITE_METHODS) or (
        isinstance(func, ast.Name)
        and func.id in _MD_WRITE_BUILTINS
        and bool(node.args)
        and _is_md_target(node.args[0])
    )


def is_context_bump(node: ast.Call) -> bool:
    """True for ``<...>.version.bump(\"context\")``."""
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == "bump"):
        return False
    if not (isinstance(func.value, ast.Attribute) and func.value.attr == "version"):
        return False
    if len(node.args) != 1:
        return False
    arg = node.args[0]
    return isinstance(arg, ast.Constant) and arg.value == "context"


def calls_in(fn: ast.FunctionDef) -> list[ast.Call]:
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call)]


def missing_context_bumps(path: Path) -> dict[str, list[str]]:
    """Report direct md/ml writers in a module without a context version bump."""
    offenders: dict[str, list[str]] = {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        calls = calls_in(fn)
        if not any(is_md_ml_write(c) for c in calls):
            continue
        if not any(is_context_bump(c) for c in calls):
            offenders.setdefault(path.name, []).append(fn.name)
    return offenders
