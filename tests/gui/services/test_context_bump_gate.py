"""Structural gate: writing md/ml must bump the ``context`` resource version.

A past concurrency bug was a *hidden contract* — "every code path that
mutates MetaDict/ModuleLibrary content must ``version.bump("context")``" — that
lived only in comments and was physically scattered across ContextService (8
methods), the three context-switch methods, and WritebackService. A new md/ml
writer that forgets the bump silently reopens that bug; no behavioural test
catches it because the data write still "works".

This AST gate makes the contract structural (mirroring
``test_app_service_decoupling``'s import-discipline gate): it scans the two
modules that write md/ml directly and asserts that any function whose body
performs an md/ml write call also contains a ``version.bump("context")`` call.

It checks *presence*, not correctness of the guard condition — exactly the
altitude that catches that failure mode ("forgot to bump at all")
without coupling to control flow. Reads (``getattr``) and pure field swaps
(``dataclasses.replace`` in ``set_context``) contain no write marker and are
not flagged.
"""

from __future__ import annotations

import ast
from pathlib import Path

import zcu_tools.gui.services as services_pkg

_SERVICES_DIR = Path(services_pkg.__file__).parent

# Modules that mutate MetaDict / ModuleLibrary *content* directly. ContextService
# is the canonical owner; WritebackService writes ctx.md/ctx.ml without going
# through it (ADR-0008: delegating would create an app-service inter-dependency).
_MD_ML_WRITER_MODULES = ("context.py", "writeback.py")

# ML write methods (called as ``<something>.<name>(...)``).
_ML_WRITE_METHODS = frozenset(
    {"register_module", "delete_module", "register_waveform", "delete_waveform"}
)
# MD writes go through the builtins setattr/delattr on a ``md``-named target.
_MD_WRITE_BUILTINS = frozenset({"setattr", "delattr"})


def _is_md_target(arg: ast.expr) -> bool:
    """True if ``arg`` names the MetaDict (``md`` or ``*.md``)."""
    if isinstance(arg, ast.Name):
        return arg.id == "md"
    if isinstance(arg, ast.Attribute):
        return arg.attr == "md"
    return False


def _is_md_ml_write(node: ast.Call) -> bool:
    func = node.func
    # ml.register_module(...) / ctx.ml.delete_waveform(...) etc.
    if isinstance(func, ast.Attribute) and func.attr in _ML_WRITE_METHODS:
        return True
    # setattr(md, ...) / delattr(ctx.md, ...)
    if (
        isinstance(func, ast.Name)
        and func.id in _MD_WRITE_BUILTINS
        and node.args
        and _is_md_target(node.args[0])
    ):
        return True
    return False


def _is_context_bump(node: ast.Call) -> bool:
    """True for ``<...>.version.bump("context")``."""
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == "bump"):
        return False
    if not (isinstance(func.value, ast.Attribute) and func.value.attr == "version"):
        return False
    if len(node.args) != 1:
        return False
    arg = node.args[0]
    return isinstance(arg, ast.Constant) and arg.value == "context"


def _calls_in(fn: ast.FunctionDef) -> list[ast.Call]:
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call)]


def test_md_ml_writers_bump_context_version():
    offenders: dict[str, list[str]] = {}
    for module in _MD_ML_WRITER_MODULES:
        path = _SERVICES_DIR / module
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            calls = _calls_in(fn)
            if not any(_is_md_ml_write(c) for c in calls):
                continue  # function writes no md/ml content
            if not any(_is_context_bump(c) for c in calls):
                offenders.setdefault(module, []).append(fn.name)

    assert not offenders, (
        "every function that writes MetaDict/ModuleLibrary content must "
        'self.version.bump("context") (hidden-contract gate). '
        f"Missing the bump: {offenders}"
    )


def test_gate_detects_a_missing_bump():
    """The gate must actually fail when a writer omits the bump (guards against
    a vacuously-passing matcher that recognises no write at all)."""
    src = (
        "def writer(self):\n"
        "    ml = self._state.exp_context.ml\n"
        "    ml.register_module(foo=bar)\n"  # write, but no bump
    )
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    calls = _calls_in(fn)
    assert any(_is_md_ml_write(c) for c in calls)
    assert not any(_is_context_bump(c) for c in calls)


def test_gate_recognises_a_correct_writer():
    src = (
        "def writer(self):\n"
        "    md = self._state.exp_context.md\n"
        "    setattr(md, key, value)\n"
        '    self._state.version.bump("context")\n'
    )
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    calls = _calls_in(fn)
    assert any(_is_md_ml_write(c) for c in calls)
    assert any(_is_context_bump(c) for c in calls)


def test_gate_ignores_reads_and_pure_swaps():
    """getattr reads and dataclasses.replace swaps are not md/ml writes."""
    src = (
        "def reader(self):\n"
        "    md = self._state.exp_context.md\n"
        "    current = getattr(md, key, None)\n"
        "    new_ctx = dataclasses.replace(self._state.exp_context, md=md)\n"
        "    return current\n"
    )
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    calls = _calls_in(fn)
    assert not any(_is_md_ml_write(c) for c in calls)
