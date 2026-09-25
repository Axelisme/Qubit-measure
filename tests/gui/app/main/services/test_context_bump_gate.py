"""Main WritebackService md/ml writers retain the context version bump.

The shared matcher also retains the original three small checks below. Session's
ContextService has its own scan under tests/gui/session/services/.
"""

from __future__ import annotations

import ast
from pathlib import Path

import zcu_tools.gui.app.main.services as services_pkg

from tests.gui._context_bump_gate import (
    calls_in,
    is_context_bump,
    is_md_ml_write,
    missing_context_bumps,
)


def test_md_ml_writers_bump_context_version() -> None:
    assert services_pkg.__file__ is not None
    path = Path(services_pkg.__file__).parent / "writeback.py"
    offenders = missing_context_bumps(path)
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
    calls = calls_in(fn)
    assert any(is_md_ml_write(c) for c in calls)
    assert not any(is_context_bump(c) for c in calls)


def test_gate_recognises_a_correct_writer():
    src = (
        "def writer(self):\n"
        "    md = self._state.exp_context.md\n"
        "    setattr(md, key, value)\n"
        '    self._state.version.bump("context")\n'
    )
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    calls = calls_in(fn)
    assert any(is_md_ml_write(c) for c in calls)
    assert any(is_context_bump(c) for c in calls)


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
    calls = calls_in(fn)
    assert not any(is_md_ml_write(c) for c in calls)
