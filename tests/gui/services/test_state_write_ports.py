"""State satisfies the narrow tab state ports (ADR-0026 §3).

The run / analyze policies depend on ``RunStatePort`` / ``AnalyzeStatePort``
rather than the concrete ``State``. These are structural ``runtime_checkable``
Protocols, so ``State`` must satisfy them without any inheritance change — this
test locks that contract so a future State refactor that drops/renames one of
the narrowed methods fails loudly here.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from zcu_tools.gui.app.main.services.ports import (
    AnalyzeStatePort,
    RunStatePort,
    TabAnalyzeWritePort,
    TabResultWritePort,
)
from zcu_tools.gui.app.main.state import State


def test_state_satisfies_tab_result_write_port() -> None:
    state = State(MagicMock())
    assert isinstance(state, TabResultWritePort)


def test_state_satisfies_tab_analyze_write_port() -> None:
    state = State(MagicMock())
    assert isinstance(state, TabAnalyzeWritePort)


def test_state_satisfies_run_state_port() -> None:
    state = State(MagicMock())
    assert isinstance(state, RunStatePort)


def test_state_satisfies_analyze_state_port() -> None:
    state = State(MagicMock())
    assert isinstance(state, AnalyzeStatePort)
