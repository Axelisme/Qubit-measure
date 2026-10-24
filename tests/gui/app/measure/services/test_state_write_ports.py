"""State satisfies the remaining analyze state ports (ADR-0066)."""

from __future__ import annotations

from unittest.mock import MagicMock

from zcu_tools.gui.app.measure.services.ports import (
    AnalyzeStatePort,
)
from zcu_tools.gui.app.measure.state import State


def test_state_satisfies_analyze_state_port() -> None:
    state = State(MagicMock())
    assert isinstance(state, AnalyzeStatePort)
