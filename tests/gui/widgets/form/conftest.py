"""Shared external timer control for FormEditor behaviour tests."""

import pytest
from zcu_tools.gui.widgets.form import value_widgets

from tests.gui.widgets.form._clock import Clock


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    """Control external QTimer scheduling without replacing commit behaviour."""
    clock = Clock()
    monkeypatch.setattr(value_widgets, "QTimer", clock.timer)
    return clock
