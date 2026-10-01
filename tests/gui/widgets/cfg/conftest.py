"""Fixtures shared by cfg form widget tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.event_bus import BaseEventBus as EventBus


@pytest.fixture()
def ctrl():
    c = MagicMock()
    c.get_bus.return_value = EventBus()
    c.get_current_md.return_value = MagicMock()
    c.get_current_ml.return_value = MagicMock()
    c.arb_waveforms.list_data_keys.return_value = []
    return c
