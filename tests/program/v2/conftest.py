from __future__ import annotations

import pytest
from qick import QickConfig
from zcu_tools.program.v2 import make_mock_soccfg


@pytest.fixture
def mock_soccfg() -> QickConfig:
    return make_mock_soccfg()
