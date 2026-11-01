from __future__ import annotations

import pytest
from qick import QickConfig

from tests.program._profiles import load_soccfg


@pytest.fixture
def mock_soccfg() -> QickConfig:
    return load_soccfg()
