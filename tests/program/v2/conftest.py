from __future__ import annotations

from tests.program._profiles import load_soccfg
import pytest
from qick import QickConfig



@pytest.fixture
def mock_soccfg() -> QickConfig:
    return load_soccfg()
