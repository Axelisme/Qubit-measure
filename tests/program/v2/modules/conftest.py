from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_prog():
    prog = MagicMock()
    prog.soccfg = {"tprocs": [{"f_time": 430.08}]}
    return prog
