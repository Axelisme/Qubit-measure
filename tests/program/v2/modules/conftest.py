from unittest.mock import MagicMock

import pytest

_FCLK = 430.08


def _make_ch_entry(f_fabric=_FCLK):
    return {"f_fabric": f_fabric}


def _make_ro_entry(f_output=_FCLK):
    return {"f_output": f_output}


@pytest.fixture
def mock_prog():
    prog = MagicMock()
    prog.soccfg = {
        "tprocs": [{"f_time": _FCLK}],
        # provide entries for channels 0-4 so run() tests don't KeyError
        "gens": [_make_ch_entry() for _ in range(5)],
        "readouts": [_make_ro_entry() for _ in range(5)],
    }
    return prog
