import numpy as np
from qick.asm_v2 import QickParam
from zcu_tools.program.v2.modules.util import (
    calc_max_length,
    get_fclk,
    merge_max_length,
    round_timestamp,
)


def test_get_fclk_tproc(mock_prog):
    assert get_fclk(mock_prog) == 430.08


def test_round_timestamp_float_ceil(mock_prog):
    t = 1.234  # us
    out = round_timestamp(mock_prog, t, take_ceil=True)
    fclk = 430.08
    expected = int(np.ceil(t * fclk)) / fclk
    assert out == expected


def test_round_timestamp_float_floor(mock_prog):
    t = 1.234
    out = round_timestamp(mock_prog, t, take_ceil=False)
    fclk = 430.08
    expected = int(np.floor(t * fclk)) / fclk
    assert out == expected


def test_round_timestamp_qickparam(mock_prog):
    qp = QickParam(start=1.0, spans={"s": 2.5})
    out = round_timestamp(mock_prog, qp)
    assert isinstance(out, QickParam)
    fclk = 430.08
    assert out.start == int(np.ceil(1.0 * fclk)) / fclk
    assert out.spans["s"] == int(np.ceil(2.5 * fclk)) / fclk


def test_calc_max_length_strictly_larger():
    assert calc_max_length(2.0, 1.0) == 2.0
    assert calc_max_length(0.5, 1.5) == 1.5


def test_merge_max_length_single():
    assert merge_max_length(1.0) == 1.0


def test_merge_max_length_reduces_comparable():
    assert merge_max_length(1.0, 2.0, 0.5) == 2.0
