import math
import warnings

import pytest
from qick.asm_v2 import QickParam, QickProgramV2
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modules.util import (
    calc_max_length,
    get_fclk,
    merge_max_length,
    round_timestamp,
)


def test_get_fclk():
    prog = QickProgramV2(make_mock_soccfg())
    assert get_fclk(prog, gen_ch=0) == 384.0
    assert get_fclk(prog, ro_ch=0) == 307.2
    assert get_fclk(prog) == 430.08

    with pytest.raises(RuntimeError, match="can't specify both gen_ch and ro_ch!"):
        get_fclk(prog, gen_ch=0, ro_ch=0)


def test_round_timestamp():
    prog = QickProgramV2(make_mock_soccfg())  # fclk = 430.08
    # 0.001 * 430.08 = 0.43008 cycle
    assert math.isclose(round_timestamp(prog, 0.001, take_ceil=True), 1.0 / 430.08)
    assert math.isclose(round_timestamp(prog, 0.001, take_ceil=False), 0.0 / 430.08)

    # QickParam
    param = QickParam(start=0.001, spans={"a": 0.005})
    rounded = round_timestamp(prog, param, take_ceil=True)
    assert isinstance(rounded, QickParam)
    assert math.isclose(rounded.start, 1.0 / 430.08)
    assert math.isclose(
        rounded.spans["a"], 3.0 / 430.08
    )  # 0.005 * 430.08 = 2.1504 -> ceil 3.0


def test_calc_max_length():
    assert calc_max_length(1.0, 2.0) == 2.0
    assert calc_max_length(3.0, 1.5) == 3.0

    param1 = QickParam(start=1.0, spans={"a": 2.0})  # maxval = 3.0
    param2 = QickParam(start=2.0, spans={"a": -2.0})  # maxval = 2.0

    with pytest.warns(UserWarning, match="using the maximum length for calculation"):
        res = calc_max_length(param1, param2)
        assert res == 3.0


def test_merge_max_length():
    assert merge_max_length(1.0) == 1.0
    assert merge_max_length(1.0, 2.0, 3.0) == 3.0

    with pytest.raises(ValueError, match="at least one length must be provided"):
        merge_max_length()

    param1 = QickParam(start=1.0, spans={"a": 1.0})  # maxval 2.0
    param2 = QickParam(start=1.5, spans={"b": 2.0})  # maxval 3.5
    param3 = QickParam(start=0.5, spans={"c": 0.1})  # maxval 0.6

    with pytest.warns(UserWarning, match="Using the maximum length among them"):
        res = merge_max_length(param1, param2, param3)
        assert res == 3.5
