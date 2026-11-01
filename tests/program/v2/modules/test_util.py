from tests.program._profiles import load_soccfg
import math

import pytest
from qick.asm_v2 import QickParam, QickProgramV2

from zcu_tools.program.v2.modules.util import (
    get_fclk,
    round_timestamp,
)


def test_get_fclk():
    prog = QickProgramV2(load_soccfg())
    assert get_fclk(prog, gen_ch=0) == 384.0
    assert get_fclk(prog, ro_ch=0) == 307.2
    assert get_fclk(prog) == 430.08

    with pytest.raises(RuntimeError, match="can't specify both gen_ch and ro_ch!"):
        get_fclk(prog, gen_ch=0, ro_ch=0)


def test_round_timestamp():
    prog = QickProgramV2(load_soccfg())  # fclk = 430.08
    # 0.001 * 430.08 = 0.43008 cycle
    assert math.isclose(round_timestamp(prog, 0.001, take_ceil=True), 1.0 / 430.08)
    assert math.isclose(round_timestamp(prog, 0.001, take_ceil=False), 0.0 / 430.08)

    # QickParam
    param = QickParam(start=0.001, spans={"a": 0.005})
    rounded = round_timestamp(prog, param, take_ceil=True)
    assert isinstance(rounded, QickParam)
    assert not isinstance(rounded, float)
    assert not isinstance(rounded, int)
    assert math.isclose(rounded.start, 1.0 / 430.08)
    assert math.isclose(
        rounded.spans["a"], 3.0 / 430.08
    )  # 0.005 * 430.08 = 2.1504 -> ceil 3.0
