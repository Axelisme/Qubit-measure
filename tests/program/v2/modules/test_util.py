import math

import pytest
from qick.asm_v2 import QickParam, QickProgramV2
from zcu_tools.program.v2.modules.util import (
    get_fclk,
    round_timestamp,
)

from tests.program._profiles import load_soccfg


def test_get_fclk():
    prog = QickProgramV2(load_soccfg())
    assert get_fclk(prog, gen_ch=0) == prog.soccfg["gens"][0]["f_fabric"]
    assert get_fclk(prog, ro_ch=0) == prog.soccfg["readouts"][0]["f_output"]
    assert get_fclk(prog) == prog.soccfg["tprocs"][0]["f_time"]

    with pytest.raises(RuntimeError, match="can't specify both gen_ch and ro_ch!"):
        get_fclk(prog, gen_ch=0, ro_ch=0)


def test_round_timestamp():
    prog = QickProgramV2(load_soccfg())
    cycle = 1 / prog.soccfg["tprocs"][0]["f_time"]
    timestamp = 0.4 * cycle
    assert math.isclose(round_timestamp(prog, timestamp, take_ceil=True), cycle)
    assert round_timestamp(prog, timestamp, take_ceil=False) == 0

    # Fractional start and span ceil independently onto the selected clock.
    param = QickParam(start=timestamp, spans={"a": 2.2 * cycle})
    rounded = round_timestamp(prog, param, take_ceil=True)
    assert isinstance(rounded, QickParam)
    assert not isinstance(rounded, float)
    assert not isinstance(rounded, int)
    assert math.isclose(rounded.start, cycle)
    assert math.isclose(rounded.spans["a"], 3 * cycle)
