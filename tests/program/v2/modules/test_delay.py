import pytest
from zcu_tools.program.v2.ir.builder import IRBuilder
from zcu_tools.program.v2.modules.delay import Delay, DelayAuto, Join, SoftDelay
from zcu_tools.program.v2.modules.pulse import Pulse


def test_softdelay_returns_rounded_delay(mock_prog):
    sd = SoftDelay("d", 0.1)
    out = sd.ir_run(IRBuilder(), t=0.0, prog=mock_prog)
    fclk = 430.08
    import numpy as np

    assert out == int(np.ceil(0.1 * fclk)) / fclk


def test_delay_auto_with_tag_rejects_reg_name():
    with pytest.raises(ValueError):
        DelayAuto("d", t="reg_name", tag="k")


def test_join_empty_rejected():
    with pytest.raises(ValueError):
        Join()


def test_join_rejects_delay_children():
    with pytest.raises(ValueError):
        Join(Delay("d", 0.1))


