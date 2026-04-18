import pytest

from zcu_tools.program.v2.modules.delay import Delay, DelayAuto, Join, SoftDelay
from zcu_tools.program.v2.modules.pulse import Pulse


def test_softdelay_returns_rounded_delay(mock_prog):
    sd = SoftDelay("d", 0.1)
    out = sd.run(mock_prog, t=0.0)
    fclk = 430.08
    import numpy as np

    assert out == int(np.ceil(0.1 * fclk)) / fclk


def test_softdelay_allows_rerun():
    assert SoftDelay("d", 0.0).allow_rerun() is True


def test_delay_tagged_blocks_rerun():
    assert Delay("d", 0.1, tag="k").allow_rerun() is False


def test_delay_untagged_allows_rerun():
    assert Delay("d", 0.1).allow_rerun() is True


def test_delay_auto_with_tag_rejects_reg_name():
    with pytest.raises(ValueError):
        DelayAuto("d", t="reg_name", tag="k")


def test_join_empty_rejected():
    with pytest.raises(ValueError):
        Join()


def test_join_rejects_delay_children():
    with pytest.raises(ValueError):
        Join(Delay("d", 0.1))


def test_join_allow_rerun_aggregates():
    p = Pulse("p", None)
    j = Join(p)
    assert j.allow_rerun() is True
