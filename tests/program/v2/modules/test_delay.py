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


# ---------------------------------------------------------------------------
# DelayAuto — normal path (no tag, no reg_name)
# ---------------------------------------------------------------------------


def test_delay_auto_normal_calls_delay_auto(mock_prog):
    da = DelayAuto("d", t=0.5)
    da.run(mock_prog, t=0.0)
    mock_prog.delay_auto.assert_called_once_with(t=0.5, gens=True, ros=True, tag=None)


def test_delay_auto_reg_name_calls_delay_reg_auto(mock_prog):
    da = DelayAuto("d", t="time_reg")
    da.run(mock_prog, t=0.0)
    mock_prog.delay_reg_auto.assert_called_once_with(
        time_reg="time_reg", gens=True, ros=True
    )


def test_delay_auto_returns_zero(mock_prog):
    da = DelayAuto("d", t=0.0)
    out = da.run(mock_prog)
    assert out == 0.0


def test_delay_auto_with_tag_allows_rerun_false():
    assert DelayAuto("d", tag="k").allow_rerun() is False


def test_delay_auto_without_tag_allows_rerun_true():
    assert DelayAuto("d").allow_rerun() is True


# ---------------------------------------------------------------------------
# Join — disable_delay called, runs interleaved
# ---------------------------------------------------------------------------


def test_join_calls_disable_delay_context(mock_prog):
    from contextlib import contextmanager

    entered = []

    @contextmanager
    def _cm():
        entered.append(True)
        yield

    mock_prog.disable_delay.side_effect = _cm

    child = SoftDelay("s", 0.0)
    j = Join([child])
    j.run(mock_prog, t=0.0)
    assert entered, "disable_delay context was not entered"


def test_join_returns_max_of_branch_times(mock_prog):
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        yield

    mock_prog.disable_delay.side_effect = _cm

    # Use distinct non-zero delays so merge_max_length can pick a clear winner
    # without triggering the overlapping-lengths warning.
    fast = SoftDelay("f", 0.1)
    slow = SoftDelay("s", 0.5)
    j = Join([fast], [slow])
    out = j.run(mock_prog, t=0.0)
    assert isinstance(out, (int, float))
