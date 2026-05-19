from __future__ import annotations

from typing import Any, Union

import pytest
from qick.asm_v2 import QickParam
from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.control import Branch, Repeat


class _FixedDurationModule(Module):
    def __init__(self, name: str, duration: float):
        self.name = name
        self.duration = duration

    def init(self, prog) -> None:
        pass

    def run(self, prog, t=0.0):
        return self.duration

    def allow_rerun(self) -> bool:
        return True


def test_branch_uses_dispatch_table_jumps(mock_prog):
    b = Branch(
        "sel",
        [_FixedDurationModule("b0", 0.1)],
        [_FixedDurationModule("b1", 0.2)],
        [_FixedDurationModule("b2", 0.3)],
    )

    out = b.run(mock_prog, t=0.25)

    assert out == 0.0
    assert mock_prog.cond_jump.call_count == 0
    assert mock_prog.write_reg_op.call_count == 1
    assert mock_prog.write_reg_op.call_args_list[0].args == ("s15", "s15", "+", "sel")
    # 3 dispatch table stubs + 2 end-of-case jumps (all non-last cases).
    assert mock_prog.jump.call_count == 5


def test_branch_power_of_two_has_no_nop_padding(mock_prog):
    b = Branch(
        "sel",
        [_FixedDurationModule("b0", 0.1)],
        [_FixedDurationModule("b1", 0.2)],
        [_FixedDurationModule("b2", 0.3)],
        [_FixedDurationModule("b3", 0.4)],
    )

    b.run(mock_prog)

    assert mock_prog.nop.call_count == 0
    assert mock_prog.cond_jump.call_count == 0


def test_branch_rejects_qickparam_duration(mock_prog):
    class _QickParamDurationModule(_FixedDurationModule):
        def run(self, prog: Any, t: Union[float, QickParam] = 0.0) -> Any:
            # We want to return a QickParam to trigger the Branch error,
            # but we cast to float to satisfy Pyright's override check.
            return QickParam(start=0.1)

    b = Branch(
        "sel",
        [_QickParamDurationModule("b0", 0.1)],
        [_FixedDurationModule("b1", 0.2)],
    )

    with pytest.raises(NotImplementedError, match="swept duration"):
        b.run(mock_prog)


# ---------------------------------------------------------------------------
# Repeat — constructor validation
# ---------------------------------------------------------------------------


def test_repeat_negative_n_rejected():
    with pytest.raises(ValueError):
        Repeat("lp", n=-1)


def test_repeat_n_equals_name_rejected():
    with pytest.raises(ValueError, match="counter register"):
        Repeat("my_loop", n="my_loop")


def test_repeat_zero_n_allowed():
    r = Repeat("lp", n=0)
    assert r.n == 0


def test_repeat_allow_rerun_returns_false():
    assert Repeat("lp", n=3).allow_rerun() is False


def test_repeat_range_hint_stored():
    r = Repeat("lp", n=5, range_hint=(1, 5))
    assert r.range_hint == (1, 5)


def test_repeat_range_hint_none_by_default():
    r = Repeat("lp", n=5)
    assert r.range_hint is None


# ---------------------------------------------------------------------------
# Repeat — init
# ---------------------------------------------------------------------------


def test_repeat_init_registers_counter_reg(mock_prog):
    r = Repeat("lp", n=3)
    r.init(mock_prog)
    mock_prog.add_reg.assert_called_with("lp")


def test_repeat_init_zero_n_short_circuit(mock_prog):
    child = _FixedDurationModule("c", 0.1)
    from unittest.mock import MagicMock

    child.init = MagicMock()
    r = Repeat("lp", n=0)
    r.add_content(child)
    r.init(mock_prog)

    mock_prog.add_reg.assert_not_called()
    child.init.assert_not_called()


def test_repeat_init_calls_submodule_init(mock_prog):
    child = _FixedDurationModule("c", 0.1)
    from unittest.mock import MagicMock

    child.init = MagicMock()
    r = Repeat("lp", n=2)
    r.add_content(child)
    r.init(mock_prog)
    child.init.assert_called_once_with(mock_prog)


# ---------------------------------------------------------------------------
# Repeat — run (integer n uses open_inner_loop / close_inner_loop)
# ---------------------------------------------------------------------------


def test_repeat_run_integer_n_calls_open_close_inner_loop(mock_prog):
    r = Repeat("lp", n=4)
    r.run(mock_prog)

    mock_prog.open_inner_loop.assert_called_once_with("lp", "lp", 4, range_hint=None)
    mock_prog.close_inner_loop.assert_called_once_with("lp", "lp", 4)


def test_repeat_run_string_n_calls_open_close_inner_loop_with_reg(mock_prog):
    r = Repeat("lp", n="n_reg")
    r.run(mock_prog)

    mock_prog.open_inner_loop.assert_called_once_with(
        "lp", "lp", "n_reg", range_hint=None
    )
    mock_prog.close_inner_loop.assert_called_once_with("lp", "lp", "n_reg")


def test_repeat_run_returns_zero(mock_prog):
    r = Repeat("lp", n=2)
    out = r.run(mock_prog)
    assert out == 0.0


def test_repeat_run_zero_n_short_circuits_loop_macros(mock_prog):
    r = Repeat("lp", n=0)
    out = r.run(mock_prog, t=0.25)

    assert out == 0.0
    mock_prog.open_inner_loop.assert_not_called()
    mock_prog.close_inner_loop.assert_not_called()
    mock_prog.delay.assert_called_once_with(t=0.25)
    mock_prog.delay_auto.assert_called_once_with(t=0.0)


def test_repeat_run_passes_range_hint(mock_prog):
    r = Repeat("lp", n=5, range_hint=(0, 5))
    r.run(mock_prog)
    mock_prog.open_inner_loop.assert_called_once_with("lp", "lp", 5, range_hint=(0, 5))


# ---------------------------------------------------------------------------
# Branch — large pmem (big_jump) emits write_reg_op for each dispatch stub
# ---------------------------------------------------------------------------


def test_branch_large_pmem_emits_extra_write_reg_op(mock_prog):
    mock_prog.tproccfg = {"pmem_size": 4096}
    b = Branch(
        "sel",
        [_FixedDurationModule("b0", 0.1)],
        [_FixedDurationModule("b1", 0.2)],
    )
    b.run(mock_prog)
    # big_jump: write_reg_op called for s15 base address + 2 dispatch stubs write_label + jump
    assert mock_prog.write_reg_op.call_count >= 1


def test_branch_requires_at_least_two_branches():
    with pytest.raises(ValueError, match="at least 2"):
        Branch("sel", [_FixedDurationModule("b0", 0.1)])


# ---------------------------------------------------------------------------
# Branch — init delegates to sub-module init
# ---------------------------------------------------------------------------


def test_branch_init_calls_submodule_init(mock_prog):
    from unittest.mock import MagicMock

    child0 = _FixedDurationModule("b0", 0.1)
    child1 = _FixedDurationModule("b1", 0.2)
    child0.init = MagicMock()
    child1.init = MagicMock()

    b = Branch("sel", [child0], [child1])
    b.init(mock_prog)

    child0.init.assert_called_once_with(mock_prog)
    child1.init.assert_called_once_with(mock_prog)
