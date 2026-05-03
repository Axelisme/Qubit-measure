from __future__ import annotations

from typing import Any, Union, cast

from qick.asm_v2 import QickParam
from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.control import Branch


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


def test_branch_uses_binary_dispatch_cond_jump(mock_prog):
    b = Branch(
        "sel",
        [_FixedDurationModule("b0", 0.1)],
        [_FixedDurationModule("b1", 0.2)],
        [_FixedDurationModule("b2", 0.3)],
    )

    out = b.run(mock_prog, t=0.25)

    assert out == 0.0
    assert mock_prog.cond_jump.call_count == 2

    mids = sorted(call.args[4] for call in mock_prog.cond_jump.call_args_list)
    assert mids == [1, 2]
    assert all(call.args[1] == "sel" for call in mock_prog.cond_jump.call_args_list)
    assert all(call.args[2] == "S" for call in mock_prog.cond_jump.call_args_list)
    assert all(call.args[3] == "-" for call in mock_prog.cond_jump.call_args_list)


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


def test_branch_rejects_qickparam_duration(mock_prog):
    class _QickParamDurationModule(_FixedDurationModule):
        def run(self, prog: Any, t: Union[float, QickParam] = 0.0) -> float:
            # We want to return a QickParam to trigger the Branch error,
            # but we cast to float to satisfy Pyright's override check.
            return cast(float, QickParam(start=0.1))

    b = Branch(
        "sel",
        [_QickParamDurationModule("b0", 0.1)],
        [_FixedDurationModule("b1", 0.2)],
    )

    try:
        b.run(mock_prog)
    except NotImplementedError as exc:
        assert "swept duration" in str(exc)
    else:
        raise AssertionError(
            "Expected NotImplementedError for QickParam branch duration"
        )
