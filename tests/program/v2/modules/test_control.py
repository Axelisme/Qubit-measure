from __future__ import annotations

from qick.asm_v2 import QickParam
from zcu_tools.program.v2.ir.builder import IRBuilder
from zcu_tools.program.v2.ir.nodes import IRBranch, IRDelay, IRDelayAuto, IRLoop, IRSeq
from zcu_tools.program.v2.lower import Emitter
from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.control import Branch, Repeat, SoftRepeat


class _FixedDurationModule(Module):
    def __init__(self, name: str, duration: float):
        self.name = name
        self.duration = duration

    def init(self, prog) -> None:
        pass

    def ir_run(self, builder, t: float | QickParam, prog) -> float | QickParam:
        return self.duration


def test_branch_uses_binary_dispatch_cond_jump(mock_prog):
    b = Branch(
        "sel",
        [_FixedDurationModule("b0", 0.1)],
        [_FixedDurationModule("b1", 0.2)],
        [_FixedDurationModule("b2", 0.3)],
    )

    bld = IRBuilder()
    out = b.ir_run(bld, t=0.25, prog=mock_prog)
    Emitter(mock_prog).emit(bld.build())

    assert out == 0.0
    assert mock_prog.cond_jump.call_count == 2

    mids = sorted(call.kwargs["arg2"] for call in mock_prog.cond_jump.call_args_list)
    assert mids == [1, 2]
    assert all(call.args[1] == "sel" for call in mock_prog.cond_jump.call_args_list)
    assert all(call.args[2] == "S" for call in mock_prog.cond_jump.call_args_list)
    assert all(call.kwargs["op"] == "-" for call in mock_prog.cond_jump.call_args_list)


def test_branch_does_not_pad_nop_in_module_ir(mock_prog):
    b = Branch(
        "sel",
        [_FixedDurationModule("b0", 0.1)],
        [_FixedDurationModule("b1", 0.2)],
        [_FixedDurationModule("b2", 0.3)],
    )

    bld = IRBuilder()
    b.ir_run(bld, t=0.0, prog=mock_prog)
    Emitter(mock_prog).emit(bld.build())

    # Branch-path balancing is handled by IR passes, not module ir_run.
    assert mock_prog.nop.call_count == 0


def test_branch_power_of_two_has_no_nop_padding(mock_prog):
    b = Branch(
        "sel",
        [_FixedDurationModule("b0", 0.1)],
        [_FixedDurationModule("b1", 0.2)],
        [_FixedDurationModule("b2", 0.3)],
        [_FixedDurationModule("b3", 0.4)],
    )

    bld = IRBuilder()
    b.ir_run(bld, t=0.0, prog=mock_prog)
    Emitter(mock_prog).emit(bld.build())

    assert mock_prog.nop.call_count == 0


def test_branch_rejects_qickparam_duration(mock_prog):
    class _QickParamDurationModule(_FixedDurationModule):
        def ir_run(self, builder, t: float | QickParam = 0.0, prog=None):
            return QickParam(start=0.1)

    b = Branch(
        "sel",
        [_QickParamDurationModule("b0", 0.1)],
        [_FixedDurationModule("b1", 0.2)],
    )

    try:
        b.ir_run(IRBuilder(), t=0.0, prog=mock_prog)
    except NotImplementedError as exc:
        assert "swept duration" in str(exc)
    else:
        raise AssertionError("Expected NotImplementedError for QickParam branch duration")


def test_repeat_ir_run_emits_loop(mock_prog):
    r = Repeat("r", 3).add_content(_FixedDurationModule("d", 0.2))
    b = IRBuilder()

    out = r.ir_run(b, t=0.25, prog=mock_prog)
    root = b.build()

    assert out == 0.0
    assert isinstance(root, IRSeq)
    assert isinstance(root.body[0], IRDelay)
    assert isinstance(root.body[1], IRDelayAuto)
    assert isinstance(root.body[2], IRLoop)


def test_soft_repeat_ir_run_unrolls(mock_prog):
    s = SoftRepeat("sr", 2).add_content(_FixedDurationModule("d", 0.1))
    b = IRBuilder()

    out = s.ir_run(b, t=0.0, prog=mock_prog)
    root = b.build()

    assert out == 0.1
    assert isinstance(root, IRSeq)
    assert len(root.body) == 0


def test_branch_ir_run_emits_ir_branch_and_final_delay_auto(mock_prog):
    bmod = Branch(
        "sel",
        [_FixedDurationModule("b0", 0.1)],
        [_FixedDurationModule("b1", 0.2)],
        [_FixedDurationModule("b2", 0.3)],
    )
    b = IRBuilder()

    out = bmod.ir_run(b, t=0.25, prog=mock_prog)
    root = b.build()

    assert out == 0.0
    assert isinstance(root, IRSeq)
    assert isinstance(root.body[0], IRDelay)
    assert isinstance(root.body[1], IRDelayAuto)
    assert isinstance(root.body[2], IRBranch)
    assert isinstance(root.body[3], IRDelayAuto)
    assert len(root.body[2].arms) == 3
