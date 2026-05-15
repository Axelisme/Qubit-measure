from __future__ import annotations

import pytest
from zcu_tools.program.v2.ir.instructions import NopInst
from zcu_tools.program.v2.ir.node import BasicBlockNode, BlockNode, IRLoop
from zcu_tools.program.v2.ir.operands import Register
from zcu_tools.program.v2.ir.passes.loop.unroll import UnrollLoopPass
from zcu_tools.program.v2.ir.pipeline import PipeLineConfig, PipeLineContext


def _run(loop: IRLoop, ctx: PipeLineContext) -> None:
    """Run UnrollLoopPass.transform on a single IRLoop (no child_chunks needed)."""
    UnrollLoopPass().transform(loop, [], ctx)


def test_unroll_rejects_non_general_counter():
    # s11 is a system register, not a general-purpose register.
    loop = IRLoop(
        name="bad_loop",
        counter_reg=Register("s11"),
        n=5,
        body=BlockNode(insts=[BasicBlockNode(insts=[NopInst()])]),
    )
    ctx = PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)

    with pytest.raises(ValueError, match="is not a general-purpose register"):
        _run(loop, ctx)


def test_unroll_rejects_n_reg_equal_counter_reg():
    loop = IRLoop(
        name="bad_loop",
        counter_reg=Register("r1"),
        n=Register("r1"),
        body=BlockNode(insts=[BasicBlockNode(insts=[NopInst()])]),
    )
    ctx = PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)

    with pytest.raises(ValueError, match="n_reg and counter_reg are the same register"):
        _run(loop, ctx)


def test_unroll_rejects_n_reg_conflict_s15():
    # s15 is reserved for absolute jumps in unroll logic.
    loop = IRLoop(
        name="bad_loop",
        counter_reg=Register("r1"),
        n=Register("s15"),
        body=BlockNode(insts=[BasicBlockNode(insts=[NopInst()])]),
    )
    ctx = PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)

    with pytest.raises(
        ValueError, match="conflicts with reserved address register s15"
    ):
        _run(loop, ctx)


def test_unroll_accepts_valid_registers():
    loop = IRLoop(
        name="good_loop",
        counter_reg=Register("r1"),
        n=Register("r2"),
        body=BlockNode(insts=[BasicBlockNode(insts=[NopInst()])]),
    )
    ctx = PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)

    # Should not raise
    _run(loop, ctx)
